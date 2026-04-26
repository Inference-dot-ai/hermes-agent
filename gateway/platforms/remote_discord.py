"""Remote Discord adapter — receives Discord Gateway events via HTTP.

Architecture:
- A central Cloudflare Durable Object holds the shared bot's Gateway WS.
- It dispatches events to a Worker route, which routes per-customer to this VM.
- This adapter listens on http://0.0.0.0:8645 (reverse-proxied at /api/* on
  the customer's exe.dev VM) and feeds events into hermes' agent loop.
- Outbound responses POST to the Worker's /api/discord/send (which calls
  Discord REST with the customer's webhook persona).

See https://github.com/Inference-dot-ai/hermes-agent for context — this fork
exists to add the REMOTE_DISCORD platform alongside the existing DISCORD one.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Any, Optional

import aiohttp
from aiohttp import web

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8645
DEFAULT_HOST = "0.0.0.0"
CONTRACT_VERSION = 1


# ----------------------------------------------------------------------- routing

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("true", "1", "yes", "on")


def _env_csv(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {x.strip() for x in raw.split(",") if x.strip()}


def _is_dm(event: dict[str, Any]) -> bool:
    return not event.get("guild_id")


def _bot_mentioned(event: dict[str, Any], bot_user_id: str) -> bool:
    for m in event.get("mentions", []) or []:
        if str(m.get("id")) == bot_user_id:
            return True
    content = event.get("content") or ""
    return f"<@{bot_user_id}>" in content or f"<@!{bot_user_id}>" in content


def _should_respond(event: dict[str, Any], bot_user_id: str) -> bool:
    """Match hermes' built-in DiscordAdapter mention rules:
    - DMs always respond
    - Server channels: respond on @mention or DISCORD_FREE_RESPONSE_CHANNELS
    - DISCORD_IGNORE_NO_MENTION (default true): skip messages mentioning OTHER users
    """
    if _is_dm(event):
        return True
    free_channels = _env_csv("DISCORD_FREE_RESPONSE_CHANNELS")
    require_mention = _env_bool("DISCORD_REQUIRE_MENTION", True)
    ignore_no_mention = _env_bool("DISCORD_IGNORE_NO_MENTION", True)
    channel_id = str(event.get("channel_id") or "")
    is_free = channel_id in free_channels
    mentioned = _bot_mentioned(event, bot_user_id)
    if not require_mention or is_free:
        return True
    if mentioned:
        return True
    if ignore_no_mention and (event.get("mentions") or []):
        return False
    return False


_BOT_MENTION_RE = re.compile(r"<@!?\d+>")


def _strip_bot_mention(content: str, bot_user_id: str) -> str:
    return (
        content.replace(f"<@{bot_user_id}>", "")
        .replace(f"<@!{bot_user_id}>", "")
        .strip()
    )


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return urlsafe_b64decode(s + pad)


# ----------------------------------------------------------------------- adapter

class RemoteDiscordAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.REMOTE_DISCORD)
        self.port = int(os.getenv("REMOTE_DISCORD_PORT", str(DEFAULT_PORT)))
        self.host = os.getenv("REMOTE_DISCORD_HOST", DEFAULT_HOST)
        self.hmac_secret = os.getenv("REMOTE_DISCORD_HMAC_SECRET", "")
        self.bot_user_id = os.getenv("REMOTE_DISCORD_BOT_USER_ID", "")
        self.outbound_url = os.getenv("REMOTE_DISCORD_OUTBOUND_URL", "")
        self.drr_api_key = os.getenv("OPENAI_API_KEY", "")
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._http: Optional[aiohttp.ClientSession] = None

    # lifecycle

    async def connect(self) -> bool:
        if not self.hmac_secret or not self.bot_user_id or not self.outbound_url:
            self._set_fatal_error(
                "CONFIG_MISSING",
                "REMOTE_DISCORD_HMAC_SECRET, _BOT_USER_ID, and _OUTBOUND_URL are required",
                retryable=False,
            )
            return False
        app = web.Application()
        app.router.add_get("/api/healthcheck", self._handle_health)
        app.router.add_post("/api/discord/event", self._handle_event)
        app.router.add_post("/api/discord/reset", self._handle_reset)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._http = aiohttp.ClientSession()
        self._mark_connected()
        logger.info("[remote-discord] listening on %s:%d", self.host, self.port)
        return True

    async def disconnect(self) -> None:
        if self._http:
            await self._http.close()
            self._http = None
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    # HTTP handlers

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "adapter": "remote_discord",
            "contract_version": CONTRACT_VERSION,
        })

    async def _handle_event(self, request: web.Request) -> web.Response:
        raw = await request.text()
        sig = request.headers.get("x-discord-signature", "")
        if not self._verify_hmac(raw, sig):
            return web.Response(status=401, text="bad signature")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return web.Response(status=400, text="bad json")
        t = payload.get("t")
        d = payload.get("d") or {}
        if t == "MESSAGE_CREATE":
            asyncio.create_task(self._on_message_create(d))
        return web.Response(status=200, text="ok")

    async def _handle_reset(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            logger.info("[remote-discord] reset request for %s", body.get("conversation"))
        except Exception:
            pass
        return web.json_response({"ok": True})

    def _verify_hmac(self, body: str, signature: str) -> bool:
        if not signature or not self.hmac_secret:
            return False
        expected = hmac.new(self.hmac_secret.encode(), body.encode(), hashlib.sha256).digest()
        try:
            given = _b64url_decode(signature)
        except Exception:
            return False
        return hmac.compare_digest(expected, given)

    # event → MessageEvent

    async def _on_message_create(self, d: dict[str, Any]) -> None:
        try:
            if d.get("author", {}).get("bot"):
                return
            if not _should_respond(d, self.bot_user_id):
                return
            content = _strip_bot_mention(d.get("content") or "", self.bot_user_id)
            author = d.get("author") or {}
            channel_id = str(d.get("channel_id") or "")
            chat_type = "dm" if _is_dm(d) else "group"
            source = SessionSource(
                platform=self.platform,
                chat_id=channel_id,
                chat_name=None,
                chat_type=chat_type,
                user_id=str(author.get("id") or ""),
                user_name=author.get("username"),
            )
            event = MessageEvent(
                text=content,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=d,
                message_id=str(d.get("id") or ""),
                reply_to_message_id=str((d.get("referenced_message") or {}).get("id") or "") or None,
                reply_to_text=(d.get("referenced_message") or {}).get("content"),
            )
            await self.handle_message(event)
        except Exception:
            logger.exception("[remote-discord] failed to process MESSAGE_CREATE")

    # outbound

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        if not self._http:
            return SendResult(success=False, error="adapter not connected", retryable=True)
        body = json.dumps({
            "kind": "message",
            "channel_id": chat_id,
            "content": content,
            "persona": True,
        })
        sig = self._sign(body)
        try:
            async with self._http.post(
                self.outbound_url,
                headers={
                    "authorization": f"Bearer {self.drr_api_key}",
                    "content-type": "application/json",
                    "x-discord-signature": sig,
                },
                data=body,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status >= 400:
                    msg = await r.text()
                    return SendResult(success=False, error=f"{r.status}: {msg}", retryable=(r.status >= 500))
                data = await r.json()
                return SendResult(success=True, message_id=data.get("message_id"))
        except asyncio.TimeoutError:
            return SendResult(success=False, error="timeout", retryable=True)
        except aiohttp.ClientError as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": chat_id, "type": "group"}

    def _sign(self, body: str) -> str:
        digest = hmac.new(self.hmac_secret.encode(), body.encode(), hashlib.sha256).digest()
        return urlsafe_b64encode(digest).rstrip(b"=").decode()


def check_remote_discord_requirements() -> bool:
    return aiohttp is not None

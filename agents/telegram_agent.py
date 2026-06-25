"""
TELEGRAM SIGNAL AGENT — real-time listener
Job: Listen to configured private Telegram groups in a background thread.
     The moment a message is posted, analyze it (text + chart image).
     If it's a trade signal, post it to Render immediately.

Unlike the other agents (which poll on a schedule), this one starts once
at orchestrator launch and stays connected 24/7 via Telethon's event system.

Requires:
  TG_API_ID        — from my.telegram.org
  TG_API_HASH      — from my.telegram.org
  TG_CHANNEL_1     — numeric group ID (set by setup_telegram.command)
  TG_CHANNEL_2     — (optional) second group ID
  agents/tg_session — created by setup_telegram.command (one-time login)
"""
from __future__ import annotations
import json, os, base64, threading, time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests

from agents.state import get_state, add_knowledge, post_to_render

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RENDER_URL    = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")

TG_API_ID   = os.environ.get("TG_API_ID", "")
TG_API_HASH = os.environ.get("TG_API_HASH", "")
SESSION_PATH = str(Path.home() / "Desktop" / "CryptoDashboard" / "agents" / "tg_session")

KNOWN_COINS = {"BTC", "ETH", "XRP", "DOGE", "SOL"}

# Runtime state
_listener_thread: threading.Thread | None = None
_listener_running = False
_signals_processed = 0


def _parse_channel(raw: str):
    """Return int ID if numeric, else string @username."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def _get_channels() -> list:
    ch1 = _parse_channel(os.environ.get("TG_CHANNEL_1", ""))
    ch2 = _parse_channel(os.environ.get("TG_CHANNEL_2", ""))
    return [c for c in [ch1, ch2] if c]


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def analyze_message(text: str, image_b64: str | None, channel_id, macro_regime: str) -> dict | None:
    """
    Send message content to Claude for signal extraction.
    Returns a trade signal dict or None.
    """
    client = _claude()
    if not client:
        return None

    # Quick pre-filter: skip pure non-trading text with no image
    trading_kw = {"long", "short", "buy", "sell", "entry", "tp", "sl",
                  "target", "stop", "btc", "eth", "xrp", "sol", "doge",
                  "usdt", "breakout", "support", "resistance", "signal",
                  "trade", "position", "setup"}
    if not image_b64 and not any(kw in (text or "").lower() for kw in trading_kw):
        return None

    content = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            }
        })

    content.append({"type": "text", "text": f"""Telegram trading group message. Macro regime: {macro_regime}

MESSAGE:
{text.strip() if text else "(image only — no text)"}

Is this a trade signal? Extract it if so, return null if not.

Rules:
- Only extract if there is a clear LONG or SHORT direction and at least one price level
- BUY = LONG, SELL = SHORT
- Symbol must end in USDT. Only: BTCUSDT ETHUSDT XRPUSDT DOGEUSDT SOLUSDT
- If entry price is missing, use null (executor uses live price)
- If you see a chart image with a setup drawn on it, describe it and extract levels from the chart

Respond with ONLY:
- null (if no trade signal)
- JSON:
{{
  "symbol": "ETHUSDT",
  "direction": "SHORT",
  "timeframe": "15m",
  "entry": 2450.00,
  "tp": 2380.00,
  "sl": 2490.00,
  "rr_ratio": 1.75,
  "confidence": 7,
  "reason": "<what the channel said and why it looks valid>",
  "source": "telegram",
  "channel_id": "{channel_id}"
}}"""})

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text.strip()

        if raw.lower() in ("null", "none", ""):
            return None
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        sig = json.loads(raw)

        if not sig.get("symbol") or sig.get("direction") not in ("LONG", "SHORT"):
            return None

        sym = sig["symbol"].upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        sig["symbol"] = sym
        return sig

    except Exception as e:
        print(f"[TG AGENT] analyze error: {e}", flush=True)
        return None


def _post_signal(sig: dict):
    """Post a signal to Render and save to knowledge base."""
    global _signals_processed
    _signals_processed += 1

    print(f"[TG AGENT] *** SIGNAL: {sig.get('symbol')} {sig.get('direction')} "
          f"entry={sig.get('entry')} tp={sig.get('tp')} sl={sig.get('sl')} "
          f"rr={sig.get('rr_ratio','?')} conf={sig.get('confidence')}/10", flush=True)

    post_to_render("/api/agent/insight", {
        "type":          "telegram_signal",
        "agent":         "telegram",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "trade_signal":  sig,
        "all_signals":   [sig],
        "market_summary": f"Real-time signal from Telegram group {sig.get('channel_id','')}",
    })

    add_knowledge("telegram_signals", {
        "symbol":    sig.get("symbol"),
        "direction": sig.get("direction"),
        "channel":   sig.get("channel_id"),
        "rr":        sig.get("rr_ratio"),
        "confidence":sig.get("confidence"),
        "reason":    (sig.get("reason") or "")[:100],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _listener_loop():
    """
    Background thread: connects to Telegram and listens for new messages
    in real time. Fires instantly whenever a message is posted.
    Reconnects automatically if the connection drops.
    """
    global _listener_running

    channels = _get_channels()
    if not channels:
        print("[TG AGENT] No channels configured — listener not started", flush=True)
        return

    print(f"[TG AGENT] Starting real-time listener for {len(channels)} group(s)...", flush=True)

    while _listener_running:
        try:
            import asyncio
            from telethon import TelegramClient, events

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            client = TelegramClient(SESSION_PATH, int(TG_API_ID), TG_API_HASH, loop=loop)

            @client.on(events.NewMessage(chats=channels))
            async def handler(event):
                msg      = event.message
                text     = msg.text or ""
                image_b64 = None

                # Download image if attached
                if msg.photo or (msg.document and
                        "image" in (getattr(msg.document, "mime_type", "") or "")):
                    try:
                        img_bytes = await client.download_media(msg, bytes)
                        if img_bytes:
                            image_b64 = base64.b64encode(img_bytes).decode()
                    except Exception as img_err:
                        print(f"[TG AGENT] Image download error: {img_err}", flush=True)

                # Skip empty messages
                if not text and not image_b64:
                    return

                channel_id = event.chat_id
                preview = text[:60].replace("\n", " ") if text else "[image]"
                print(f"[TG AGENT] New message in {channel_id}: {preview}...", flush=True)

                # Get current macro regime for context
                macro_regime = get_state("macro_regime", {}).get("regime_type", "uncertain")

                # Analyze message
                sig = analyze_message(text, image_b64, channel_id, macro_regime)
                if sig:
                    _post_signal(sig)
                else:
                    print(f"[TG AGENT] No signal in message.", flush=True)

            print("[TG AGENT] Connected. Listening for messages...", flush=True)
            with client:
                client.run_until_disconnected()

        except Exception as e:
            if not _listener_running:
                break
            print(f"[TG AGENT] Connection dropped: {e} — reconnecting in 30s...", flush=True)
            time.sleep(30)

    print("[TG AGENT] Listener stopped.", flush=True)


def start_listener():
    """
    Start the real-time listener in a background daemon thread.
    Called once by the orchestrator at startup.
    """
    global _listener_thread, _listener_running

    if not TG_API_ID or not TG_API_HASH:
        print("[TG AGENT] Not configured — run setup_telegram.command first", flush=True)
        return False

    if not Path(SESSION_PATH + ".session").exists():
        print("[TG AGENT] No session file — run setup_telegram.command first", flush=True)
        return False

    channels = _get_channels()
    if not channels:
        print("[TG AGENT] No channels configured", flush=True)
        return False

    _listener_running = True
    _listener_thread = threading.Thread(target=_listener_loop, daemon=True, name="TelegramListener")
    _listener_thread.start()
    return True


def stop_listener():
    """Stop the listener thread gracefully."""
    global _listener_running
    _listener_running = False


def run() -> dict:
    """
    Called by orchestrator scheduler — just returns status.
    The real work happens in the background listener thread.
    """
    alive = _listener_thread is not None and _listener_thread.is_alive()
    print(f"[TG AGENT] Listener status: {'running' if alive else 'NOT running'} | "
          f"Signals processed: {_signals_processed}", flush=True)
    return {
        "listener_running": alive,
        "signals_processed": _signals_processed,
        "channels": _get_channels(),
    }

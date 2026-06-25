"""
TELEGRAM SIGNAL AGENT — runs every 30 minutes
Job: Monitor configured Telegram trading channels.
     Download chart images, read signal text, extract trade setups.
     Feed valid signals into the executor pipeline on Render.

Requires:
  TG_API_ID    — from my.telegram.org
  TG_API_HASH  — from my.telegram.org
  TG_CHANNEL_1 — @username or invite link of first channel
  TG_CHANNEL_2 — (optional) second channel
  tg_session   — created by setup_telegram.command (one-time login)
"""
from __future__ import annotations
import json, os, base64, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import requests

from agents.state import get_state, add_knowledge, post_to_render

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RENDER_URL    = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")

TG_API_ID    = os.environ.get("TG_API_ID", "")
TG_API_HASH  = os.environ.get("TG_API_HASH", "")
TG_CHANNEL_1 = os.environ.get("TG_CHANNEL_1", "")
TG_CHANNEL_2 = os.environ.get("TG_CHANNEL_2", "")

SESSION_PATH = str(Path.home() / "Desktop" / "CryptoDashboard" / "agents" / "tg_session")

# Coins the executor knows about
KNOWN_COINS = {"BTC", "ETH", "XRP", "DOGE", "SOL",
               "BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT"}

# Track which message IDs we've already processed (in-memory, resets on restart)
_seen_msg_ids: set = set()


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def fetch_recent_messages(channel: str, limit: int = 20) -> list:
    """Fetch recent messages from a Telegram channel using Telethon."""
    try:
        from telethon.sync import TelegramClient
        client = TelegramClient(SESSION_PATH, int(TG_API_ID), TG_API_HASH)
        client.connect()

        if not client.is_user_authorized():
            print(f"[TG AGENT] Not logged in — run setup_telegram.command first", flush=True)
            client.disconnect()
            return []

        # Only fetch messages from the last 2 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        messages = []

        for msg in client.iter_messages(channel, limit=limit):
            if msg.date.replace(tzinfo=timezone.utc) < cutoff:
                break
            if msg.id in _seen_msg_ids:
                continue

            entry = {
                "id":       msg.id,
                "channel":  channel,
                "text":     msg.text or "",
                "date":     msg.date.isoformat(),
                "has_image": False,
                "image_b64": None,
            }

            # Download image if present
            if msg.photo or (msg.document and "image" in (getattr(msg.document, "mime_type", "") or "")):
                try:
                    img_bytes = client.download_media(msg, bytes)
                    if img_bytes:
                        entry["has_image"] = True
                        entry["image_b64"] = base64.b64encode(img_bytes).decode()
                except Exception as img_err:
                    print(f"[TG AGENT] Image download error: {img_err}", flush=True)

            messages.append(entry)

        client.disconnect()
        return messages

    except ImportError:
        print("[TG AGENT] Telethon not installed — run setup_telegram.command", flush=True)
        return []
    except Exception as e:
        print(f"[TG AGENT] fetch_recent_messages error ({channel}): {e}", flush=True)
        return []


def analyze_message(msg: dict, macro_regime: str) -> dict | None:
    """
    Use Claude to analyze a Telegram message (text + optional chart image).
    Returns a trade signal dict if a clear signal is found, else None.
    """
    client = _claude()
    if not client:
        return None

    text = msg.get("text", "").strip()
    has_image = msg.get("has_image", False)
    image_b64 = msg.get("image_b64")

    # Skip if no content at all
    if not text and not has_image:
        return None

    # Quick pre-filter: skip if text has no trading-relevant keywords and no image
    trading_keywords = {"long", "short", "buy", "sell", "entry", "tp", "sl",
                        "target", "stop", "btc", "eth", "xrp", "sol", "doge",
                        "usdt", "breakout", "support", "resistance", "signal"}
    if not has_image and not any(kw in text.lower() for kw in trading_keywords):
        return None

    system = """You are a crypto trading signal extractor. Your job is to analyze Telegram messages from trading channels and determine:
1. Is this a real trade signal (entry, TP, SL for a specific coin)?
2. If yes, extract the exact trade parameters.
3. If it's just discussion, news, or non-actionable content — return null.

Be strict: only extract signals that have a clear direction (LONG/SHORT/BUY/SELL), a coin, and at least one price level (entry OR TP OR SL).
If levels are missing, do not guess — return null."""

    content = []

    # Add image if present
    if has_image and image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            }
        })

    # Add text context
    prompt_text = f"""Telegram channel message received at {msg.get('date','')}.

Current macro regime: {macro_regime}

MESSAGE TEXT:
{text if text else "(no text — image only)"}

Analyze this message. If it contains a trade signal, extract it. If not, return null.

Respond with ONLY one of:
- null (if no actionable trade signal)
- A JSON object like:
{{
  "symbol": "ETHUSDT",
  "direction": "SHORT",
  "timeframe": "15m",
  "entry": 2450.00,
  "tp": 2380.00,
  "sl": 2490.00,
  "rr_ratio": 1.75,
  "confidence": 7,
  "reason": "<2 sentences: what the channel said and why it looks valid>",
  "source": "telegram",
  "channel": "{msg.get('channel','')}",
  "raw_text": "{text[:120].replace(chr(10),' ')}"
}}

Rules:
- symbol must end in USDT (e.g. ETHUSDT not ETH)
- direction must be LONG or SHORT
- if the message says BUY = LONG, SELL = SHORT
- if no clear entry price, use null for entry (executor will use live price)
- if rr_ratio cannot be calculated from the levels, estimate it
- only return a signal for coins in: BTC ETH XRP DOGE SOL
- if the chart image shows a setup but text has no levels, describe what you see and set confidence <= 4
"""

    content.append({"type": "text", "text": prompt_text})

    try:
        msg_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        raw = msg_response.content[0].text.strip()

        if raw.lower() in ("null", "none", ""):
            return None

        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        signal = json.loads(raw)

        # Validate required fields
        if not signal.get("symbol") or not signal.get("direction"):
            return None
        if signal["direction"] not in ("LONG", "SHORT"):
            return None

        # Normalize symbol
        sym = signal["symbol"].upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        signal["symbol"] = sym

        return signal

    except Exception as e:
        print(f"[TG AGENT] analyze_message error: {e}", flush=True)
        return None


def run() -> dict:
    """Main agent loop — called by orchestrator every 30 min."""
    print("[TG AGENT] Running...", flush=True)

    if not TG_API_ID or not TG_API_HASH:
        print("[TG AGENT] Not configured — run setup_telegram.command first", flush=True)
        return {}

    channels = [c for c in [TG_CHANNEL_1, TG_CHANNEL_2] if c]
    if not channels:
        print("[TG AGENT] No channels configured", flush=True)
        return {}

    macro_regime = get_state("macro_regime", {}).get("regime_type", "uncertain")

    all_signals = []
    total_msgs  = 0

    for channel in channels:
        print(f"[TG AGENT] Fetching {channel}...", flush=True)
        messages = fetch_recent_messages(channel, limit=30)
        print(f"[TG AGENT]   {len(messages)} new messages", flush=True)
        total_msgs += len(messages)

        for msg in messages:
            _seen_msg_ids.add(msg["id"])

            signal = analyze_message(msg, macro_regime)
            if signal:
                all_signals.append(signal)
                print(f"[TG AGENT]   Signal found: {signal.get('symbol')} {signal.get('direction')} "
                      f"entry={signal.get('entry')} tp={signal.get('tp')} sl={signal.get('sl')} "
                      f"rr={signal.get('rr_ratio','?')} conf={signal.get('confidence')}/10", flush=True)
            else:
                # Brief note for non-signals so we can see the agent is reading
                txt_preview = msg.get("text","")[:60].replace("\n"," ")
                if txt_preview or msg.get("has_image"):
                    print(f"[TG AGENT]   No signal: {'[IMG] ' if msg.get('has_image') else ''}{txt_preview}...", flush=True)

        time.sleep(1)  # be polite to Telegram API

    # Post signals to Render executor pipeline
    posted = 0
    for sig in all_signals:
        success = post_to_render("/api/agent/insight", {
            "type":         "telegram_signal",
            "agent":        "telegram",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "trade_signal": sig,
            "all_signals":  [sig],
            "market_summary": f"Signal from {sig.get('channel','')}",
        })
        if success:
            posted += 1
            add_knowledge("telegram_signals", {
                "symbol":    sig.get("symbol"),
                "direction": sig.get("direction"),
                "channel":   sig.get("channel"),
                "rr":        sig.get("rr_ratio"),
                "confidence":sig.get("confidence"),
                "reason":    sig.get("reason","")[:100],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    print(f"[TG AGENT] Done. Messages scanned: {total_msgs} | Signals found: {len(all_signals)} | Posted: {posted}", flush=True)

    return {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "messages_read": total_msgs,
        "signals_found": len(all_signals),
        "signals_posted": posted,
        "channels":      channels,
    }

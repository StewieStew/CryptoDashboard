"""
TELEGRAM SIGNAL AGENT — real-time listener + direct executor
Job: Listen to configured Telegram groups. When a message contains a trade
     signal (coin + direction + SL), parse it directly and open the trade
     in the DB immediately — no analyst approval required.

Parsing rules:
  - Coin name: BTC, ETH, XRP, DOGE, SOL (+ common aliases)
  - Direction: long/buy → LONG, short/sell → SHORT
  - SL (required): "sl 59600", "stop 59600", "stop loss: 59600"
  - TP (optional): "tp 62000", "target 62000" — defaults to 2:1 R:R if missing
  - Entry (optional): "entry 60000" — defaults to live Binance price

Images are downloaded and stored for dashboard display regardless of whether
a parseable signal is found. They are NOT forwarded to the analyst.

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
import re, os, base64, threading, time
from datetime import datetime, timezone
from pathlib import Path

import requests

from agents.state import get_state, set_state, add_knowledge, post_to_render

RENDER_URL  = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")
TG_API_ID   = os.environ.get("TG_API_ID", "")
TG_API_HASH = os.environ.get("TG_API_HASH", "")
SESSION_PATH = str(Path.home() / "Desktop" / "CryptoDashboard" / "agents" / "tg_session")

# Coin aliases → USDT symbol
_COIN_MAP = {
    "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
    "ethereum": "ETHUSDT", "eth": "ETHUSDT",
    "ripple": "XRPUSDT",  "xrp": "XRPUSDT",
    "dogecoin": "DOGEUSDT", "doge": "DOGEUSDT",
    "solana": "SOLUSDT",  "sol": "SOLUSDT",
}

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


def _parse_signal_text(text: str) -> dict | None:
    """
    Regex parser for Telegram trade signals.
    Minimum required fields: coin name + direction + SL level.
    Returns a dict or None if minimum fields are not present.
    """
    t = text.lower()

    # Coin — try longest aliases first so "dogecoin" beats "doge"
    symbol = None
    for alias in sorted(_COIN_MAP, key=len, reverse=True):
        if re.search(r'\b' + re.escape(alias) + r'\b', t):
            symbol = _COIN_MAP[alias]
            break
    if not symbol:
        return None

    # Direction
    direction = None
    if re.search(r'\b(long|buy)\b', t):
        direction = "LONG"
    elif re.search(r'\b(short|sell)\b', t):
        direction = "SHORT"
    if not direction:
        return None

    # SL (required) — matches: "sl 59600", "stop: 59600", "stop loss 59600", "stop at 59600"
    sl = None
    sl_m = re.search(
        r'(?:sl|stop[ -]?loss|stop[ -]?at|stop)\s*[:\-@]?\s*([\d,]+(?:\.\d+)?)',
        t,
    )
    if sl_m:
        try:
            sl = float(sl_m.group(1).replace(',', ''))
        except ValueError:
            pass
    if sl is None:
        return None

    # TP (optional) — matches: "tp 62000", "tp1 62000", "target 62000", "take profit 62000"
    tp = None
    tp_m = re.search(
        r'(?:tp\d?|take[ -]?profit|target)\s*[:\-@]?\s*([\d,]+(?:\.\d+)?)',
        t,
    )
    if tp_m:
        try:
            tp = float(tp_m.group(1).replace(',', ''))
        except ValueError:
            pass

    # Entry hint (optional) — matches: "entry 60000", "enter at 60000"
    entry_hint = None
    entry_m = re.search(
        r'(?:entry|enter(?:[ -]?at)?)\s*[:\-@]?\s*([\d,]+(?:\.\d+)?)',
        t,
    )
    if entry_m:
        try:
            entry_hint = float(entry_m.group(1).replace(',', ''))
        except ValueError:
            pass

    return {
        "symbol":     symbol,
        "direction":  direction,
        "sl":         sl,
        "tp":         tp,          # None → calculated below as 2:1 R:R
        "entry_hint": entry_hint,  # None → use live Binance price
    }


def _execute_telegram_signal(
    parsed: dict,
    image_b64: str | None,
    mime_type: str,
    channel_id,
) -> bool:
    """
    Fetch live price, validate levels, and write the trade directly to the DB
    as status='open'. No analyst cycle, no approval gate.
    """
    global _signals_processed

    symbol    = parsed["symbol"]
    direction = parsed["direction"]
    sl        = parsed["sl"]

    # Live price from Binance
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=8,
        )
        live_price = float(r.json()["price"])
    except Exception as e:
        print(f"[TG SIGNAL] Price fetch failed for {symbol}: {e}", flush=True)
        return False

    # Use entry hint only if within 2% of live (stale channel posts are ignored)
    entry = live_price
    hint  = parsed.get("entry_hint")
    if hint and abs(hint - live_price) / live_price < 0.02:
        entry = hint

    # SL must be on the correct side of entry
    if direction == "LONG" and sl >= entry:
        print(f"[TG SIGNAL] Skipping: LONG SL={sl} >= entry={entry:.4f}", flush=True)
        return False
    if direction == "SHORT" and sl <= entry:
        print(f"[TG SIGNAL] Skipping: SHORT SL={sl} <= entry={entry:.4f}", flush=True)
        return False

    risk = abs(entry - sl)

    # TP: use channel value if valid, else default 2:1 R:R
    tp     = parsed.get("tp")
    tp_src = "signal"
    if tp is not None and direction == "LONG"  and tp <= entry: tp = None
    if tp is not None and direction == "SHORT" and tp >= entry: tp = None
    if tp is None:
        tp     = (entry + 2.0 * risk) if direction == "LONG" else (entry - 2.0 * risk)
        tp_src = "2r_default"

    entry = round(entry, 8)
    tp    = round(tp,    8)
    sl    = round(sl,    8)

    trade_id = f"TG_{symbol}_1h_{direction}_{int(time.time())}"
    trade_data = {
        "id":               trade_id,
        "symbol":           symbol,
        "interval":         "1h",
        "direction":        direction,
        "entry":            entry,
        "tp":               tp,
        "sl":               sl,
        "score":            7.0,
        "effective_score":  7.0,
        "reason":           f"[TELEGRAM SIGNAL] channel {channel_id}",
        "factors_snapshot": {"source": "telegram", "channel_id": str(channel_id)},
        "target_basis":     "telegram_signal",
        "tp_source":        tp_src,
        "opened_at":        datetime.now(timezone.utc).isoformat(),
        "status":           "open",
        "entry_type":       "market",
    }

    try:
        import learning as _learning
        logged = _learning.log_trade(trade_data)
    except Exception as e:
        print(f"[TG SIGNAL] DB write failed: {e}", flush=True)
        return False

    if not logged:
        print(f"[TG SIGNAL] Skipped (duplicate or invalid R:R): {symbol} {direction}", flush=True)
        return False

    _signals_processed += 1
    tp_label = "TP from signal" if tp_src == "signal" else "TP=2:1 R:R (default)"
    print(
        f"[TELEGRAM SIGNAL] {symbol} {direction} "
        f"entry={entry:.4f}  SL={sl:.4f}  TP={tp:.4f}  ({tp_label})",
        flush=True,
    )

    # Store image for dashboard display (not for analysis)
    if image_b64:
        set_state("telegram_latest_image", {
            "image_b64": image_b64,
            "mime_type": mime_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol":    symbol,
            "direction": direction,
        })

    # Push to Render dashboard
    post_to_render("/api/agent/insight", {
        "type":                "telegram_signal",
        "agent":               "telegram",
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "trade_signal": {
            "symbol": symbol, "direction": direction,
            "entry": entry, "tp": tp, "sl": sl,
            "reason": f"Direct copy-trade from Telegram channel {channel_id}",
        },
        "market_summary":      f"Direct execution: Telegram channel {channel_id}",
        "telegram_image_b64":  image_b64,
        "telegram_image_mime": mime_type if image_b64 else None,
    })

    add_knowledge("telegram_signals", {
        "symbol":    symbol,
        "direction": direction,
        "channel":   str(channel_id),
        "entry":     entry,
        "tp":        tp,
        "sl":        sl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return True


def _listener_loop():
    """
    Background thread: connects to Telegram and listens for new messages
    in real time. Reconnects automatically if the connection drops.
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
                msg       = event.message
                text      = msg.text or ""
                image_b64 = None
                mime_type = "image/jpeg"

                # Download image attachments only (photos and image documents)
                # Videos, audio, stickers, and other document types are skipped.
                if msg.photo:
                    try:
                        img_bytes = await client.download_media(msg, bytes)
                        if img_bytes:
                            image_b64 = base64.b64encode(img_bytes).decode()
                    except Exception as img_err:
                        print(f"[TG AGENT] Image download error: {img_err}", flush=True)
                elif msg.document:
                    doc_mime = getattr(msg.document, "mime_type", "") or ""
                    if doc_mime.startswith("image/"):
                        try:
                            img_bytes = await client.download_media(msg, bytes)
                            if img_bytes:
                                image_b64 = base64.b64encode(img_bytes).decode()
                                mime_type = doc_mime
                        except Exception as img_err:
                            print(f"[TG AGENT] Image download error: {img_err}", flush=True)

                # Skip completely empty messages
                if not text and not image_b64:
                    return

                channel_id = event.chat_id
                preview = text[:60].replace("\n", " ") if text else "[image only]"
                print(f"[TG AGENT] Message in {channel_id}: {preview}...", flush=True)

                # Parse text for a trade signal and execute directly
                if text:
                    parsed = _parse_signal_text(text)
                    if parsed:
                        _execute_telegram_signal(
                            parsed,
                            image_b64=image_b64,
                            mime_type=mime_type,
                            channel_id=channel_id,
                        )
                        return

                # No parseable signal — store image for reference/display if present
                if image_b64:
                    try:
                        set_state("telegram_latest_image", {
                            "image_b64": image_b64,
                            "mime_type": mime_type,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "symbol":    None,
                            "direction": None,
                        })
                    except Exception:
                        pass
                    print(f"[TG AGENT] Image stored for reference (no signal found).", flush=True)
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
    Called by orchestrator scheduler — just returns listener status.
    The real work happens in the background listener thread.
    """
    alive = _listener_thread is not None and _listener_thread.is_alive()
    print(
        f"[TG AGENT] Listener: {'running' if alive else 'NOT running'} | "
        f"Signals executed: {_signals_processed}",
        flush=True,
    )
    return {
        "listener_running":  alive,
        "signals_processed": _signals_processed,
        "channels":          _get_channels(),
    }

"""
Discord webhook notifications for trade signals and closures.

Set DISCORD_WEBHOOK_URL in your Render environment variables.
If the env var is missing, all calls silently no-op.
"""

import os
import requests

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ── Color palette (matches dashboard UI) ─────────────────────────────────────
_COL_LONG_DAY   = 0xBC8CFF   # purple  — Day   LONG
_COL_LONG_SWING = 0x58A6FF   # blue    — Swing LONG
_COL_SHORT_DAY  = 0xFF8C00   # orange  — Day   SHORT
_COL_SHORT_SWING = 0xF85149  # red     — Swing SHORT
_COL_WIN        = 0x3FB950   # green
_COL_LOSS       = 0xF85149   # red
_COL_CANCEL     = 0x484F58   # grey


# ── Helpers ───────────────────────────────────────────────────────────────────

def _post(payload: dict) -> None:
    """POST to Discord webhook, silently ignore all errors."""
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception:
        pass


def _fmt(v) -> str:
    """Format a price value with appropriate decimal places."""
    if v is None:
        return "—"
    n = float(v)
    if n >= 10000:  return f"{n:,.2f}"
    if n >= 100:    return f"{n:.3f}"
    if n >= 1:      return f"{n:.4f}"
    if n >= 0.01:   return f"{n:.5f}"
    return f"{n:.6f}"


def _tier(interval: str) -> str:
    return "Day" if interval in ("15m", "30m", "1h") else "Swing"


def _signal_color(direction: str, interval: str) -> int:
    tier = _tier(interval)
    if direction == "LONG":
        return _COL_LONG_DAY if tier == "Day" else _COL_LONG_SWING
    return _COL_SHORT_DAY if tier == "Day" else _COL_SHORT_SWING


# ── Public API ────────────────────────────────────────────────────────────────

def send_signal_alert(trade: dict) -> None:
    """
    Send a rich Discord embed when the scanner logs a new signal.

    Expected keys in `trade`: symbol, interval, direction, entry, tp, sl,
                               score, reason, target_basis
    """
    direction = trade.get("direction", "LONG")
    interval  = trade.get("interval", "4h")
    sym       = trade.get("symbol", "???")
    tier      = _tier(interval)
    is_long   = direction == "LONG"

    emoji     = "🟢" if is_long else "🔴"
    entry     = float(trade.get("entry") or 0)
    tp        = float(trade.get("tp")    or 0)
    sl        = float(trade.get("sl")    or 0)
    score     = trade.get("score", 0)
    reason    = trade.get("reason", "")
    tbasis    = trade.get("target_basis", "")

    # Percentage distances
    tp_pct = abs((tp - entry) / entry * 100) if entry else 0
    sl_pct = abs((sl - entry) / entry * 100) if entry else 0
    rr     = round(tp_pct / sl_pct, 1) if sl_pct else 0

    tp_sign = "+" if is_long else "-"
    sl_sign = "-" if is_long else "+"

    fields = [
        {"name": "Entry",
         "value": f"`${_fmt(entry)}`", "inline": True},
        {"name": f"Target ({tp_sign}{tp_pct:.1f}%)",
         "value": f"`${_fmt(tp)}`",   "inline": True},
        {"name": f"Stop ({sl_sign}{sl_pct:.1f}%)",
         "value": f"`${_fmt(sl)}`",   "inline": True},
        {"name": "Score",
         "value": f"`{score}/10`",    "inline": True},
        {"name": "R:R",
         "value": f"`{rr}:1`",        "inline": True},
        {"name": "Tier",
         "value": f"`{tier}`",        "inline": True},
    ]

    if reason:
        fields.append({"name": "Reason",
                        "value": reason[:256], "inline": False})
    if tbasis:
        fields.append({"name": "TP Basis",
                        "value": tbasis[:128], "inline": False})

    # ── Claude AI assessment ──────────────────────────────────────────────────
    ai = trade.get("ai_analysis") or {}
    if ai:
        confidence     = ai.get("confidence", "?")
        recommendation = ai.get("recommendation", "")
        reasoning      = ai.get("reasoning", "")
        risks          = ai.get("risks", [])
        positives      = ai.get("positives", [])

        rec_emoji = {"strong_take": "🔥", "take": "✅", "skip": "⛔"}.get(recommendation, "🤖")
        ai_lines  = [f"{rec_emoji} **{recommendation.replace('_',' ').title()}** — Confidence: {confidence}/100"]
        if reasoning:
            ai_lines.append(reasoning)
        if positives:
            ai_lines.append("**+** " + " · ".join(positives[:2]))
        if risks:
            ai_lines.append("**⚠** " + " · ".join(risks[:2]))

        fields.append({"name": "🤖 Claude AI Assessment",
                        "value": "\n".join(ai_lines)[:512], "inline": False})

    embed = {
        "color": _signal_color(direction, interval),
        "title": f"{emoji} {direction} — {sym}  ({tier} · {interval})",
        "fields": fields,
        "footer": {"text": "Crypto Dashboard · Paper Trade"},
    }

    _post({"embeds": [embed]})


def send_close_alert(trade: dict, status: str,
                     close_price: float, roi: float) -> None:
    """
    Send a Discord embed when a trade is closed (TP hit, SL hit, or manual).

    Expected keys in `trade`: symbol, interval, direction, entry
    """
    direction = trade.get("direction", "LONG")
    interval  = trade.get("interval", "4h")
    sym       = trade.get("symbol", "???")
    tier      = _tier(interval)
    entry     = float(trade.get("entry") or 0)

    if status == "win":
        emoji = "✅"
        color = _COL_WIN
        how   = "TP Hit"
    elif status == "loss":
        emoji = "❌"
        color = _COL_LOSS
        how   = "SL Hit"
    else:
        emoji = "➖"
        color = _COL_CANCEL
        how   = "Cancelled"

    roi_str = f"+{roi:.2f}%" if roi >= 0 else f"{roi:.2f}%"

    embed = {
        "color": color,
        "title": f"{emoji} {status.upper()} ({how}) — {sym}  ({tier} · {interval} · {direction})",
        "fields": [
            {"name": "Entry",
             "value": f"`${_fmt(entry)}`",       "inline": True},
            {"name": "Close",
             "value": f"`${_fmt(close_price)}`", "inline": True},
            {"name": "ROI",
             "value": f"`{roi_str}`",            "inline": True},
        ],
        "footer": {"text": "Crypto Dashboard · Paper Trade"},
    }

    _post({"embeds": [embed]})

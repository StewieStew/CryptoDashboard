"""
RISK AGENT — runs every 5 minutes
Job: Monitor all open trades. Check if market conditions have changed since
     entry. Flag trades that should be exited early. Track daily drawdown.
     Never opens trades — only protects existing ones.
Output: risk alerts, exit recommendations, portfolio health
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone

import requests
import anthropic

from agents.state import (set_state, get_state, add_report,
                          post_to_render)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BINANCE_BASE  = "https://api.binance.us/api/v3"


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def get_live_price(symbol: str) -> float:
    try:
        r = requests.get(f"{BINANCE_BASE}/ticker/price",
                         params={"symbol": symbol}, timeout=8)
        return float(r.json()["price"])
    except Exception:
        return 0.0


def get_recent_candles(symbol: str, interval: str = "1h", limit: int = 20) -> list:
    try:
        r = requests.get(f"{BINANCE_BASE}/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=10)
        return [{"open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4])}
                for c in r.json()]
    except Exception:
        return []


def calc_unrealized_pnl(trade: dict, live_price: float) -> dict:
    try:
        entry     = float(trade.get("entry", 0))
        tp        = float(trade.get("tp", 0))
        sl        = float(trade.get("sl", 0))
        direction = trade.get("direction", "LONG")

        if direction == "LONG":
            pnl_pct  = (live_price - entry) / entry * 100
            sl_dist  = (entry - sl) / entry * 100
            tp_dist  = (tp - live_price) / live_price * 100
            progress = (live_price - entry) / (tp - entry) * 100 if tp > entry else 0
        else:
            pnl_pct  = (entry - live_price) / entry * 100
            sl_dist  = (sl - entry) / entry * 100
            tp_dist  = (live_price - tp) / live_price * 100
            progress = (entry - live_price) / (entry - tp) * 100 if entry > tp else 0

        return {
            "pnl_pct":   round(pnl_pct, 3),
            "sl_dist":   round(abs(sl_dist), 3),
            "tp_dist":   round(tp_dist, 3),
            "progress":  round(progress, 1),
            "live_price": live_price,
        }
    except Exception:
        return {}


def run() -> dict:
    """Monitor open trades and assess risk."""
    print("  Running...", flush=True)

    # Get open trades from Render
    open_trades = []
    try:
        r = requests.get(
            f"{get_state('render_url', os.environ.get('RENDER_URL', 'https://cryptodashboard-nuf5.onrender.com'))}/api/trades",
            timeout=12
        )
        all_trades  = r.json() if r.status_code == 200 else []
        open_trades = [t for t in all_trades if t.get("status") == "open"]
        closed_recent = [t for t in all_trades if t.get("status") in ("win","loss")][-10:]
    except Exception:
        closed_recent = []

    if not open_trades:
        print("  No open trades to monitor.", flush=True)
        result = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "open_count":   0,
            "alerts":       [],
            "portfolio_ok": True,
        }
        set_state("risk_status", result)
        return result

    # Get macro and analyst context
    macro    = get_state("macro_regime", {})
    analyst  = get_state("analyst_ratings", {})
    macro_regime  = macro.get("regime_type", "uncertain")
    coin_ratings  = analyst.get("ratings", {})
    coin_bias     = macro.get("coin_bias", {})

    # Assess each trade
    trade_assessments = []
    alerts = []

    for trade in open_trades:
        sym       = trade.get("symbol", "")
        direction = trade.get("direction", "")
        entry     = float(trade.get("entry", 0))
        opened_at = trade.get("opened_at", "")
        reason    = trade.get("reason", "")
        signal_type = trade.get("signal_type", trade.get("tp_source",""))

        live_price = get_live_price(sym)
        if not live_price:
            continue

        pnl_info  = calc_unrealized_pnl(trade, live_price)
        candles   = get_recent_candles(sym, "1h", 20)

        coin      = sym.replace("USDT", "")
        rating    = coin_ratings.get(sym, 5)
        bias      = coin_bias.get(coin, "neutral")

        # Risk flags
        flags = []
        if direction == "LONG"  and bias == "short": flags.append("macro bias turned SHORT")
        if direction == "SHORT" and bias == "long":  flags.append("macro bias turned LONG")
        if direction == "LONG"  and macro_regime == "bear_trending": flags.append("bear trending regime — LONGs at risk")
        if direction == "SHORT" and macro_regime == "bull_trending": flags.append("bull trending regime — SHORTs at risk")
        if int(rating) <= 3:  flags.append(f"analyst rating dropped to {rating}/10")
        if pnl_info.get("pnl_pct", 0) < -1.5: flags.append(f"unrealized loss {pnl_info['pnl_pct']:.2f}%")

        assessment = {
            "trade_id":    trade.get("id", ""),
            "symbol":      sym,
            "direction":   direction,
            "entry":       entry,
            "live_price":  live_price,
            "pnl":         pnl_info,
            "rating":      rating,
            "macro_bias":  bias,
            "flags":       flags,
            "opened_at":   opened_at,
            "reason":      reason,
        }
        trade_assessments.append(assessment)
        if flags:
            alerts.append({
                "trade_id": trade.get("id",""),
                "symbol":   sym,
                "direction": direction,
                "flags":    flags,
                "pnl_pct":  pnl_info.get("pnl_pct", 0),
            })

    # Daily P&L tracking
    recent_pnl  = sum(t.get("roi_pct") or 0 for t in closed_recent)
    daily_losses = sum(1 for t in closed_recent[-5:] if t.get("status") == "loss")

    portfolio_health = "good"
    if recent_pnl < -5:   portfolio_health = "drawdown"
    if daily_losses >= 4: portfolio_health = "loss_streak"

    # Ask Claude for exit recommendations if there are flagged trades
    exit_recs = []
    if alerts and _claude():
        flagged_str = json.dumps(alerts, indent=2)
        assessments_str = json.dumps([{
            "symbol": a["symbol"],
            "direction": a["direction"],
            "entry": a["entry"],
            "live_price": a["live_price"],
            "pnl_pct": a["pnl"].get("pnl_pct"),
            "tp_progress": a["pnl"].get("progress"),
            "flags": a["flags"],
        } for a in trade_assessments if a.get("flags")], indent=2)

        prompt = f"""You are the Risk Manager on a crypto trading desk.
Macro regime: {macro_regime} | Portfolio health: {portfolio_health}
Recent P&L: {recent_pnl:+.2f}% | Consecutive losses: {daily_losses}

Flagged open trades needing review:
{assessments_str}

For each flagged trade, decide: HOLD or EXIT.
Exit only if the trade thesis is clearly broken (regime reversed, approaching SL, macro fully opposed).
Do not exit just because of a flag — only if multiple factors suggest the setup is invalid.

Respond with ONLY this JSON:
{{
  "recommendations": [
    {{
      "trade_id": "<id>",
      "symbol": "<sym>",
      "action": "<hold|monitor|exit>",
      "reason": "<one sentence>",
      "urgency": "<low|medium|high>"
    }}
  ],
  "portfolio_note": "<one sentence on overall portfolio risk>",
  "max_urgency": "<low|medium|high>"
}}"""

        try:
            client = _claude()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            rec_data = json.loads(raw)
            exit_recs = rec_data.get("recommendations", [])

            # Post urgent exit recs to Render
            high_urgency = [r for r in exit_recs if r.get("urgency") == "high"]
            if high_urgency:
                post_to_render("/api/agent/report", {
                    "type":    "risk_alert",
                    "agent":   "risk",
                    "alerts":  high_urgency,
                    "portfolio_note": rec_data.get("portfolio_note"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            print(f"  Claude error: {e}", flush=True)

    result = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "open_count":       len(open_trades),
        "alerts":           alerts,
        "exit_recs":        exit_recs,
        "portfolio_health": portfolio_health,
        "portfolio_pnl":    round(recent_pnl, 3),
        "assessments":      trade_assessments,
    }

    set_state("risk_status", result)
    add_report("risk", "trade_monitor", {
        "open_count": len(open_trades),
        "alerts":     len(alerts),
        "health":     portfolio_health,
        "exit_recs":  exit_recs,
    })

    alert_count = len(alerts)
    exit_count  = sum(1 for r in exit_recs if r.get("action") == "exit")
    print(f"  Done. Open: {len(open_trades)} | Alerts: {alert_count} | "
          f"Exit recs: {exit_count} | Health: {portfolio_health}", flush=True)
    return result

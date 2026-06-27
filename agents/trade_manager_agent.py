"""
TRADE MANAGER AGENT — runs every 5 minutes alongside Risk
Job: Actively manage open trades the way a real trader would.
     Move stop to break-even, take partial profits, trail stops.
     Does NOT open trades — only protects and optimises existing ones.

Real traders don't just set and forget. They:
  - Move SL to break-even the moment a trade moves 1R in their favour
    (now the trade can only win or scratch, never lose)
  - Take 50% off at 1.5R–2R to lock in profit
  - Trail the stop after 2R+ so the remaining position rides the move
  - Exit the full position if price stalls at a key level before TP

Output: SL adjustment instructions posted to Render /api/trade/adjust
"""
from __future__ import annotations
import os, json
from datetime import datetime, timezone, timedelta

import requests

from agents.state import get_state, set_state, post_to_render

BINANCE_BASE = "https://api.binance.us/api/v3"
RENDER_URL   = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")

# ── Thresholds ─────────────────────────────────────────────────────────────────
BE_TRIGGER_R    = 1.0   # move SL to break-even after this many R of profit
PARTIAL_R       = 1.5   # take 50% profit at this R
TRAIL_START_R   = 2.0   # start trailing stop here
TRAIL_ATR_MULT  = 1.5   # trail = ATR × this multiplier below/above current price
PENDING_MAX_HOURS  = 4.0   # cancel pending trades older than this
PENDING_DRIFT_PCT  = 2.0   # cancel if price moved this % away from entry (wrong dir)


def _get_price(symbol: str) -> float:
    try:
        r = requests.get(f"{BINANCE_BASE}/ticker/price",
                         params={"symbol": symbol}, timeout=8)
        return float(r.json()["price"])
    except Exception:
        return 0.0


def _get_atr(symbol: str, interval: str = "15m", period: int = 14) -> float:
    """Wilder's ATR — same calculation as analyst agent."""
    try:
        r = requests.get(f"{BINANCE_BASE}/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "limit": period + 10}, timeout=10)
        candles = r.json()
        trs = []
        for i in range(1, len(candles)):
            h  = float(candles[i][2])
            l  = float(candles[i][3])
            pc = float(candles[i-1][4])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if not trs:
            return 0.0
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr
    except Exception:
        return 0.0


def _r_multiple(trade: dict, live: float) -> float:
    """How many R has this trade moved in our favour? Negative = against us."""
    try:
        entry = float(trade["entry"])
        sl    = float(trade["sl"])
        risk  = abs(entry - sl)
        if risk == 0:
            return 0.0
        if trade["direction"] == "LONG":
            return (live - entry) / risk
        else:
            return (entry - live) / risk
    except Exception:
        return 0.0


def _cancel_pending(trade_id: str) -> bool:
    try:
        r = requests.post(f"{RENDER_URL}/api/trades/{trade_id}/cancel", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _check_pending_cancellations(pending_trades: list) -> int:
    """
    Cancel pending DB trades that have become invalid:
      1. Price drifted 2%+ away from entry in wrong direction
      2. SL blown through before entry
      3. Trade pending more than 4 hours
    Returns count cancelled.
    """
    cancelled = 0
    now_utc = datetime.now(timezone.utc)
    for trade in pending_trades:
        sym   = trade.get("symbol", "")
        entry = float(trade.get("entry", 0))
        sl    = float(trade.get("sl", 0))
        dirn  = trade.get("direction", "")
        tid   = trade.get("id", "")
        opened = trade.get("opened_at", "")
        if not (sym and entry and sl and dirn and tid):
            continue
        live = _get_price(sym)
        if not live:
            continue
        reason = None
        dist_pct = (live - entry) / entry * 100
        if dirn == "LONG" and dist_pct > PENDING_DRIFT_PCT:
            reason = f"price ran {dist_pct:.1f}% above limit entry — LONG setup stale"
        elif dirn == "SHORT" and dist_pct < -PENDING_DRIFT_PCT:
            reason = f"price fell {abs(dist_pct):.1f}% below limit entry — SHORT setup stale"
        elif dirn == "LONG" and live <= sl:
            reason = f"SL blown before entry (live={live:.4f} <= sl={sl:.4f})"
        elif dirn == "SHORT" and live >= sl:
            reason = f"SL blown before entry (live={live:.4f} >= sl={sl:.4f})"
        elif opened:
            try:
                age_h = (now_utc - datetime.fromisoformat(opened.replace("Z", "+00:00"))).total_seconds() / 3600
                if age_h > PENDING_MAX_HOURS:
                    reason = f"expired ({age_h:.1f}h > {PENDING_MAX_HOURS}h limit)"
            except Exception:
                pass
        if reason:
            print(f"  [PENDING CANCEL] {sym} {dirn}: {reason}", flush=True)
            if _cancel_pending(tid):
                cancelled += 1
    return cancelled


def run() -> dict:
    """Check every open trade and post management instructions to Render."""
    print("  Running trade manager...", flush=True)

    # Fetch open trades from Render
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=12)
        all_trades    = r.json() if r.status_code == 200 else []
        open_trades   = [t for t in all_trades if t.get("status") == "open"]
        pending_trades = [t for t in all_trades if t.get("status") == "pending"]
    except Exception as e:
        print(f"  [TM] Could not fetch trades: {e}", flush=True)
        return {"actions": [], "open": 0}

    # Cancel any invalidated pending trades
    if pending_trades:
        n = _check_pending_cancellations(pending_trades)
        if n:
            print(f"  Cancelled {n} stale pending trade(s).", flush=True)

    if not open_trades:
        print("  No open trades to manage.", flush=True)
        return {"actions": [], "open": 0}

    actions   = []
    trade_mgmt_state = get_state("trade_mgmt", {})  # persists what we've already done per trade

    for trade in open_trades:
        trade_id  = str(trade.get("id", ""))
        sym       = trade.get("symbol", "")
        direction = trade.get("direction", "LONG")
        entry     = float(trade.get("entry") or 0)
        sl        = float(trade.get("sl")    or 0)
        tp        = float(trade.get("tp")    or 0)

        if not (sym and entry and sl and tp):
            continue

        live  = _get_price(sym)
        if not live:
            continue

        r_mult = _r_multiple(trade, live)
        risk   = abs(entry - sl)
        atr    = _get_atr(sym)
        done   = trade_mgmt_state.get(trade_id, {})  # what we've already actioned

        coin = sym.replace("USDT", "")
        arrow = "↑" if direction == "LONG" else "↓"
        print(f"  {coin} {arrow}  live=${live:,.4f}  entry=${entry:,.4f}  "
              f"R={r_mult:+.2f}  SL=${sl:,.4f}", flush=True)

        # ── 1. Move stop to break-even ─────────────────────────────────────
        if r_mult >= BE_TRIGGER_R and not done.get("be_done"):
            # SL moves to entry ± a tiny buffer (0.05% so we don't get ticked out)
            buffer = entry * 0.0005
            new_sl = (entry + buffer) if direction == "LONG" else (entry - buffer)

            # Only move if it's an improvement (don't move SL backwards)
            if direction == "LONG"  and new_sl > sl:
                action = _post_sl_adjust(trade_id, sym, direction, new_sl,
                                         f"Break-even: price moved {r_mult:.1f}R — "
                                         f"stop moved to entry ${entry:,.4f}. Trade is now risk-free.")
                actions.append(action)
                done["be_done"] = True
                print(f"    → BE stop: ${new_sl:,.4f}  (was ${sl:,.4f})", flush=True)

            elif direction == "SHORT" and new_sl < sl:
                action = _post_sl_adjust(trade_id, sym, direction, new_sl,
                                         f"Break-even: price moved {r_mult:.1f}R — "
                                         f"stop moved to entry ${entry:,.4f}. Trade is now risk-free.")
                actions.append(action)
                done["be_done"] = True
                print(f"    → BE stop: ${new_sl:,.4f}  (was ${sl:,.4f})", flush=True)

        # ── 2. Trail stop after 2R ─────────────────────────────────────────
        if r_mult >= TRAIL_START_R and atr > 0:
            trail_dist = atr * TRAIL_ATR_MULT
            if direction == "LONG":
                new_sl = live - trail_dist
                if new_sl > sl:   # only tighten, never widen
                    action = _post_sl_adjust(trade_id, sym, direction, new_sl,
                                             f"Trailing stop: {r_mult:.1f}R in profit — "
                                             f"stop trailing {TRAIL_ATR_MULT}×ATR below price.")
                    actions.append(action)
                    done["trailing"] = True
                    print(f"    → Trail stop: ${new_sl:,.4f}  ({TRAIL_ATR_MULT}×ATR below ${live:,.4f})", flush=True)
            else:
                new_sl = live + trail_dist
                if new_sl < sl:   # only tighten
                    action = _post_sl_adjust(trade_id, sym, direction, new_sl,
                                             f"Trailing stop: {r_mult:.1f}R in profit — "
                                             f"stop trailing {TRAIL_ATR_MULT}×ATR above price.")
                    actions.append(action)
                    done["trailing"] = True
                    print(f"    → Trail stop: ${new_sl:,.4f}  ({TRAIL_ATR_MULT}×ATR above ${live:,.4f})", flush=True)

        # ── 3. Partial profit at 1.5R ──────────────────────────────────────
        if r_mult >= PARTIAL_R and not done.get("partial_done"):
            action = _post_partial(trade_id, sym, direction, live,
                                   f"Partial exit at {r_mult:.1f}R — "
                                   f"taking 50% off at ${live:,.4f}. Remainder rides to TP.")
            actions.append(action)
            done["partial_done"] = True
            print(f"    → Partial profit: 50% taken at ${live:,.4f}  ({r_mult:.1f}R)", flush=True)

        trade_mgmt_state[trade_id] = done

    # Persist what we've actioned so we don't repeat on next cycle
    set_state("trade_mgmt", trade_mgmt_state)

    # Clean up state for trades that are no longer open
    open_ids = {str(t.get("id","")) for t in open_trades}
    for tid in list(trade_mgmt_state.keys()):
        if tid not in open_ids:
            del trade_mgmt_state[tid]

    n = len(actions)
    print(f"  Done. {len(open_trades)} open trades | {n} management action{'s' if n != 1 else ''} taken.", flush=True)
    return {"actions": actions, "open": len(open_trades)}


def _post_sl_adjust(trade_id: str, symbol: str, direction: str,
                    new_sl: float, reason: str) -> dict:
    action = {
        "type":      "sl_adjust",
        "trade_id":  trade_id,
        "symbol":    symbol,
        "direction": direction,
        "new_sl":    round(new_sl, 8),
        "reason":    reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    post_to_render("/api/trade/adjust", action)
    return action


def _post_partial(trade_id: str, symbol: str, direction: str,
                  price: float, reason: str) -> dict:
    action = {
        "type":      "partial_exit",
        "trade_id":  trade_id,
        "symbol":    symbol,
        "direction": direction,
        "price":     round(price, 8),
        "pct":       50,
        "reason":    reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    post_to_render("/api/trade/adjust", action)
    return action

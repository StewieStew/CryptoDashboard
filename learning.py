"""
Adaptive Learning Engine — Trade Tracking, Post-Trade Analysis & Signal Optimization.

Storage: SQLite at /data/trades.db (Render Persistent Disk) with fallback to ./trades.db
"""

import sqlite3
import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# STORAGE PATH
# ─────────────────────────────────────────────
_DATA_DIR = "/data"
DB_PATH   = os.path.join(_DATA_DIR, "trades.db") if os.path.isdir(_DATA_DIR) else "trades.db"
_lock     = threading.Lock()

# ─────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────
FACTOR_NAMES    = ["regime", "bos", "sweep", "volume", "obv", "rsi", "adx"]
DEFAULT_WEIGHTS = {
    "regime": 2.0,   # trending regime aligned (2 pts by default)
    "bos":    2.0,   # confirmed BOS
    "sweep":  2.0,   # liquidity sweep
    "volume": 1.0,   # volume expansion / spike
    "obv":    1.0,   # OBV confirmation / divergence
    "rsi":    1.0,   # RSI confirmation
    "adx":    1.0,   # ADX trend strength
    # Max = 10 pts
}
DEFAULT_THRESHOLD = 7.0    # minimum score to fire a signal
DEFAULT_STOP_MULT = 0.1    # ATR wick buffer on structural stop (was 0.5 main stop)
ADAPT_WINDOW      = 8      # trades to look back per-factor
MIN_SAMPLES       = 2      # minimum trades before adapting a factor
WEIGHT_FLOOR      = 0.30   # minimum factor weight (as fraction of default)
WEIGHT_CEIL       = 1.50   # maximum factor weight (as multiple of default)


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db():
    with _lock:
        db = _conn()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              TEXT PRIMARY KEY,
                symbol          TEXT NOT NULL,
                interval        TEXT NOT NULL,
                direction       TEXT NOT NULL,
                entry           REAL NOT NULL,
                tp              REAL NOT NULL,
                sl              REAL NOT NULL,
                score           REAL NOT NULL,
                effective_score REAL,
                reason          TEXT,
                factors_snapshot TEXT,
                target_basis    TEXT,
                status          TEXT DEFAULT 'open',
                opened_at       TEXT NOT NULL,
                closed_at       TEXT,
                close_price     REAL,
                roi_pct         REAL,
                post_trade_analysis TEXT
            );
            CREATE TABLE IF NOT EXISTS adaptation_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                changes_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        defaults = {
            "signal_threshold": str(DEFAULT_THRESHOLD),
            "stop_multiplier":  str(DEFAULT_STOP_MULT),
            "weights":          json.dumps(DEFAULT_WEIGHTS),
        }
        for k, v in defaults.items():
            db.execute("INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)", (k, v))
        db.commit()

        # ── Config migrations ──────────────────────────────────────────────
        # Lower threshold from 8.0 → 7.0 (hard gates now do the heavy filtering)
        db.execute("UPDATE config SET value=? WHERE key='signal_threshold' AND CAST(value AS REAL) = 8.0",
                   (str(DEFAULT_THRESHOLD),))
        db.execute("UPDATE config SET value=? WHERE key='stop_multiplier' AND CAST(value AS REAL) >= 0.4",
                   (str(DEFAULT_STOP_MULT),))
        db.commit()

        # ── Schema migrations: add new columns to existing tables ──────────
        for col, defn in [
            ("breakeven_activated", "INTEGER DEFAULT 0"),
            ("ai_analysis",         "TEXT"),
            ("partial_tp",          "REAL"),
            ("partial_hit",         "INTEGER DEFAULT 0"),
        ]:
            try:
                db.execute(f"ALTER TABLE trades ADD COLUMN {col} {defn}")
                db.commit()
            except Exception:
                pass    # column already exists

        db.close()


_init_db()


# ─────────────────────────────────────────────
# CONFIG HELPERS
# ─────────────────────────────────────────────
def _get_cfg(key: str, default):
    db = _conn()
    row = db.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    db.close()
    return row[0] if row else default


def _set_cfg(db, key: str, val: str):
    db.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, val))


def get_weights() -> dict:
    raw = _get_cfg("weights", json.dumps(DEFAULT_WEIGHTS))
    try:
        w = json.loads(raw)
        # Back-fill any missing factors
        for f, d in DEFAULT_WEIGHTS.items():
            w.setdefault(f, d)
        return w
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def get_threshold() -> float:
    return float(_get_cfg("signal_threshold", DEFAULT_THRESHOLD))


def get_stop_multiplier() -> float:
    return float(_get_cfg("stop_multiplier", DEFAULT_STOP_MULT))


# ─────────────────────────────────────────────
# TIER HELPERS
# ─────────────────────────────────────────────
# Two trading tiers — max 1 open trade per symbol per tier.
# Day tier  : 15m entries (intraday, tight SL)
# Swing tier: 4h entries  (multi-day, wider SL)
_DAY_INTERVALS   = ("15m", "30m", "1h")
_SWING_INTERVALS = ("4h", "1d", "1w")

def _get_tier(interval: str) -> str:
    return "day" if interval in _DAY_INTERVALS else "swing"


def _close_internal(db, trade_id: str, close_price: float, status: str) -> None:
    """Close a trade using an existing db connection (caller must hold _lock)."""
    row = db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    if not row or row["status"] != "open":
        return
    trade = dict(row)
    if status == "cancelled" or not close_price:
        close_price = trade["entry"]
    roi      = _calc_roi(trade["direction"], trade["entry"], close_price)
    analysis = _generate_analysis(trade, close_price, status, roi)
    now      = datetime.now(timezone.utc).isoformat()
    db.execute("""
        UPDATE trades SET status=?, closed_at=?, close_price=?, roi_pct=?,
        post_trade_analysis=? WHERE id=?
    """, (status, now, round(close_price, 8), roi, json.dumps(analysis), trade_id))


# ─────────────────────────────────────────────
# TRADE OPERATIONS
# ─────────────────────────────────────────────
def log_trade(trade: dict) -> bool:
    """
    Insert a new open trade. Returns False if skipped (duplicate direction).
    Tier rules (enforced per symbol):
      - Same tier, same direction already open → skip.
      - Same tier, opposite direction open     → close existing, open new.
    """
    sym, intv, dirn = trade["symbol"], trade["interval"], trade["direction"]
    tier         = _get_tier(intv)
    tier_intervals = _DAY_INTERVALS if tier == "day" else _SWING_INTERVALS
    placeholders   = ",".join("?" * len(tier_intervals))

    with _lock:
        db = _conn()
        try:
            existing = db.execute(
                f"SELECT id, direction FROM trades "
                f"WHERE symbol=? AND interval IN ({placeholders}) AND status IN ('open','pending')",
                (sym, *tier_intervals)
            ).fetchall()

            for ex in existing:
                if ex["direction"] == dirn:
                    return False   # already in this direction on this tier (open or pending)
                else:
                    # Opposing signal — exit the current position before opening new one
                    _close_internal(db, ex["id"], trade["entry"], "cancelled")
                    _adapt(db)

            entry_f   = float(trade["entry"])
            sl_f      = float(trade["sl"])
            risk_dist = abs(entry_f - sl_f)
            partial_tp = (round(entry_f + 1.5 * risk_dist, 8) if dirn == "LONG"
                          else round(entry_f - 1.5 * risk_dist, 8))

            # Trades with a retest entry price lower/higher than current price
            # start as 'pending' — activated when price reaches the entry level.
            initial_status = trade.get("status", "pending")

            db.execute("""
                INSERT INTO trades
                (id, symbol, interval, direction, entry, tp, sl, score, effective_score,
                 reason, factors_snapshot, target_basis, opened_at, partial_tp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"], sym, intv, dirn,
                trade["entry"], trade["tp"], trade["sl"],
                trade["score"], trade.get("effective_score", trade["score"]),
                trade.get("reason", ""),
                json.dumps(trade.get("factors_snapshot", {})),
                trade.get("target_basis", ""),
                trade["opened_at"],
                partial_tp,
                initial_status,
            ))
            db.commit()
            return True
        finally:
            db.close()


def close_trade(trade_id: str, close_price: float, status: str) -> Optional[dict]:
    """Close a trade manually, generate post-trade analysis, trigger adaptation."""
    with _lock:
        db = _conn()
        try:
            row = db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
            if not row or row["status"] != "open":
                return None
            trade = dict(row)
            # Use entry price as close price for cancelled trades (0% ROI)
            if status == "cancelled" or not close_price:
                close_price = trade["entry"]
            roi   = _calc_roi(trade["direction"], trade["entry"], close_price)
            analysis = _generate_analysis(trade, close_price, status, roi)
            now = datetime.now(timezone.utc).isoformat()
            db.execute("""
                UPDATE trades SET status=?, closed_at=?, close_price=?, roi_pct=?,
                post_trade_analysis=? WHERE id=?
            """, (status, now, round(close_price, 8), roi, json.dumps(analysis), trade_id))
            db.commit()
            changes = _adapt(db)
            db.commit()
            return {**dict(db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()),
                    "adaptation_changes": changes}
        finally:
            db.close()


_DAY_MAX_HOURS   = 72    # 3 days max for Day (15m) trades
_SWING_MAX_HOURS = 336   # 14 days max for Swing (4h) trades


def auto_close(symbol: str, interval: str, current_price: float) -> tuple:
    """
    Auto-close open trades whose TP/SL has been hit, or that have expired.

    Partial TP system (1.5R):
    - When price reaches 1.5× risk distance, 50% of position is booked at that level
      and the stop loss moves to breakeven (entry).
    - On final close, ROI is blended: 0.5×partial_roi + 0.5×remainder_roi.
    - If the 50% remainder hits breakeven, the trade still counts as a WIN
      (since the partial locked in profit).

    Breakeven trailing stop:
    - Once price moves 1× risk distance in our favour, SL moves to entry (breakeven).

    Returns: (closed_list, partials_list)
    """
    closed   = []
    partials = []
    tier      = "day" if interval in ("15m", "30m", "1h") else "swing"
    max_hours = _DAY_MAX_HOURS if tier == "day" else _SWING_MAX_HOURS
    # Pending trades expire faster — if no retest within this window, miss the trade
    _PENDING_MAX_HOURS = {"day": 6, "swing": 48}
    pending_max = _PENDING_MAX_HOURS.get(tier, 24)

    with _lock:
        db = _conn()
        try:
            # ── Activate or expire pending (retest-entry) trades ──────────────
            pending_rows = db.execute(
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status='pending'",
                (symbol, interval)
            ).fetchall()
            for row in pending_rows:
                t = dict(row)
                entry_px  = float(t["entry"])
                direction = t["direction"]
                # Check if retest level has been reached
                activated = (
                    (direction == "LONG"  and current_price <= entry_px) or
                    (direction == "SHORT" and current_price >= entry_px)
                )
                if activated:
                    db.execute("UPDATE trades SET status='open' WHERE id=?", (t["id"],))
                    continue
                # Check pending timeout — price never pulled back, miss the trade
                try:
                    opened = datetime.fromisoformat(t["opened_at"].replace("Z", "+00:00"))
                    age_h  = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                    if age_h > pending_max:
                        now_ts = datetime.now(timezone.utc).isoformat()
                        db.execute(
                            "UPDATE trades SET status='cancelled', closed_at=?, "
                            "close_price=?, roi_pct=0 WHERE id=?",
                            (now_ts, entry_px, t["id"])
                        )
                        closed.append({
                            "id": t["id"], "symbol": t["symbol"],
                            "interval": t["interval"], "direction": t["direction"],
                            "entry": entry_px, "close_price": entry_px,
                            "roi_pct": 0.0, "status": "cancelled",
                        })
                except Exception:
                    pass
            db.commit()

            rows = db.execute(
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status='open'",
                (symbol, interval)
            ).fetchall()
            for row in rows:
                t   = dict(row)
                hit = None

                # ── Risk distance used for breakeven calculation ──────────
                risk_d = abs(float(t["entry"]) - float(t["sl"]))

                # ── Activate breakeven once price moves 1R in our favour ──
                be_active = bool(t.get("breakeven_activated", 0))
                if not be_active:
                    if t["direction"] == "LONG" and current_price >= t["entry"] + risk_d:
                        db.execute("UPDATE trades SET breakeven_activated=1 WHERE id=?", (t["id"],))
                        be_active = True
                    elif t["direction"] == "SHORT" and current_price <= t["entry"] - risk_d:
                        db.execute("UPDATE trades SET breakeven_activated=1 WHERE id=?", (t["id"],))
                        be_active = True

                # ── Effective SL: entry (breakeven) or original SL ────────
                eff_sl = float(t["entry"]) if be_active else float(t["sl"])

                # ── Partial TP: lock in 50% at 1.5R, activate breakeven ────
                partial_tp_val = t.get("partial_tp")
                partial_hit_db = bool(t.get("partial_hit", 0))
                if partial_tp_val and not partial_hit_db:
                    hit_partial = (
                        (t["direction"] == "LONG"  and current_price >= float(partial_tp_val)) or
                        (t["direction"] == "SHORT" and current_price <= float(partial_tp_val))
                    )
                    if hit_partial:
                        db.execute(
                            "UPDATE trades SET partial_hit=1, breakeven_activated=1 WHERE id=?",
                            (t["id"],)
                        )
                        partial_hit_db = True
                        be_active      = True
                        eff_sl         = float(t["entry"])
                        partials.append({
                            "id":            t["id"],
                            "symbol":        t["symbol"],
                            "interval":      t["interval"],
                            "direction":     t["direction"],
                            "entry":         t["entry"],
                            "partial_price": float(partial_tp_val),
                        })

                # ── TP / SL checks ────────────────────────────────────────
                if t["direction"] == "LONG":
                    if current_price >= t["tp"]:
                        hit = "win"
                    elif current_price <= eff_sl:
                        hit = "loss" if not be_active else "cancelled"
                elif t["direction"] == "SHORT":
                    if current_price <= t["tp"]:
                        hit = "win"
                    elif current_price >= eff_sl:
                        hit = "loss" if not be_active else "cancelled"

                # ── Time-based stop ───────────────────────────────────────
                if not hit:
                    try:
                        opened = datetime.fromisoformat(
                            t["opened_at"].replace("Z", "+00:00")
                        )
                        age_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                        if age_h > max_hours:
                            hit = "cancelled"
                    except Exception:
                        pass

                # ── Stagnation exit (spec §7/§8: no momentum after 12 candles)
                # If the trade has been open > N hours AND price has barely moved
                # (within 30% of risk distance from entry) → dead trade, exit.
                if not hit:
                    try:
                        _STAG_HOURS = {"day": 3.0, "swing": 24.0}   # 12 × candle period
                        stag_limit  = _STAG_HOURS.get(tier, 6.0)
                        if age_h > stag_limit and risk_d > 0:
                            move = abs(current_price - float(t["entry"]))
                            if move < 0.30 * risk_d:  # price hasn't gone anywhere
                                hit = "cancelled"
                    except Exception:
                        pass

                if hit:
                    close_px = round(current_price, 8)
                    # ── Blended ROI when partial TP was already booked ───────
                    if partial_hit_db and partial_tp_val:
                        p_roi = _calc_roi(t["direction"], float(t["entry"]), float(partial_tp_val))
                        if hit == "win":
                            # 50% booked at partial_tp + 50% at full TP
                            tp_roi = _calc_roi(t["direction"], float(t["entry"]), float(t["tp"]))
                            roi = round(0.5 * p_roi + 0.5 * tp_roi, 2)
                        else:
                            # cancelled (breakeven) or edge-case loss:
                            # 50% at partial_tp + 50% at close price
                            remainder = (0.0 if hit == "cancelled"
                                         else _calc_roi(t["direction"], float(t["entry"]), close_px))
                            roi = round(0.5 * p_roi + 0.5 * remainder, 2)
                            if roi > 0:
                                hit = "win"   # net profitable — count as win
                            if hit == "cancelled":
                                close_px = float(t["entry"])
                    else:
                        if hit == "cancelled":
                            close_px = float(t["entry"])   # 0% ROI on breakeven/time exits
                        roi = _calc_roi(t["direction"], float(t["entry"]), close_px)
                    analysis = _generate_analysis(t, close_px, hit, roi)
                    now      = datetime.now(timezone.utc).isoformat()
                    db.execute("""
                        UPDATE trades SET status=?, closed_at=?, close_price=?, roi_pct=?,
                        post_trade_analysis=? WHERE id=?
                    """, (hit, now, close_px, roi, json.dumps(analysis), t["id"]))
                    closed.append({
                        "id":          t["id"],
                        "symbol":      t["symbol"],
                        "interval":    t["interval"],
                        "direction":   t["direction"],
                        "entry":       t["entry"],
                        "close_price": close_px,
                        "roi_pct":     roi,
                        "status":      hit,
                    })
            if closed:
                _adapt(db)
            db.commit()
        finally:
            db.close()
    return closed, partials


def update_trade_ai(trade_id: str, ai_result: dict) -> None:
    """Store Claude AI analysis on a trade record."""
    if not ai_result:
        return
    with _lock:
        db = _conn()
        try:
            db.execute(
                "UPDATE trades SET ai_analysis=? WHERE id=?",
                (json.dumps(ai_result), trade_id)
            )
            db.commit()
        finally:
            db.close()


def update_trailing_stops(symbol: str, interval: str, current_price: float,
                          swing_highs: list, swing_lows: list) -> list:
    """
    After a trade reaches 2R profit, trail the stop loss to the most recent
    swing structure to lock in gains while letting winners run.

    LONG:  trail SL up to the highest swing LOW that is above the current SL
           (each new HL raises the floor)
    SHORT: trail SL down to the lowest swing HIGH that is below the current SL
           (each new LH lowers the ceiling)

    swing_highs / swing_lows: flat lists of prices (floats), most recent last.
    Returns list of trade IDs whose SL was updated.
    """
    updated = []
    with _lock:
        db = _conn()
        try:
            rows = db.execute(
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status='open'",
                (symbol, interval)
            ).fetchall()
            for row in rows:
                t         = dict(row)
                entry     = float(t["entry"])
                current_sl = float(t["sl"])
                risk_d    = abs(entry - current_sl)
                direction = t["direction"]

                if risk_d <= 0:
                    continue

                # Only trail once 2R in profit
                profit_r = (
                    (current_price - entry) / risk_d if direction == "LONG"
                    else (entry - current_price) / risk_d
                )
                if profit_r < 2.0:
                    continue

                if direction == "LONG" and swing_lows:
                    # Highest swing low that is: above current SL AND below current price
                    candidates = [p for p in swing_lows
                                  if p > current_sl and p < current_price]
                    if candidates:
                        new_sl = round(max(candidates), 8)
                        if new_sl > current_sl:
                            db.execute("UPDATE trades SET sl=? WHERE id=?",
                                       (new_sl, t["id"]))
                            updated.append(t["id"])

                elif direction == "SHORT" and swing_highs:
                    # Lowest swing high that is: below current SL AND above current price
                    candidates = [p for p in swing_highs
                                  if p < current_sl and p > current_price]
                    if candidates:
                        new_sl = round(min(candidates), 8)
                        if new_sl < current_sl:
                            db.execute("UPDATE trades SET sl=? WHERE id=?",
                                       (new_sl, t["id"]))
                            updated.append(t["id"])

            if updated:
                db.commit()
        finally:
            db.close()
    return updated


def _calc_roi(direction: str, entry: float, close_price: float) -> float:
    if not entry or entry == 0:
        return 0.0
    if direction == "LONG":
        return round((close_price - entry) / entry * 100, 2)
    return round((entry - close_price) / entry * 100, 2)


# ─────────────────────────────────────────────
# POST-TRADE ANALYSIS
# ─────────────────────────────────────────────
def _generate_analysis(trade: dict, close_price: float, status: str, roi: float) -> dict:
    try:
        snap = json.loads(trade.get("factors_snapshot") or "{}")
    except (json.JSONDecodeError, TypeError):
        snap = {}
    dirn        = trade["direction"]
    entry       = trade["entry"]
    tp          = trade["tp"]
    sl          = trade["sl"]
    score       = trade.get("score", 0)
    interval    = trade.get("interval", "?").upper()
    sym         = trade["symbol"].replace("USDT", "")
    risk_dist   = abs(sl - entry)
    reward_dist = abs(tp - entry)
    actual_move = abs(close_price - entry)
    move_pct    = actual_move / entry * 100

    lines      = []
    key_lesson = ""
    causes     = []

    # ── WIN ─────────────────────────────────────────────────────────────────
    if status == "win":
        rr_achieved = round(actual_move / risk_dist, 2) if risk_dist else 0
        lines.append(
            f"Target reached at ${close_price:,.4f} — +{roi:.1f}% ROI from entry ${entry:,.4f}. "
            f"Price moved {move_pct:.2f}% in our direction, achieving {rr_achieved:.1f}:1 R:R."
        )
        winning = [k for k, v in snap.items() if v]
        if winning:
            lines.append(f"Active confluence at entry: {', '.join(winning)}.")

        if score >= 9:
            lines.append(
                "High-conviction setup (9+/10) — strong multi-factor alignment led to clean follow-through."
            )
        if snap.get("sweep") and dirn == "LONG":
            lines.append(
                "Bullish sweep preceding the entry effectively cleared sell-side liquidity, "
                "allowing price to advance without resistance overhead."
            )
        elif snap.get("sweep") and dirn == "SHORT":
            lines.append(
                "Bearish sweep cleared buy-side liquidity before entry, "
                "reducing upside pressure and enabling the downside follow-through."
            )
        if snap.get("obv"):
            lines.append(
                "OBV confirmed the directional bias — institutional volume was flowing "
                "in the same direction as the trade throughout."
            )
        key_lesson = (
            f"{'Sweep + BOS + volume alignment' if snap.get('sweep') and snap.get('volume') else 'BOS + confluence'} "
            "working well in current market conditions. This pattern is reinforcing the model."
        )

    # ── LOSS ────────────────────────────────────────────────────────────────
    elif status == "loss":
        stop_tightness = risk_dist / entry * 100
        lines.append(
            f"Stopped out at ${close_price:,.4f} — {roi:.1f}% loss from entry ${entry:,.4f}. "
            f"Stop was {stop_tightness:.2f}% from entry "
            f"({'very tight' if stop_tightness < 0.3 else 'tight' if stop_tightness < 0.8 else 'reasonable'})."
        )

        # Diagnose causes
        if not snap.get("regime"):
            causes.append(
                "Macro regime was NOT aligned with the trade direction. "
                "Trading against the macro trend significantly reduces BOS follow-through probability. "
                "This is the most common reason for BOS failures."
            )

        if snap.get("bos") and actual_move < risk_dist * 0.3:
            causes.append(
                f"Price barely moved ({move_pct:.2f}%) before stopping out — likely a FALSE BOS. "
                "False breaks occur when institutional players engineer a breakout to trap retail traders "
                "before reversing. Look for high-volume rejection candles immediately after the BOS candle."
            )
        elif snap.get("bos") and actual_move < risk_dist * 0.7:
            causes.append(
                "BOS was confirmed but momentum quickly stalled before reaching TP. "
                "This often happens when the move lacks continuation volume or when the broader "
                "market structure creates an invisible ceiling/floor."
            )

        if dirn == "SHORT" and snap.get("sweep"):
            causes.append(
                "A bullish liquidity sweep was present at entry — this is a double-edged signal for shorts. "
                "While sweeps can precede reversals, they also indicate smart-money ABSORPTION of selling. "
                "If the sweep buyers held their position, they would have reversed the move."
            )
        elif dirn == "LONG" and snap.get("sweep"):
            causes.append(
                "A bearish sweep was present — distribution may not have fully completed. "
                "Sellers still in control could have caused the reversal after entry."
            )

        if not snap.get("obv"):
            causes.append(
                "OBV was NOT confirming the trade direction at signal time. "
                "Volume flow opposing the trade thesis is a high-risk warning sign — "
                "this factor should carry more weight in future signal evaluation."
            )

        if not snap.get("volume"):
            causes.append(
                "No volume expansion or spike at the BOS candle. "
                "Low-volume breakouts are frequently false — institutional participation "
                "is required to sustain a structural break."
            )

        if stop_tightness < 0.3:
            causes.append(
                f"Stop was extremely tight ({stop_tightness:.2f}% from entry). "
                "Even routine candle noise can trigger this. "
                "The system is evaluating whether to widen the ATR stop multiplier."
            )

        if score == 7:
            causes.append(
                f"Signal fired at the minimum threshold (7/{int(sum(DEFAULT_WEIGHTS.values()))}). "
                "Minimum-threshold signals have historically lower follow-through rates. "
                "The system will raise the threshold if low-score signals continue to fail."
            )

        if not causes:
            causes.append(
                "Market conditions changed unexpectedly after entry. "
                "No single high-probability cause identified — this is within normal variance."
            )

        lines.append("Most likely cause(s):")
        for i, c in enumerate(causes[:3], 1):
            lines.append(f"  {i}. {c}")

        absent = [k for k, v in snap.items() if not v]
        if absent:
            lines.append(
                f"Factors NOT present at entry: {', '.join(absent)}. "
                "These missing confirmations may have been key warning signs."
            )

        key_lesson = causes[0].split(".")[0] if causes else "Market conditions not favorable."

    else:
        lines.append("Trade was cancelled — no performance data recorded.")
        key_lesson = "Cancelled."

    return {
        "summary":         lines[0] if lines else "",
        "detail":          "\n".join(lines[1:]),
        "key_lesson":      key_lesson,
        "factors_present": [k for k, v in snap.items() if v],
        "factors_absent":  [k for k, v in snap.items() if not v],
        "status":          status,
        "roi_pct":         roi,
    }


# ─────────────────────────────────────────────
# ADAPTATION ENGINE
# ─────────────────────────────────────────────
def _adapt(db) -> list:
    """
    Analyze recent trade performance. Adjust factor weights, signal threshold,
    and stop multiplier. Returns list of human-readable change strings.
    """
    closed = db.execute("""
        SELECT factors_snapshot, status, roi_pct, sl, entry, direction
        FROM trades WHERE status IN ('win','loss')
        ORDER BY closed_at DESC LIMIT ?
    """, (ADAPT_WINDOW * 4,)).fetchall()

    if not closed:
        return []

    changes   = []
    weights   = get_weights()
    threshold = float(db.execute("SELECT value FROM config WHERE key='signal_threshold'").fetchone()[0])
    stop_mult = float(db.execute("SELECT value FROM config WHERE key='stop_multiplier'").fetchone()[0])

    # ── Per-factor weight adaptation ──────────────────────
    for factor in FACTOR_NAMES:
        factor_rows = []
        for r in closed:
            snap = json.loads(r["factors_snapshot"] or "{}")
            if snap.get(factor):
                factor_rows.append(r)
            if len(factor_rows) >= ADAPT_WINDOW:
                break

        if len(factor_rows) < MIN_SAMPLES:
            continue

        wins     = sum(1 for r in factor_rows if r["status"] == "win")
        win_rate = wins / len(factor_rows)
        old_w    = weights.get(factor, DEFAULT_WEIGHTS[factor])
        default_w= DEFAULT_WEIGHTS[factor]
        floor_w  = default_w * WEIGHT_FLOOR
        ceil_w   = default_w * WEIGHT_CEIL

        if win_rate < 0.40:
            # Factor present but keeps losing — reduce its weight
            new_w = max(round(old_w * 0.75, 2), floor_w)
            if abs(new_w - old_w) > 0.01:
                changes.append(
                    f"⬇ '{factor}' weight {old_w:.2f}→{new_w:.2f} "
                    f"(present in {wins}/{len(factor_rows)} wins, win rate {win_rate:.0%})"
                )
                weights[factor] = new_w

        elif win_rate > 0.70:
            # Factor present → winning often — boost its weight
            new_w = min(round(old_w * 1.10, 2), ceil_w)
            if abs(new_w - old_w) > 0.01:
                changes.append(
                    f"⬆ '{factor}' weight {old_w:.2f}→{new_w:.2f} "
                    f"(present in {wins}/{len(factor_rows)} wins, win rate {win_rate:.0%})"
                )
                weights[factor] = new_w

        elif old_w < default_w and win_rate >= 0.50:
            # Weight was reduced but factor is recovering — nudge back toward default
            new_w = min(round(old_w * 1.05, 2), default_w)
            if abs(new_w - old_w) > 0.01:
                changes.append(
                    f"↗ '{factor}' weight recovering {old_w:.2f}→{new_w:.2f} "
                    f"(improving win rate {win_rate:.0%})"
                )
                weights[factor] = new_w

    # ── Overall win rate → threshold adaptation ──────────
    recent = list(closed[:ADAPT_WINDOW])
    if len(recent) >= MIN_SAMPLES:
        overall_wr = sum(1 for r in recent if r["status"] == "win") / len(recent)
        old_thresh = threshold

        if overall_wr < 0.40 and threshold < 9.0:
            threshold = min(round(threshold + 0.5, 1), 9.0)
            changes.append(
                f"⬆ Signal threshold {old_thresh:.1f}→{threshold:.1f} "
                f"(last {len(recent)} trades: {overall_wr:.0%} win rate, target >55%)"
            )
        elif overall_wr > 0.70 and threshold > 6.5:
            threshold = max(round(threshold - 0.25, 1), 6.5)
            changes.append(
                f"⬇ Signal threshold {old_thresh:.1f}→{threshold:.1f} "
                f"(win rate {overall_wr:.0%} — allowing slightly lower bar)"
            )

    # ── Consecutive loss streak protection ───────────────────────────────────
    consec_losses = 0
    for r in list(closed[:8]):
        if r["status"] == "loss":
            consec_losses += 1
        else:
            break
    if consec_losses >= 3 and threshold < 9.5:
        new_thresh = min(round(threshold + 0.5, 1), 9.5)
        if new_thresh != threshold:
            changes.append(
                f"⬆ Signal threshold {threshold:.1f}→{new_thresh:.1f} "
                f"({consec_losses} consecutive losses — tightening entry criteria)"
            )
            threshold = new_thresh

    # ── Stop multiplier adaptation ────────────────────────
    # If recent losses were stopped out within 0.5% of entry → stops too tight
    recent_losses = [r for r in closed[:ADAPT_WINDOW] if r["status"] == "loss"]
    if len(recent_losses) >= 3:
        tight = sum(
            1 for r in recent_losses
            if r["sl"] and r["entry"]
            and abs(float(r["sl"]) - float(r["entry"])) / float(r["entry"]) < 0.005
        )
        if tight >= 3 and stop_mult < 1.0:
            new_mult = min(round(stop_mult + 0.1, 1), 1.0)
            if new_mult != stop_mult:
                changes.append(
                    f"⬆ Stop multiplier {stop_mult:.1f}→{new_mult:.1f}×ATR "
                    f"({tight}/5 recent stops hit within 0.5% of entry — likely noise)"
                )
                stop_mult = new_mult

    # ── Persist changes ───────────────────────────────────
    if changes:
        _set_cfg(db, "weights",          json.dumps(weights))
        _set_cfg(db, "signal_threshold", str(threshold))
        _set_cfg(db, "stop_multiplier",  str(stop_mult))
        db.execute(
            "INSERT INTO adaptation_log (timestamp, changes_json) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(changes))
        )

    return changes


# ─────────────────────────────────────────────
# PUBLIC READ FUNCTIONS
# ─────────────────────────────────────────────
def get_trades() -> list:
    db   = _conn()
    rows = db.execute("SELECT * FROM trades ORDER BY opened_at DESC").fetchall()
    db.close()
    result = []
    for r in rows:
        t = dict(r)
        t["factors_snapshot"]    = json.loads(t.get("factors_snapshot") or "{}")
        t["post_trade_analysis"] = json.loads(t.get("post_trade_analysis") or "null")
        t["ai_analysis"]         = json.loads(t.get("ai_analysis") or "null")
        result.append(t)
    return result


def get_adaptation_log(limit: int = 15) -> list:
    db   = _conn()
    rows = db.execute(
        "SELECT timestamp, changes_json FROM adaptation_log ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    db.close()
    return [{"timestamp": r[0], "changes": json.loads(r[1])} for r in rows]


def get_ai_accuracy() -> dict:
    """
    Compute Claude's own prediction accuracy from every closed trade
    that has an ai_analysis record.

    Returns a dict keyed by recommendation type, e.g.:
    {
      "strong_take": {"total":12, "wins":9, "losses":3, "win_rate_pct":75.0, "avg_roi_pct":2.1},
      "take":        {"total":8,  "wins":5, "losses":3, "win_rate_pct":62.5, "avg_roi_pct":0.8},
    }
    Only closed (win/loss) trades are counted — open trades have no outcome yet.
    """
    trades = get_trades()
    closed = [t for t in trades
              if t.get("status") in ("win", "loss") and t.get("ai_analysis")]

    by_rec: dict = {}
    for t in closed:
        ai  = t.get("ai_analysis") or {}
        rec = ai.get("recommendation", "")
        if not rec:
            continue
        if rec not in by_rec:
            by_rec[rec] = {"wins": 0, "losses": 0, "total_roi": 0.0}
        if t["status"] == "win":
            by_rec[rec]["wins"] += 1
        else:
            by_rec[rec]["losses"] += 1
        by_rec[rec]["total_roi"] += float(t.get("roi_pct") or 0)

    result = {}
    for rec, s in by_rec.items():
        total = s["wins"] + s["losses"]
        result[rec] = {
            "total":        total,
            "wins":         s["wins"],
            "losses":       s["losses"],
            "win_rate_pct": round(s["wins"] / total * 100, 1) if total else 0.0,
            "avg_roi_pct":  round(s["total_roi"] / total, 2)  if total else 0.0,
        }
    return result


def get_learning_state() -> dict:
    weights   = get_weights()
    threshold = get_threshold()
    stop_mult = get_stop_multiplier()
    trades    = get_trades()
    closed    = [t for t in trades if t["status"] in ("win", "loss")]
    wins      = [t for t in closed if t["status"] == "win"]
    win_rate  = round(len(wins) / len(closed) * 100, 1) if closed else None

    factor_states = {}
    for f in FACTOR_NAMES:
        dw = DEFAULT_WEIGHTS[f]
        cw = weights.get(f, dw)
        factor_states[f] = {
            "default": dw,
            "current": round(cw, 2),
            "ratio":   round(cw / dw, 2),
            "status":  ("reduced" if cw < dw * 0.95
                        else "boosted" if cw > dw * 1.02
                        else "normal"),
        }

    return {
        "weights":          factor_states,
        "signal_threshold": threshold,
        "default_threshold":DEFAULT_THRESHOLD,
        "stop_multiplier":  stop_mult,
        "default_stop_mult":DEFAULT_STOP_MULT,
        "overall_win_rate": win_rate,
        "total_closed":     len(closed),
        "adaptation_log":   get_adaptation_log(10),
    }


# ─────────────────────────────────────────────
# REGIME STORAGE (per-symbol)
# ─────────────────────────────────────────────

def save_regime(symbol: str, regime: dict) -> None:
    """Store the detected market regime for a symbol in the config table."""
    with _lock:
        db = _conn()
        try:
            _set_cfg(db, f"regime_{symbol.upper()}", json.dumps(regime))
            db.commit()
        finally:
            db.close()


def get_regime(symbol: str) -> dict | None:
    """Retrieve the last detected regime for a symbol, or None if not yet stored."""
    raw = _get_cfg(f"regime_{symbol.upper()}", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


# ─────────────────────────────────────────────
# AUTO-DEPLOY (Weekly Review)
# ─────────────────────────────────────────────

def auto_deploy_params(new_params: dict, reason: str, improvement: dict) -> bool:
    """
    Auto-deploy parameter changes validated by the weekly Claude review.
    Updates config keys and logs to adaptation_log with source='weekly_review'.
    Returns True if at least one parameter was changed.

    Supported param names:
        signal_threshold, stop_multiplier, adx_threshold, body_ratio_min, min_rr
        weight_<factor>  (e.g. weight_regime, weight_bos, ...)
    """
    _PARAM_MAP = {
        "signal_threshold": "signal_threshold",
        "stop_multiplier":  "stop_multiplier",
        "adx_threshold":    "adx_threshold",
        "body_ratio_min":   "body_ratio_min",
        "min_rr":           "min_rr",
    }

    changed = []
    with _lock:
        db = _conn()
        try:
            weights         = get_weights()
            weights_changed = False

            for param, value in new_params.items():
                if param.startswith("weight_"):
                    factor = param[len("weight_"):]
                    if factor in weights:
                        old_val = weights[factor]
                        weights[factor] = round(float(value), 3)
                        changed.append(
                            f"⚙ Weekly review: '{param}' {old_val:.3f}→{float(value):.3f} — {reason}"
                        )
                        weights_changed = True

                elif param in _PARAM_MAP:
                    cfg_key = _PARAM_MAP[param]
                    old_raw = _get_cfg(cfg_key, "?")
                    _set_cfg(db, cfg_key, str(round(float(value), 4)))
                    changed.append(
                        f"⚙ Weekly review: '{param}' {old_raw}→{value} — {reason}"
                    )

            if weights_changed:
                _set_cfg(db, "weights", json.dumps(weights))

            if changed:
                db.execute(
                    "INSERT INTO adaptation_log (timestamp, changes_json) VALUES (?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps({
                            "source":      "weekly_review",
                            "changes":     changed,
                            "improvement": improvement,
                        }),
                    ),
                )
                db.commit()
        finally:
            db.close()

    return len(changed) > 0


# ─────────────────────────────────────────────
# PER-SYMBOL PARAMS (Research Campaign)
# ─────────────────────────────────────────────

def save_symbol_params(symbol: str, interval: str, params: dict) -> None:
    """
    Save the best backtest-validated parameters for a specific symbol+interval.
    Stored in the config table as JSON under key 'sym_params_{SYMBOL}_{INTERVAL}'.
    The live scanner will use these instead of global defaults when present.
    """
    key = f"sym_params_{symbol.upper()}_{interval}"
    with _lock:
        db = _conn()
        try:
            _set_cfg(db, key, json.dumps(params))
            db.commit()
        finally:
            db.close()


def get_symbol_params(symbol: str, interval: str) -> dict | None:
    """
    Retrieve per-symbol params set by a research campaign, or None if not saved.
    """
    key = f"sym_params_{symbol.upper()}_{interval}"
    raw = _get_cfg(key, None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

"""
Adaptive Learning Engine — Trade Tracking, Post-Trade Analysis & Signal Optimization.
"""
from __future__ import annotations

import sqlite3
import json
import os
import threading
from datetime import timedelta, datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# STORAGE PATH
# ─────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
# Use DB_PATH env var if set; otherwise auto-detect Render's persistent /data disk,
# falling back to the local app directory for development.
_DEFAULT_DB = (
    "/data/trades.db"
    if os.environ.get("RENDER") or os.path.isdir("/data")
    else os.path.join(_HERE, "trades.db")
)
DB_PATH = os.environ.get("DB_PATH", _DEFAULT_DB)
_lock     = threading.Lock()

# ─────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────
FACTOR_NAMES    = ["regime", "bos", "sweep", "volume", "obv", "rsi", "adx",
                   "fvg", "fib", "liquidity"]
DEFAULT_WEIGHTS = {
    "regime":    2.0,   # trending regime aligned (2 pts by default)
    "bos":       2.0,   # confirmed BOS
    "sweep":     2.0,   # liquidity sweep
    "volume":    1.0,   # volume expansion / spike
    "obv":       1.0,   # OBV confirmation / divergence
    "rsi":       1.0,   # RSI confirmation
    "adx":       1.0,   # ADX trend strength
    "fvg":       1.5,   # Fair Value Gap aligned with setup
    "fib":       1.5,   # Fibonacci golden pocket retracement
    "liquidity": 1.0,   # Equal highs / equal lows liquidity pool
    # Core max ≈ 10 pts; FVG + FIB + liquidity are bonus precision factors
}
DEFAULT_THRESHOLD = 7.0    # minimum score to fire a signal
DEFAULT_STOP_MULT = 0.3    # ATR wick buffer on structural stop — 0.3×ATR gives ~3-5 pts breathing room on ETH/BTC 15m
ADAPT_WINDOW      = 8      # trades to look back per-factor
MIN_SAMPLES       = 2      # minimum trades before adapting a factor
WEIGHT_FLOOR_ABS  = 1.0    # absolute minimum — no active factor weight can drop below 1.0
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
        # Raise stop_multiplier to new default (0.3) if still at old tight value (0.1)
        # IMPORTANT: never reset stop_multiplier downward — let the adaptation engine control it
        db.execute("UPDATE config SET value=? WHERE key='stop_multiplier' AND CAST(value AS REAL) < 0.3",
                   (str(DEFAULT_STOP_MULT),))
        db.commit()

        # ── Schema migrations: add new columns to existing tables ──────────
        for col, defn in [
            ("breakeven_activated", "INTEGER DEFAULT 0"),
            ("ai_analysis",         "TEXT"),
            ("partial_tp",          "REAL"),
            ("partial_hit",         "INTEGER DEFAULT 0"),
            ("tp_source",           "TEXT DEFAULT 'unknown'"),
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
# Two trading tiers — used for max-hold-time / stagnation limits only.
# Conflict check is per-symbol per-interval (not per-tier), so e.g.
# a BTC/USDT 4h LONG and BTC/USDT 15m SHORT can coexist.
_DAY_INTERVALS   = ("15m", "30m", "1h")
_SWING_INTERVALS = ("4h", "1d", "1w")

def _get_tier(interval: str) -> str:
    return "day" if interval in _DAY_INTERVALS else "swing"


def _close_internal(db, trade_id: str, close_price: float, status: str) -> None:
    """Close a trade using an existing db connection (caller must hold _lock)."""
    row = db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    if not row or row["status"] not in ("open", "pending"):
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
    Insert a new open trade. Returns False if skipped (already open on same symbol+interval).
    Conflict rule (enforced per symbol per interval):
      - Same symbol AND same interval already open → skip (let existing trade run to SL/TP).
      - Different timeframe on the same symbol is allowed (e.g. 4h LONG + 15m SHORT can coexist).
    """
    sym, intv, dirn = trade["symbol"], trade["interval"], trade["direction"]

    with _lock:
        db = _conn()
        try:
            existing = db.execute(
                "SELECT id FROM trades "
                "WHERE symbol=? AND interval=? AND direction=? "
                "AND status IN ('open','pending') LIMIT 1",
                (sym, intv, dirn)
            ).fetchone()

            if existing:
                logger.info(
                    f"[SKIP] already have open {sym} {intv} {dirn} "
                    f"(id={existing[0]}) — skipping duplicate"
                )
                return False

            # ── Cooldown: skip if same sym+intv+dir closed within last interval ──
            _INTERVAL_SECS = {
                "1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,
                "1h":3600,"2h":7200,"4h":14400,"1d":86400
            }
            _cool_secs = _INTERVAL_SECS.get(intv, 900)
            _last_closed = db.execute(
                "SELECT closed_at FROM trades "
                "WHERE symbol=? AND interval=? AND direction=? AND closed_at IS NOT NULL "
                "ORDER BY closed_at DESC LIMIT 1",
                (sym, intv, dirn)
            ).fetchone()
            if _last_closed:
                try:
                    _last_dt = datetime.fromisoformat(_last_closed[0].replace("Z","+00:00"))
                    _since   = (datetime.now(timezone.utc) - _last_dt).total_seconds()
                    if _since < _cool_secs:
                        logger.info(f"[COOLDOWN] {sym} {intv} {dirn} — {_since:.0f}s since last close, need {_cool_secs}s")
                        return False
                except Exception:
                    pass

            # ── BOS dedup: skip if same entry level closed in last 24h ──
            entry_f   = float(trade["entry"])
            _recent = db.execute(
                "SELECT entry FROM trades "
                "WHERE symbol=? AND interval=? AND direction=? AND closed_at IS NOT NULL "
                "AND closed_at > datetime('now','-24 hours')",
                (sym, intv, dirn)
            ).fetchall()
            for _row in _recent:
                _prev_entry = float(_row[0])
                if _prev_entry > 0 and abs(entry_f - _prev_entry) / _prev_entry < 0.001:
                    logger.info(f"[BOS_DEDUP] {sym} {intv} {dirn} entry {entry_f:.4f} within 0.1% of recent {_prev_entry:.4f}")
                    return False
            sl_f      = float(trade["sl"])
            risk_dist = abs(entry_f - sl_f)

            tp_f = float(trade["tp"])

            # ── Minimum R:R guard: skip trades below 1:1 ────────────────────
            reward_dist = abs(tp_f - entry_f)
            _tp_on_right_side = (tp_f > entry_f) if dirn == "LONG" else (tp_f < entry_f)
            _sl_on_right_side = (sl_f < entry_f) if dirn == "LONG" else (sl_f > entry_f)
            if not _tp_on_right_side or not _sl_on_right_side:
                logger.info(
                    f"[RR_SKIP] {sym} {intv} {dirn} — TP or SL on wrong side of entry "
                    f"(entry={entry_f}, tp={tp_f}, sl={sl_f})"
                )
                return False
            if risk_dist == 0 or reward_dist / risk_dist < 1.0:
                _rr = 0.0 if risk_dist == 0 else reward_dist / risk_dist
                logger.info(
                    f"[RR_SKIP] {sym} {intv} {dirn} — R:R={_rr:.2f} below 1.0 "
                    f"(entry={entry_f}, tp={tp_f}, sl={sl_f})"
                )
                return False
            # ── end R:R guard ─────────────────────────────────────────────────

            partial_tp = (round(entry_f + 1.5 * risk_dist, 8) if dirn == "LONG"
                          else round(entry_f - 1.5 * risk_dist, 8))

            # Market order: all trades open immediately at signal time.
            initial_status = trade.get("status", "open")

            db.execute("""
                INSERT INTO trades
                (id, symbol, interval, direction, entry, tp, sl, score, effective_score,
                 reason, factors_snapshot, target_basis, tp_source, opened_at, partial_tp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"], sym, intv, dirn,
                trade["entry"], trade["tp"], trade["sl"],
                trade["score"], trade.get("effective_score", trade["score"]),
                trade.get("reason", ""),
                json.dumps(trade.get("factors_snapshot", {})),
                trade.get("target_basis", ""),
                trade.get("tp_source", "unknown"),
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
            if not row or row["status"] not in ("open", "pending"):
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




def auto_close(symbol: str, interval: str, current_price: float) -> tuple:
    """
    Auto-close open trades whose SL has been hit.

    Breakeven trailing stop:
    - Once price moves 1× risk distance in our favour, SL moves to entry (breakeven).

    Trades close ONLY when price hits SL or TP. No time-based or stagnation exits.

    Returns: (closed_list, partials_list)
    """
    closed   = []
    partials = []

    with _lock:
        db = _conn()
        try:

            rows = db.execute(
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status IN ('open','pending')",
                (symbol, interval)
            ).fetchall()
            for row in rows:
                t   = dict(row)
                hit = None

                # ── Risk distance used for breakeven calculation ──────────
                risk_d = abs(float(t["entry"]) - float(t["sl"]))

                # ── BE is activated structurally by update_trailing_stops()
                # when the trailing SL crosses above entry (LONG) or below entry (SHORT).
                # No arbitrary spot-price trigger here.
                be_active = bool(t.get("breakeven_activated", 0))

                # ── Effective SL: always use current SL from DB ──────────
                # (trailing stop updater writes entry/above to sl column when
                #  breakeven activates, so t["sl"] is already correct)
                eff_sl = float(t["sl"])

                # ── Minimum duration guard — skip closes on sub-candle trades ──
                _INTERVAL_SECS = {
                    "1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,
                    "1h":3600,"2h":7200,"4h":14400,"1d":86400
                }
                try:
                    _opened = datetime.fromisoformat(t["opened_at"].replace("Z","+00:00"))
                    _age    = (datetime.now(timezone.utc) - _opened).total_seconds()
                    _min    = _INTERVAL_SECS.get(t["interval"], 900)
                    if _age < _min:
                        continue   # too young — let the candle close first
                except Exception:
                    pass

                # ── TP check — must run before SL check ──────────────────
                if t["direction"] == "LONG":
                    if current_price >= float(t["tp"]):
                        hit = "win"
                elif t["direction"] == "SHORT":
                    if current_price <= float(t["tp"]):
                        hit = "win"

                # ── SL check ──────────────────────────────────────────────
                if not hit:
                    if t["direction"] == "LONG":
                        if current_price <= eff_sl:
                            if not be_active:
                                hit = "loss"
                            elif eff_sl > float(t["entry"]):  # trailing SL above entry → profitable exit
                                hit = "win"
                            else:
                                hit = "cancelled"             # SL at entry → scratch
                    elif t["direction"] == "SHORT":
                        if current_price >= eff_sl:
                            if not be_active:
                                hit = "loss"
                            elif eff_sl < float(t["entry"]):  # trailing SL below entry → profitable exit
                                hit = "win"
                            else:
                                hit = "cancelled"             # SL at entry → scratch

                if hit:
                    if hit == "loss":
                        close_px = round(eff_sl, 8)         # snap to exact SL
                    elif hit == "cancelled":
                        close_px = float(t["entry"])         # scratch at entry
                    else:
                        # Win: TP hit closes at TP; trailing SL hit closes at eff_sl
                        _tp_hit = (
                            (t["direction"] == "LONG"  and current_price >= float(t["tp"])) or
                            (t["direction"] == "SHORT" and current_price <= float(t["tp"]))
                        )
                        close_px = round(float(t["tp"]), 8) if _tp_hit else round(eff_sl, 8)
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
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status IN ('open','pending')",
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

                # Trail once 1.5R in profit (break-even activation point)
                profit_r = (
                    (current_price - entry) / risk_d if direction == "LONG"
                    else (entry - current_price) / risk_d
                )
                if profit_r < 1.5:
                    continue

                if direction == "LONG" and swing_lows:
                    # Highest swing low that is: above current SL AND below current price
                    candidates = [p for p in swing_lows
                                  if p > current_sl and p < current_price]
                    if candidates:
                        new_sl = round(max(candidates), 8)
                        if new_sl > current_sl:
                            # If trailing SL crosses above entry → structural BE confirmed
                            be_now = 1 if new_sl > entry else t.get("breakeven_activated", 0)
                            db.execute(
                                "UPDATE trades SET sl=?, breakeven_activated=? WHERE id=?",
                                (new_sl, be_now, t["id"])
                            )
                            if be_now and not t.get("breakeven_activated"):
                                logger.info(
                                    f"[BE_STRUCTURAL] {t['symbol']} {interval} "
                                    f"trailing SL {new_sl:.4f} crossed above entry {entry:.4f} "
                                    f"— breakeven activated"
                                )
                            updated.append(t["id"])

                elif direction == "SHORT" and swing_highs:
                    # Lowest swing high that is: below current SL AND above current price
                    candidates = [p for p in swing_highs
                                  if p < current_sl and p > current_price]
                    if candidates:
                        new_sl = round(min(candidates), 8)
                        if new_sl < current_sl:
                            # If trailing SL crosses below entry → structural BE confirmed
                            be_now = 1 if new_sl < entry else t.get("breakeven_activated", 0)
                            db.execute(
                                "UPDATE trades SET sl=?, breakeven_activated=? WHERE id=?",
                                (new_sl, be_now, t["id"])
                            )
                            if be_now and not t.get("breakeven_activated"):
                                logger.info(
                                    f"[BE_STRUCTURAL] {t['symbol']} {interval} "
                                    f"trailing SL {new_sl:.4f} crossed below entry {entry:.4f} "
                                    f"— breakeven activated"
                                )
                            updated.append(t["id"])

            if updated:
                db.commit()
        finally:
            db.close()
    return updated


COMMISSION_RT = 0.20  # 0.10% entry + 0.10% exit = 0.20 percentage-point round-trip cost

def _calc_roi(direction: str, entry: float, close_price: float) -> float:
    if not entry or entry == 0:
        return 0.0
    if direction == "LONG":
        raw = (close_price - entry) / entry * 100
    else:
        raw = (entry - close_price) / entry * 100
    return round(raw - COMMISSION_RT, 2)


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
        ceil_w   = default_w * WEIGHT_CEIL

        # Weights only go UP — losses never penalize a factor below WEIGHT_FLOOR_ABS
        if win_rate > 0.70:
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

    # ── Enforce absolute weight floor on all non-zero-default factors ────────────
    for f, d in DEFAULT_WEIGHTS.items():
        if d > 0 and weights.get(f, 0) < WEIGHT_FLOOR_ABS:
            weights[f] = WEIGHT_FLOOR_ABS

    # ── Overall win rate → threshold adaptation ──────────
    recent = list(closed[:ADAPT_WINDOW])
    if len(recent) >= MIN_SAMPLES:
        overall_wr = sum(1 for r in recent if r["status"] == "win") / len(recent)
        old_thresh = threshold

        if overall_wr < 0.40 and threshold < 8.0:
            threshold = min(round(threshold + 0.5, 1), 8.0)
            changes.append(
                f"⬆ Signal threshold {old_thresh:.1f}→{threshold:.1f} "
                f"(last {len(recent)} trades: {overall_wr:.0%} win rate, target >55%)"
            )
        elif overall_wr > 0.70 and threshold > 7.0:
            threshold = max(round(threshold - 0.25, 1), 7.0)
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
    if consec_losses >= 3 and threshold < 8.0:
        new_thresh = min(round(threshold + 0.5, 1), 8.0)
        if new_thresh != threshold:
            changes.append(
                f"⬆ Signal threshold {threshold:.1f}→{new_thresh:.1f} "
                f"({consec_losses} consecutive losses — tightening entry criteria)"
            )
            threshold = new_thresh

    # ── Stop multiplier adaptation ────────────────────────
    # If recent losses were stopped out within 1.0% of entry → stops too tight
    recent_losses = [r for r in closed[:ADAPT_WINDOW] if r["status"] == "loss"]
    if len(recent_losses) >= 2:
        tight = sum(
            1 for r in recent_losses
            if r["sl"] and r["entry"]
            and abs(float(r["sl"]) - float(r["entry"])) / float(r["entry"]) < 0.01
        )
        if tight >= 2 and stop_mult < 1.5:
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


def get_open_trades(symbol: str | None = None, interval: str | None = None) -> list:
    """Return all open trades, optionally filtered by symbol and/or interval."""
    db   = _conn()
    q    = "SELECT * FROM trades WHERE status IN ('open','pending')"
    args: list = []
    if symbol:
        q += " AND symbol=?";   args.append(symbol)
    if interval:
        q += " AND interval=?"; args.append(interval)
    q += " ORDER BY opened_at ASC"
    rows = db.execute(q, args).fetchall()
    db.close()
    result = []
    for r in rows:
        t = dict(r)
        t["factors_snapshot"] = json.loads(t.get("factors_snapshot") or "{}")
        t["ai_analysis"]      = json.loads(t.get("ai_analysis") or "null")
        result.append(t)
    return result


def has_active_trade(symbol: str, interval: str, direction: str) -> bool:
    """Return True if there is already an open or pending trade for this symbol+interval+direction."""
    db  = _conn()
    row = db.execute(
        "SELECT 1 FROM trades WHERE symbol=? AND interval=? AND direction=? "
        "AND status IN ('open','pending') LIMIT 1",
        (symbol, interval, direction)
    ).fetchone()
    db.close()
    return row is not None


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


def get_tp_source_stats(min_samples: int = 2) -> dict:
    """
    Return win rate and trade count for each tp_source from closed trades.
    Only includes sources with at least min_samples closed trades.
    """
    db = _conn()
    rows = db.execute("""
        SELECT tp_source, status, COUNT(*) as cnt
        FROM trades
        WHERE status IN ('win','loss') AND tp_source IS NOT NULL
        GROUP BY tp_source, status
    """).fetchall()
    db.close()

    raw: dict = {}
    for row in rows:
        src = row["tp_source"] or "unknown"
        if src not in raw:
            raw[src] = {"wins": 0, "losses": 0}
        if row["status"] == "win":
            raw[src]["wins"] = row["cnt"]
        else:
            raw[src]["losses"] = row["cnt"]

    result = {}
    for src, counts in raw.items():
        total = counts["wins"] + counts["losses"]
        if total >= min_samples:
            result[src] = {
                "win_rate":  round(counts["wins"] / total, 3),
                "total":     total,
                "wins":      counts["wins"],
                "losses":    counts["losses"],
            }
    return result


def tp_source_threshold_adjustment(tp_source: str, min_samples: int = 2) -> float:
    """
    Return extra points to ADD to the effective score threshold based on
    the tp_source's historical win rate.  Higher penalty = harder to fire.

    Win rate ≥ 55%  →  0.0  (performing fine, no penalty)
    Win rate 35–55% → +0.5  (mediocre, slight raise)
    Win rate < 35%  → +1.5  (consistently losing, significant raise)

    Returns 0.0 if the source has fewer than min_samples trades (not enough data).
    """
    stats = get_tp_source_stats(min_samples)
    if tp_source not in stats:
        return 0.0
    wr = stats[tp_source]["win_rate"]
    if wr >= 0.55:
        return 0.0
    elif wr >= 0.35:
        return 0.5
    else:
        return 1.5


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
        "tp_source_stats":  get_tp_source_stats(),
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

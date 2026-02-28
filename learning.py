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
FACTOR_NAMES    = ["regime", "bos", "sweep", "volume", "obv", "rsi", "fib"]
DEFAULT_WEIGHTS = {
    "regime": 2.0,   # trending regime aligned (2 pts by default)
    "bos":    2.0,   # confirmed BOS
    "sweep":  2.0,   # liquidity sweep
    "volume": 1.0,   # volume expansion / spike
    "obv":    1.0,   # OBV confirmation / divergence
    "rsi":    1.0,   # RSI confirmation
    "fib":    1.0,   # Fibonacci confluence
}
DEFAULT_THRESHOLD = 7.0    # minimum score to fire a signal
DEFAULT_STOP_MULT = 0.5    # ATR multiplier for stop placement
ADAPT_WINDOW      = 5      # trades to look back per-factor
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
# TRADE OPERATIONS
# ─────────────────────────────────────────────
def log_trade(trade: dict) -> bool:
    """Insert a new open trade. Returns False if a duplicate open trade already exists."""
    sym, intv, dirn = trade["symbol"], trade["interval"], trade["direction"]
    with _lock:
        db = _conn()
        try:
            dupe = db.execute(
                "SELECT id FROM trades WHERE symbol=? AND interval=? AND direction=? AND status='open'",
                (sym, intv, dirn)
            ).fetchone()
            if dupe:
                return False
            db.execute("""
                INSERT INTO trades
                (id, symbol, interval, direction, entry, tp, sl, score, effective_score,
                 reason, factors_snapshot, target_basis, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"], sym, intv, dirn,
                trade["entry"], trade["tp"], trade["sl"],
                trade["score"], trade.get("effective_score", trade["score"]),
                trade.get("reason", ""),
                json.dumps(trade.get("factors_snapshot", {})),
                trade.get("target_basis", ""),
                trade["opened_at"],
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


def auto_close(symbol: str, interval: str, current_price: float) -> list:
    """Auto-close open trades whose TP or SL has been crossed."""
    closed = []
    with _lock:
        db = _conn()
        try:
            rows = db.execute(
                "SELECT * FROM trades WHERE symbol=? AND interval=? AND status='open'",
                (symbol, interval)
            ).fetchall()
            for row in rows:
                t = dict(row)
                hit = None
                if t["direction"] == "LONG":
                    if current_price >= t["tp"]:
                        hit = "win"
                    elif current_price <= t["sl"]:
                        hit = "loss"
                elif t["direction"] == "SHORT":
                    if current_price <= t["tp"]:
                        hit = "win"
                    elif current_price >= t["sl"]:
                        hit = "loss"
                if hit:
                    roi      = _calc_roi(t["direction"], t["entry"], current_price)
                    analysis = _generate_analysis(t, current_price, hit, roi)
                    now      = datetime.now(timezone.utc).isoformat()
                    db.execute("""
                        UPDATE trades SET status=?, closed_at=?, close_price=?, roi_pct=?,
                        post_trade_analysis=? WHERE id=?
                    """, (hit, now, round(current_price, 8), roi, json.dumps(analysis), t["id"]))
                    closed.append(t["id"])
            if closed:
                _adapt(db)
            db.commit()
        finally:
            db.close()
    return closed


def _calc_roi(direction: str, entry: float, close_price: float) -> float:
    if direction == "LONG":
        return round((close_price - entry) / entry * 100, 2)
    return round((entry - close_price) / entry * 100, 2)


# ─────────────────────────────────────────────
# POST-TRADE ANALYSIS
# ─────────────────────────────────────────────
def _generate_analysis(trade: dict, close_price: float, status: str, roi: float) -> dict:
    snap        = json.loads(trade.get("factors_snapshot") or "{}")
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

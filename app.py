"""
Crypto Analysis Dashboard — Flask Server
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from analysis import full_analysis, chart_for_timeframe
from datetime import datetime, timezone
import time
import threading
import uuid
import os
import json
import pandas as pd
import analysis
import learning
import notifications
import ai_analysis
import market_data
import backtester
import regime

import logging
logger = logging.getLogger(__name__)

app          = Flask(__name__)
_cache       = {}
_lock        = threading.Lock()
TTL          = 300   # 5-min analysis cache

_chart_cache = {}
_clk         = threading.Lock()
CHART_TTL    = 120   # 2-min chart cache

VALID_INTERVALS  = ["15m", "30m", "1h", "4h", "1d", "1w"]
DEFAULT_SYMBOLS  = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT",  "DOGEUSDT",
    "SOLUSDT", "BNBUSDT", "ADAUSDT",  "AVAXUSDT",
    "LINKUSDT","LTCUSDT", "DOTUSDT",  "NEARUSDT", "ATOMUSDT",
]

# Paper-trade mode: only these coins on 4H using discovery-validated params.
PAPER_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
PAPER_INTERVALS = ["1h", "4h"]

# Three-tier scanning: 15m (scalp), 1h (day trade — MACD primary), 4h (swing).
SCAN_INTERVALS   = ["15m", "1h", "4h"]

# Grows as users view additional coins — persists for the life of the server process
_known_symbols   = set(DEFAULT_SYMBOLS)

# Scanner status — visible in /api/scanner-status
_scanner_status  = {
    "running":          False,
    "scans_completed":  0,
    "last_scan_at":     None,   # ISO timestamp
    "last_scan_ago_s":  None,   # seconds since last scan
    "signals_logged":   0,
    "trades_closed":    0,
}

# Backtest jobs — keyed by job_id (uuid)
_backtest_jobs: dict = {}
_bt_lock = threading.Lock()

# Research campaigns — keyed by campaign_id (uuid)
_campaigns: dict = {}
_camp_lock = threading.Lock()


# ── Risk Engine ───────────────────────────────────────────────────────────────
import numpy as _np

_risk_state = {
    "daily_pnl_pct":      0.0,   # running daily P&L as fraction (not %)
    "peak_equity_pct":    0.0,   # high-water mark
    "current_equity_pct": 0.0,   # cumulative P&L fraction
    "daily_reset_date":   None,  # date string of last reset (UTC)
    "halt_until":         None,  # ISO timestamp — trading halted until
    "trades_today":       0,
}
_risk_lock = threading.Lock()

RISK_PER_TRADE_PCT  = 0.01   # 1% of account per trade
DAILY_LOSS_HALT_PCT = 0.05   # halt if daily loss exceeds 5%
DRAWDOWN_REDUCE_PCT = 0.15   # reduce size if drawdown > 15%
FEE_SLIPPAGE_PCT    = 0.001  # 0.1% fee + slippage per side (each way)


def _reset_daily_risk() -> None:
    """Reset daily P&L counter at start of each new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _risk_lock:
        if _risk_state["daily_reset_date"] != today:
            _risk_state["daily_pnl_pct"]  = 0.0
            _risk_state["trades_today"]   = 0
            _risk_state["daily_reset_date"] = today
            # Clear intra-day halt on new day
            _risk_state["halt_until"] = None


def _risk_gate_open() -> bool:
    """
    Returns True when trading is permitted.
    Blocks on: (a) daily loss ≥ 5%, (b) active halt timer.
    """
    _reset_daily_risk()
    with _risk_lock:
        if _risk_state["halt_until"]:
            try:
                halt_dt = datetime.fromisoformat(_risk_state["halt_until"])
                if datetime.now(timezone.utc) < halt_dt:
                    return False
                _risk_state["halt_until"] = None
            except Exception:
                _risk_state["halt_until"] = None
        if _risk_state["daily_pnl_pct"] <= -DAILY_LOSS_HALT_PCT:
            return False
    return True


def _record_trade_pnl(roi_pct: float) -> None:
    """Update risk state after a trade closes (roi_pct is %, e.g. 2.5 = +2.5%)."""
    from datetime import timedelta
    frac = roi_pct / 100.0
    with _risk_lock:
        _risk_state["daily_pnl_pct"]      += frac
        _risk_state["current_equity_pct"] += frac
        _risk_state["trades_today"]       += 1
        if _risk_state["current_equity_pct"] > _risk_state["peak_equity_pct"]:
            _risk_state["peak_equity_pct"] = _risk_state["current_equity_pct"]
        # Trigger halt if daily loss limit breached
        if _risk_state["daily_pnl_pct"] <= -DAILY_LOSS_HALT_PCT:
            _risk_state["halt_until"] = (
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat()


def _position_size_factor() -> float:
    """
    Returns a multiplier (0.25 – 1.0) applied to position sizing.
    Scales down to 0.50× if drawdown > 10%, 0.25× if > 15%.
    """
    with _risk_lock:
        peak = _risk_state["peak_equity_pct"]
        cur  = _risk_state["current_equity_pct"]
    if peak <= 0:
        return 1.0
    dd = (peak - cur) / (1.0 + abs(peak)) if peak != 0 else 0
    if dd >= DRAWDOWN_REDUCE_PCT:
        return 0.25
    elif dd >= 0.10:
        return 0.50
    return 1.0


def _rr_after_fees(entry: float, tp: float, sl: float, direction: str) -> float:
    """
    Net R:R after 0.1% entry + 0.1% exit fees and estimated slippage.
    Returns 0.0 if risk is zero or negative.
    """
    fee = FEE_SLIPPAGE_PCT
    if direction == "LONG":
        eff_entry  = entry * (1.0 + fee)
        eff_tp     = tp    * (1.0 - fee)
        eff_sl     = sl    * (1.0 + fee)
        reward     = eff_tp - eff_entry
        risk       = eff_entry - eff_sl
    else:
        eff_entry  = entry * (1.0 - fee)
        eff_tp     = tp    * (1.0 + fee)
        eff_sl     = sl    * (1.0 - fee)
        reward     = eff_entry - eff_tp
        risk       = eff_sl   - eff_entry
    if risk <= 0:
        return 0.0
    return round(reward / risk, 3)


# ── Mean Reversion Signal (for RANGING regime) ────────────────────────────────
def _mean_reversion_signal(sym: str, interval: str, data: dict) -> dict | None:
    """
    Bollinger Bands (20, 2σ) + RSI strategy for RANGING markets.
      LONG : price ≤ lower band AND RSI < 30  → exit at midband
      SHORT: price ≥ upper band AND RSI > 70  → exit at midband
    Min R:R 1.5 after fees required.
    """
    try:
        candles = data.get("chart", {}).get("candles", [])
        if len(candles) < 25:
            return None

        closes = _np.array([c["close"] for c in candles], dtype=float)
        highs  = _np.array([c["high"]  for c in candles], dtype=float)
        lows   = _np.array([c["low"]   for c in candles], dtype=float)
        cur    = float(closes[-1])

        # Bollinger Bands (20-period, 2 std)
        bb_period = 20
        bb_slice  = closes[-bb_period:]
        mid       = float(_np.mean(bb_slice))
        std       = float(_np.std( bb_slice))
        upper_bb  = mid + 2.0 * std
        lower_bb  = mid - 2.0 * std

        # RSI — prefer value from analysis data, else compute from candles
        rsi_raw = data.get("rsi", {})
        if isinstance(rsi_raw, dict):
            rsi_val = rsi_raw.get("value")
        else:
            rsi_val = rsi_raw if isinstance(rsi_raw, (int, float)) else None
        if rsi_val is None:
            deltas = _np.diff(closes[-15:])
            gains  = _np.where(deltas > 0, deltas, 0.0)
            losses = _np.where(deltas < 0, -deltas, 0.0)
            avg_g  = float(_np.mean(gains[-14:]))  if len(gains)  >= 14 else float(_np.mean(gains))
            avg_l  = float(_np.mean(losses[-14:])) if len(losses) >= 14 else float(_np.mean(losses))
            rsi_val = 100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l > 0 else 50.0

        # LONG: price at or below lower BB, RSI oversold
        if cur <= lower_bb * 1.005 and rsi_val < 30:
            sl = float(_np.min(lows[-5:])) * 0.9985
            tp = mid
            net_rr = _rr_after_fees(cur, tp, sl, "LONG")
            if net_rr < 1.5 or (cur - sl) <= 0:
                return None
            return {
                "direction":        "LONG",
                "entry":            round(cur, 8),
                "target":           round(tp, 8),
                "stop":             round(sl, 8),
                "score":            6.5,
                "signal_type":      "MEAN_REVERSION",
                "reason":           f"BB lower touch RSI={rsi_val:.0f} → midband exit",
                "current_price":    cur,
                "target_basis":     "BB midband",
                "factors_snapshot": {"bb_lower": round(lower_bb, 8), "rsi": round(rsi_val, 1), "net_rr": net_rr},
            }

        # SHORT: price at or above upper BB, RSI overbought
        if cur >= upper_bb * 0.9950 and rsi_val > 70:
            sl = float(_np.max(highs[-5:])) * 1.0015
            tp = mid
            net_rr = _rr_after_fees(cur, tp, sl, "SHORT")
            if net_rr < 1.5 or (sl - cur) <= 0:
                return None
            return {
                "direction":        "SHORT",
                "entry":            round(cur, 8),
                "target":           round(tp, 8),
                "stop":             round(sl, 8),
                "score":            6.5,
                "signal_type":      "MEAN_REVERSION",
                "reason":           f"BB upper touch RSI={rsi_val:.0f} → midband exit",
                "current_price":    cur,
                "target_basis":     "BB midband",
                "factors_snapshot": {"bb_upper": round(upper_bb, 8), "rsi": round(rsi_val, 1), "net_rr": net_rr},
            }
    except Exception:
        pass
    return None


# ── Volatility Squeeze Entry ──────────────────────────────────────────────────
def _squeeze_signal(sym: str, interval: str, data: dict) -> dict | None:
    """
    Detects Bollinger Band squeeze followed by volume spike + breakout.
    Squeeze = current BB width < 0.5× its 20-bar rolling average.
    Breakout = close outside the band on ≥ 1.5× average volume.
    Enters in direction of the breakout with a 2R target.
    """
    try:
        candles = data.get("chart", {}).get("candles", [])
        if len(candles) < 45:
            return None

        closes  = _np.array([c["close"]          for c in candles], dtype=float)
        highs   = _np.array([c["high"]            for c in candles], dtype=float)
        lows    = _np.array([c["low"]             for c in candles], dtype=float)
        volumes = _np.array([c.get("volume", 0.0) for c in candles], dtype=float)
        cur     = float(closes[-1])

        # Build rolling BB width (as % of midpoint) over last 40 bars
        bb_period = 20
        widths = []
        for i in range(bb_period, len(closes)):
            chunk = closes[i - bb_period:i]
            m = float(_np.mean(chunk))
            w = float(_np.std(chunk)) / m if m > 0 else 0.0
            widths.append(w)
        if len(widths) < 20:
            return None

        avg_width = float(_np.mean(widths[-20:]))
        cur_width = widths[-1]
        if cur_width >= avg_width * 0.5:   # not in a squeeze
            return None

        # Volume must be elevated on the breakout bar
        avg_vol = float(_np.mean(volumes[-20:]))
        cur_vol = float(volumes[-1])
        if avg_vol <= 0 or cur_vol < avg_vol * 1.5:
            return None

        # Current BB bands
        bb_slice = closes[-bb_period:]
        mid      = float(_np.mean(bb_slice))
        std      = float(_np.std( bb_slice))
        upper_bb = mid + 2.0 * std
        lower_bb = mid - 2.0 * std
        vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0

        if cur > upper_bb:
            sl     = lower_bb
            risk   = cur - sl
            if risk <= 0:
                return None
            tp     = cur + risk * 2.0
            net_rr = _rr_after_fees(cur, tp, sl, "LONG")
            if net_rr < 1.8:
                return None
            return {
                "direction":        "LONG",
                "entry":            round(cur, 8),
                "target":           round(tp, 8),
                "stop":             round(sl, 8),
                "score":            7.0,
                "signal_type":      "VOL_SQUEEZE",
                "reason":           f"BB squeeze breakout UP vol={vol_ratio}×avg",
                "current_price":    cur,
                "target_basis":     "2R squeeze target",
                "factors_snapshot": {"bb_width": round(cur_width, 6), "vol_ratio": vol_ratio, "net_rr": net_rr},
            }

        if cur < lower_bb:
            sl     = upper_bb
            risk   = sl - cur
            if risk <= 0:
                return None
            tp     = cur - risk * 2.0
            net_rr = _rr_after_fees(cur, tp, sl, "SHORT")
            if net_rr < 1.8:
                return None
            return {
                "direction":        "SHORT",
                "entry":            round(cur, 8),
                "target":           round(tp, 8),
                "stop":             round(sl, 8),
                "score":            7.0,
                "signal_type":      "VOL_SQUEEZE",
                "reason":           f"BB squeeze breakout DOWN vol={vol_ratio}×avg",
                "current_price":    cur,
                "target_basis":     "2R squeeze target",
                "factors_snapshot": {"bb_width": round(cur_width, 6), "vol_ratio": vol_ratio, "net_rr": net_rr},
            }
    except Exception:
        pass
    return None


# ── Per-Config Capital Auto-Weighting ─────────────────────────────────────────
_config_weights: dict = {}       # {(sym, interval): weight_multiplier}
_config_weights_lock = threading.Lock()


def _update_config_weights() -> None:
    """
    Groups closed trades by (symbol, interval). Calculates win rate and profit
    factor per pair, then assigns a capital-weight multiplier:
      pf ≥ 2.0 and wr ≥ 60%  → 1.5× (scale up)
      pf < 1.0 or wr < 35%   → 0.25× (scale down)
      otherwise               → 1.0× (neutral)
    Requires ≥ 5 closed trades to change the weight of any config.
    Results are stored in _config_weights and logged to scanner_status.
    """
    try:
        from collections import defaultdict
        trades = learning.get_trades()
        closed = [t for t in trades if t.get("status") in ("win", "loss")]
        if not closed:
            return

        groups: dict = defaultdict(list)
        for t in closed:
            key = (t.get("symbol", ""), t.get("interval", ""))
            groups[key].append(t)

        new_weights: dict = {}
        for (sym, iv), group in groups.items():
            if len(group) < 5:
                continue
            wins       = [t for t in group if t.get("status") == "win"]
            losses_lst = [t for t in group if t.get("status") == "loss"]
            wr         = len(wins) / len(group)
            gross_p    = sum(abs(t.get("roi_pct") or 0) for t in wins)
            gross_l    = sum(abs(t.get("roi_pct") or 0) for t in losses_lst)
            pf         = (gross_p / gross_l) if gross_l > 0 else (2.5 if gross_p > 0 else 0.0)

            if pf >= 2.0 and wr >= 0.60:
                w = 1.5
            elif pf < 1.0 or wr < 0.35:
                w = 0.25
            else:
                w = 1.0
            new_weights[(sym, iv)] = w

        with _config_weights_lock:
            _config_weights.update(new_weights)

        # Persist as symbol-level param adjustment so scanner picks it up
        for (sym, iv), w in new_weights.items():
            existing = learning.get_symbol_params(sym, iv) or {}
            existing["capital_weight"] = w
            learning.save_symbol_params(sym, iv, existing)

    except Exception:
        pass


def _get_config_weight(sym: str, interval: str) -> float:
    """Return the stored capital-weight multiplier for a (symbol, interval) pair."""
    with _config_weights_lock:
        return _config_weights.get((sym, interval), 1.0)


# ── Apply walk-forward validated 4H params from discovery checkpoint ──────────
def _apply_discovery_params() -> None:
    """
    Saves the best per-coin strategy params+weights from the discovery checkpoint
    into the DB so the scanner picks them up automatically.
    Called once at startup.
    """
    best = {
        ("BTCUSDT", "4h"): {
            "config":           "Structure + Volume",
            "score_threshold":  6.5,
            "min_rr":           2.0,
            "adx_threshold":    30,
            "body_ratio_min":   0.30,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 2.0, "rsi": 0.0, "adx": 0.0,
                "volume": 2.0, "obv": 1.0, "regime": 1.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("ETHUSDT", "4h"): {
            "config":           "Trend + S/R + RSI",
            "score_threshold":  7.0,
            "min_rr":           2.0,
            "adx_threshold":    30,
            "body_ratio_min":   0.10,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "rsi": 2.0, "adx": 0.0,
                "volume": 0.0, "obv": 0.0, "regime": 2.5,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("XRPUSDT", "4h"): {
            "config":           "Full Precision",
            "score_threshold":  7.0,
            "min_rr":           2.0,
            "adx_threshold":    15,
            "body_ratio_min":   0.10,
            "level_touch_min":  1,
            "weights": {
                "bos": 1.5, "sweep": 1.5, "fib": 1.5, "fvg": 1.5,
                "volume": 1.0, "regime": 1.0, "liquidity": 1.0,
                "rsi": 0.0, "adx": 0.0, "obv": 0.0,
            },
        },
        ("BTCUSDT", "1h"): {
            "config":           "Structure + Volume",
            "score_threshold":  7.0,
            "min_rr":           2.0,
            "adx_threshold":    28,
            "body_ratio_min":   0.30,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 2.0, "rsi": 0.5, "adx": 0.0,
                "volume": 2.0, "obv": 1.0, "regime": 1.5,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("ETHUSDT", "1h"): {
            "config":           "Trend + S/R + RSI",
            "score_threshold":  7.5,
            "min_rr":           2.0,
            "adx_threshold":    28,
            "body_ratio_min":   0.15,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 1.5, "rsi": 2.0, "adx": 0.0,
                "volume": 0.5, "obv": 0.0, "regime": 2.5,
                "fvg": 0.5, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("XRPUSDT", "1h"): {
            "config":           "Full Precision",
            "score_threshold":  7.5,
            "min_rr":           2.0,
            "adx_threshold":    15,
            "body_ratio_min":   0.15,
            "level_touch_min":  1,
            "weights": {
                "bos": 1.5, "sweep": 1.5, "fib": 1.5, "fvg": 1.5,
                "volume": 1.0, "regime": 1.5, "liquidity": 1.0,
                "rsi": 0.5, "adx": 0.0, "obv": 0.0,
            },
        },
        # ── 15m scalp params (min_rr=2.0 — lower bar suits noise on short TF) ──
        ("BTCUSDT", "15m"): {
            "config":           "Structure + Volume",
            "score_threshold":  6.0,
            "min_rr":           2.0,
            "adx_threshold":    25,
            "body_ratio_min":   0.30,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 2.0, "rsi": 0.0, "adx": 0.0,
                "volume": 2.0, "obv": 1.0, "regime": 1.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("ETHUSDT", "15m"): {
            "config":           "Structure + Volume",
            "score_threshold":  7.5,
            "min_rr":           2.0,
            "adx_threshold":    30,
            "body_ratio_min":   0.10,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 2.0, "volume": 2.0, "obv": 1.0,
                "regime": 1.0, "rsi": 0.0, "adx": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("XRPUSDT", "15m"): {
            "config":           "Trend + S/R + RSI",
            "score_threshold":  7.0,
            "min_rr":           2.0,
            "adx_threshold":    15,
            "body_ratio_min":   0.20,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "rsi": 2.0, "regime": 2.5,
                "adx": 0.0, "volume": 0.0, "obv": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("DOGEUSDT", "15m"): {
            "config":           "Trend + Structure + RSI",
            "score_threshold":  5.5,
            "min_rr":           2.0,
            "adx_threshold":    20,
            "body_ratio_min":   0.30,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 1.0, "rsi": 2.0, "regime": 2.0,
                "adx": 0.0, "volume": 0.0, "obv": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
    }
    for (sym, interval), params in best.items():
        learning.save_symbol_params(sym, interval, params)

_apply_discovery_params()


# ── Session filter ────────────────────────────────────────────────────────────
def _in_active_session() -> bool:
    """
    True during European or US trading sessions (7am–10pm UTC).
    Day (15m) signals are only logged during active sessions when volume is highest.
    Swing (4h) signals fire any time.
    """
    h = datetime.now(timezone.utc).hour
    return 7 <= h < 22


# ── Price-based auto-close helper ────────────────────────────────────────────
def _auto_close_from_data(data: dict, sym: str, interval: str) -> None:
    """Auto-close trades whose TP/SL has been hit based on current price.
    Signal logging is handled ONLY by the background scanner (with bias filter).
    """
    cur_price = data.get("current_price", 0)
    if cur_price:
        learning.auto_close(sym, interval, float(cur_price))


# ── Higher-TF bias filter ─────────────────────────────────────────────────────
def _bias_agrees(signal_dir: str, htf_data: dict) -> bool:
    """
    Returns True only if the higher-timeframe context agrees with the signal.
    Day (15m) signals filtered by 1h context.
    Swing (4h) signals filtered by 1d context.

    Agreement = HTF price above 200 EMA (for LONG) or below (for SHORT),
                OR a confirmed BOS on the HTF in the same direction.
    """
    regime    = htf_data.get("regime", {})
    structure = htf_data.get("structure", {})
    above_200 = regime.get("above_200", False)
    bull_bos  = structure.get("bullish_bos", False) if structure else False
    bear_bos  = structure.get("bearish_bos", False) if structure else False

    if signal_dir == "LONG":
        return above_200 or bull_bos
    else:  # SHORT
        return (not above_200) or bear_bos


# ── Background scanner ────────────────────────────────────────────────────────
def _resolve_open_trades(sym: str, interval: str) -> int:
    """
    Fetch recent OHLCV bars and check every open trade's TP/SL against bar highs/lows.

    More accurate than spot-price polling: catches TP/SL hits that occurred between
    two 5-minute scan cycles (e.g. a wick that touched SL and recovered before next scan).

    Returns the number of trades resolved.
    """
    open_trades = learning.get_open_trades(sym, interval)
    if not open_trades:
        return 0

    # Fetch enough bars to cover any open trade's max lifetime
    lookback = {"15m": 500, "1h": 220, "4h": 120}.get(interval, 220)
    try:
        df = analysis.fetch_ohlcv(sym, interval, lookback)
    except Exception:
        return 0
    if df is None or df.empty:
        return 0

    resolved = 0
    for trade in open_trades:
        trade_id  = trade["id"]
        direction = trade["direction"]
        tp        = float(trade["tp"])
        sl        = float(trade["sl"])
        try:
            opened_at = pd.Timestamp(trade["opened_at"])
        except Exception:
            continue

        # Only examine bars that opened AFTER the trade was logged
        future = df[df.index > opened_at]
        if future.empty:
            continue

        outcome     = None
        close_price = None
        for _, bar in future.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            # TP removed — trailing stop lets winners run; only SL closes here
            if direction == "LONG":
                if lo <= sl:
                    outcome = "loss"; close_price = sl;  break
            else:  # SHORT
                if hi >= sl:
                    outcome = "loss"; close_price = sl;  break

        if outcome:
            try:
                closed = learning.close_trade(trade_id, close_price, outcome)
                if closed:
                    resolved += 1
                    notifications.send_close_alert(
                        closed, closed["status"], closed["close_price"], closed["roi_pct"]
                    )
            except Exception:
                pass

    return resolved


def _background_scanner() -> None:
    """
    Runs forever in a daemon thread.
    Every 5 minutes scans ALL paper symbols × PAPER_INTERVALS.

    Upgrade (v2) — per scan cycle:
      • Risk gate: skip all signals if daily loss ≥ 5% or drawdown halt active
      • Market regime: TRENDING → BOS strategies only
                        RANGING   → mean-reversion (BB+RSI) + vol-squeeze
                        UNCERTAIN → trend signals only (higher score bar)
      • Fee-adjusted R:R: every signal must clear min_rr AFTER 0.1% fees/slippage
      • Per-config capital weighting updated after each full sweep
      • P&L recorded per closed trade to drive drawdown calculations
    """
    _scanner_status["running"] = True
    time.sleep(10)          # short startup pause, then begin immediately

    while True:
        syms = PAPER_SYMBOLS   # paper-trade: locked to discovery-validated coins only
        scan_signals = 0
        scan_closes  = 0

        # ── Daily risk reset ──────────────────────────────────────────────────
        _reset_daily_risk()

        for sym in syms:
            # Fetch bias TFs: 1h for confirmation candle check, 1d for 4H bias
            bias_cache: dict[str, dict] = {}
            for bias_tf in ("1h", "1d"):
                try:
                    bias_data = full_analysis(sym, bias_tf)
                    bias_cache[bias_tf] = bias_data
                    # Cache the bias data too
                    with _lock:
                        _cache[f"{sym}_{bias_tf}"] = (bias_data, time.time())
                except Exception:
                    bias_cache[bias_tf] = {}
                time.sleep(1)

            for interval in PAPER_INTERVALS:
                try:
                    # ── Risk gate: skip new entries if limits breached ────────
                    if not _risk_gate_open():
                        # Still resolve open trades even when halted
                        scan_closes += _resolve_open_trades(sym, interval)
                        continue

                    # Load per-symbol discovery params (weights + thresholds)
                    sym_p = learning.get_symbol_params(sym, interval) or {}
                    sym_weights = sym_p.get("weights") or None
                    data = full_analysis(sym, interval, weights=sym_weights)
                    now  = time.time()
                    data["cache_age"] = 0
                    with _lock:
                        _cache[f"{sym}_{interval}"] = (data, now)

                    # ── Detect per-symbol market regime from already-computed data ──
                    regime_label = "UNCERTAIN"
                    try:
                        detected_regime = regime.detect_regime_from_data(data)
                        regime_label    = detected_regime.get("label", "UNCERTAIN")
                        learning.save_regime(sym, detected_regime)
                        base_p = {
                            "score_threshold": sym_p.get("score_threshold", learning.get_threshold()),
                            "min_rr":          sym_p.get("min_rr",          2.0),
                            "adx_threshold":   sym_p.get("adx_threshold",   25),
                            "body_ratio_min":  sym_p.get("body_ratio_min",  0.30),
                        }
                        regime_p = regime.get_regime_params(detected_regime, base_p)
                    except Exception:
                        regime_p = {}

                    htf        = "1h" if interval == "15m" else "1d"
                    htf_d      = bias_cache.get(htf, {})
                    is_day     = interval in ("15m", "30m", "1h")
                    session_ok = (not is_day) or _in_active_session()

                    # ── Build candidate signal list based on regime ───────────
                    # regime_label is TRENDING, RANGING, or UNCERTAIN
                    candidate_sigs = []

                    # (A) BOS trend-following signal — always check but only keep
                    #     in TRENDING or UNCERTAIN (with higher bar).
                    bos_sig = data.get("signal")
                    if bos_sig:
                        regime_threshold = regime_p.get("score_threshold")
                        if regime_threshold and bos_sig.get("score", 0) < regime_threshold:
                            bos_sig = None
                    if bos_sig and regime_label in ("TRENDING", "UNCERTAIN"):
                        candidate_sigs.append(bos_sig)

                    # (B) Mean-reversion signal — only in RANGING regime
                    if regime_label == "RANGING":
                        mr_sig = _mean_reversion_signal(sym, interval, data)
                        if mr_sig:
                            candidate_sigs.append(mr_sig)

                    # (C) Volatility squeeze — all regimes (momentum breakout)
                    sq_sig = _squeeze_signal(sym, interval, data)
                    if sq_sig:
                        candidate_sigs.append(sq_sig)

                    # ── Process each candidate signal ─────────────────────────
                    for sig in candidate_sigs:
                        if not sig or not session_ok:
                            continue

                        sig_type = sig.get("signal_type", "")
                        # Mean-reversion signals skip the HTF bias filter
                        # (they are counter-trend by design)
                        if sig_type not in ("MEAN_REVERSION",) and not _bias_agrees(sig["direction"], htf_d):
                            continue

                        # ── Signal quality gates (Fixes 2, 4, 5, 6) ──────────
                        factors        = sig.get("factors_snapshot", {})
                        score          = sig.get("score", 0)
                        base_threshold = regime_p.get(
                            "score_threshold",
                            sym_p.get("score_threshold", learning.get_threshold()),
                        )

                        # Fix 2: Regime hard-block for BOS trend signals ───────
                        if sig_type not in ("MEAN_REVERSION", "VOL_SQUEEZE", "MACD_EMA_VOL"):
                            if sig["direction"] == "LONG" and not factors.get("regime"):
                                logger.info(
                                    f"[REGIME BLOCK] {sym} {interval}: LONG rejected — regime=False"
                                )
                                continue
                            if sig["direction"] == "SHORT" and not factors.get("regime"):
                                logger.info(
                                    f"[REGIME BLOCK] {sym} {interval}: SHORT rejected — regime=False"
                                )
                                continue

                        # Fix 5: OBV=False raises effective threshold by 1.0 for LONGs ──
                        effective_threshold = base_threshold
                        if sig["direction"] == "LONG" and not factors.get("obv"):
                            effective_threshold += 1.0
                            logger.debug(
                                f"[OBV PENALTY] {sym} {interval}: OBV=False on LONG, "
                                f"threshold raised to {effective_threshold}"
                            )
                        if score < effective_threshold:
                            logger.info(
                                f"[THRESHOLD BLOCK] {sym} {interval}: score={score:.1f} "
                                f"< effective_threshold={effective_threshold:.1f}"
                            )
                            continue

                        # Fix 4: Require smart-money factor for sub-threshold signals ──
                        smart_money_confirmed = (
                            factors.get("fvg") or factors.get("fib") or factors.get("liquidity")
                        )
                        if score < (base_threshold + 1.0) and not smart_money_confirmed:
                            logger.info(
                                f"[SMART-MONEY BLOCK] {sym} {interval}: score={score:.1f} "
                                f"< {base_threshold + 1.0:.1f}, no FVG/FIB/Liquidity — skipping"
                            )
                            continue

                        # Fix 6: Skip same-symbol same-direction duplicate ─────
                        _existing = [
                            t for t in learning.get_open_trades(sym)
                            if t["direction"] == sig["direction"]
                        ]
                        if _existing:
                            logger.info(
                                f"[DUPE BLOCK] {sym} {interval}: {sig['direction']} already "
                                f"open on {_existing[0].get('interval', '?')} — skipping"
                            )
                            continue

                        # ── Fee-adjusted R:R gate ─────────────────────────────
                        net_rr = _rr_after_fees(
                            sig["entry"], sig["target"], sig["stop"], sig["direction"]
                        )
                        min_rr_required = regime_p.get("min_rr", sym_p.get("min_rr", 1.8))
                        if net_rr < min_rr_required:
                            continue   # Skip: spread/fees eat the edge

                        # ── 1H confirmation candle for 4H / 1H swing signals ──
                        # Mean-reversion and squeeze entries skip this filter
                        if sig_type not in ("MEAN_REVERSION", "VOL_SQUEEZE"):
                            if interval == "4h":
                                h1_candles = bias_cache.get("1h", {}).get("chart", {}).get("candles", [])
                                if h1_candles:
                                    h1 = h1_candles[-1]
                                    h1_rng   = h1["high"] - h1["low"]
                                    h1_body  = abs(h1["close"] - h1["open"])
                                    h1_ratio = h1_body / h1_rng if h1_rng > 0 else 0
                                    h1_bull  = h1["close"] > h1["open"]
                                    h1_bear  = h1["close"] < h1["open"]
                                    if sig["direction"] == "LONG"  and not (h1_bull and h1_ratio >= 0.40):
                                        continue
                                    if sig["direction"] == "SHORT" and not (h1_bear and h1_ratio >= 0.40):
                                        continue
                            if interval == "1h":
                                h1_candles = bias_cache.get("1h", {}).get("chart", {}).get("candles", [])
                                if h1_candles:
                                    h1 = h1_candles[-1]
                                    h1_rng   = h1["high"] - h1["low"]
                                    h1_body  = abs(h1["close"] - h1["open"])
                                    h1_ratio = h1_body / h1_rng if h1_rng > 0 else 0
                                    h1_bull  = h1["close"] > h1["open"]
                                    h1_bear  = h1["close"] < h1["open"]
                                    if sig["direction"] == "LONG"  and not (h1_bull and h1_ratio >= 0.35):
                                        continue
                                    if sig["direction"] == "SHORT" and not (h1_bear and h1_ratio >= 0.35):
                                        continue

                        # ── Capital weight: skip if config is in penalty zone ──
                        cap_w = _get_config_weight(sym, interval)
                        if cap_w <= 0.25:
                            # Still log but flag as reduced-size
                            sig = dict(sig)
                            sig["reason"] = f"[size={cap_w:.2f}×] " + sig.get("reason", "")

                        trade_id = f"{sym}_{interval}_{sig['direction']}_{int(time.time())}"
                        cur_px   = sig.get("current_price", sig["entry"])
                        vwap_val = data.get("vwap") or 0
                        is_macd  = sig_type == "MACD_EMA_VOL"
                        # Mean-reversion and squeeze signals are immediate entries
                        is_immediate = is_macd or sig_type in ("MEAN_REVERSION", "VOL_SQUEEZE")

                        trade_data = {
                            "id":               trade_id,
                            "symbol":           sym,
                            "interval":         interval,
                            "direction":        sig["direction"],
                            "entry":            sig["entry"],
                            "current_price":    cur_px,
                            "tp":               sig["target"],
                            "sl":               sig["stop"],
                            "score":            sig["score"],
                            "effective_score":  sig["score"],
                            "reason":           sig.get("reason", ""),
                            "factors_snapshot": sig.get("factors_snapshot", {}),
                            "target_basis":     sig.get("target_basis", ""),
                            "opened_at":        datetime.now(timezone.utc).isoformat(),
                            "status":           "open" if is_immediate else "pending",
                            "adx_value":        data.get("adx", {}).get("value", 0),
                            "vwap_side":        ("above" if vwap_val and cur_px > vwap_val
                                                 else "below"),
                        }
                        logged = learning.log_trade(trade_data)
                        if logged:
                            scan_signals += 1
                            notifications.send_signal_alert({
                                "symbol":        sym,
                                "interval":      interval,
                                "direction":     sig["direction"],
                                "entry":         sig["entry"],
                                "tp":            sig["target"],
                                "sl":            sig["stop"],
                                "score":         sig["score"],
                                "reason":        sig.get("reason", ""),
                                "target_basis":  sig.get("target_basis", ""),
                                "ai_analysis":   {},
                                "pending":       not is_immediate,
                                "current_price": cur_px,
                            })

                    # ── Bar-accurate TP/SL resolution ─────────────────────────
                    scan_closes += _resolve_open_trades(sym, interval)

                    # ── Auto-close trades; record P&L for risk engine ─────────
                    cur_price = data.get("current_price", 0)
                    if cur_price:
                        closed, partials = learning.auto_close(sym, interval, float(cur_price))
                        for p in partials:
                            notifications.send_partial_alert(p, p["partial_price"])
                        for c in closed:
                            notifications.send_close_alert(
                                c, c["status"], c["close_price"], c["roi_pct"]
                            )
                            # Feed closed P&L into risk engine
                            try:
                                _record_trade_pnl(float(c.get("roi_pct") or 0))
                            except Exception:
                                pass
                        scan_closes += len(closed)

                        # Trailing stop: after 2R, trail SL to structural swing
                        swings     = data.get("swings", {})
                        sw_highs   = [v for _, v in swings.get("highs", [])]
                        sw_lows    = [v for _, v in swings.get("lows",  [])]
                        learning.update_trailing_stops(
                            sym, interval, float(cur_price), sw_highs, sw_lows
                        )

                except Exception:
                    pass
                time.sleep(1)   # respect Binance public API rate limits

        # ── Update status and auto-weights after each full sweep ──────────────
        now_iso = datetime.now(timezone.utc).isoformat()
        _scanner_status["scans_completed"] += 1
        _scanner_status["last_scan_at"]    = now_iso
        _scanner_status["signals_logged"]  += scan_signals
        _scanner_status["trades_closed"]   += scan_closes

        # Rebalance per-config capital weights every sweep
        try:
            _update_config_weights()
        except Exception:
            pass

        time.sleep(300)         # 5 minutes between full sweeps


threading.Thread(target=_background_scanner, daemon=True).start()


# ── Weekly review job ─────────────────────────────────────────────────────────

def _weekly_review_job() -> dict:
    """
    Run the weekly Claude strategy review:
      1. Collect all closed trades (min 10 required).
      2. Run backtester baseline on BTCUSDT 4H with current params.
      3. Ask Claude Sonnet for parameter proposals.
      4. Validate each proposal with backtester — deploy if win_rate improves ≥5%
         AND drawdown doesn't worsen more than 10%.
      5. Send Discord summary.

    Returns a summary dict (also used by /api/weekly-review manual trigger).
    """
    trades = learning.get_trades()
    closed = [t for t in trades if t.get("status") in ("win", "loss")]
    if len(closed) < 10:
        return {"status": "skipped", "reason": f"Need ≥10 closed trades (have {len(closed)})"}

    state = learning.get_learning_state()
    current_params = {
        "signal_threshold": state.get("signal_threshold", 7.0),
        "stop_multiplier":  state.get("stop_multiplier",  0.1),
        "adx_threshold":    25,
        "body_ratio_min":   0.30,
        "min_rr":           3.0,
        **{f"weight_{k}": v["current"]
           for k, v in state.get("weights", {}).items()},
    }

    # Baseline backtest (current params, BTCUSDT 4H, 1 year for speed)
    try:
        baseline = backtester.run_backtest("BTCUSDT", "4h", current_params, years=1)
    except Exception as e:
        baseline = {"win_rate": 0, "max_drawdown": 0, "error": str(e)}

    # Claude weekly review
    review = ai_analysis.get_weekly_review(trades, current_params, state)
    if not review or "error" in review:
        return {"status": "error", "reason": review.get("error", "Claude unavailable")}

    # Validate proposals and deploy if backtester confirms improvement
    deployed = []
    for proposal in review.get("parameter_proposals", []):
        try:
            test_params = dict(current_params)
            test_params[proposal["param"]] = float(proposal["proposed"])
            result = backtester.run_backtest("BTCUSDT", "4h", test_params, years=1)

            baseline_wr = baseline.get("win_rate", 0)
            baseline_dd = baseline.get("max_drawdown", 0) or 1.0
            improved_wr = result.get("win_rate", 0) >= baseline_wr + 0.05
            safe_dd     = result.get("max_drawdown", 0) <= baseline_dd * 1.10

            if improved_wr and safe_dd:
                deployed_ok = learning.auto_deploy_params(
                    {proposal["param"]: float(proposal["proposed"])},
                    reason=proposal.get("reason", "Weekly review"),
                    improvement={"baseline": baseline, "proposed": result},
                )
                if deployed_ok:
                    deployed.append(proposal)
        except Exception:
            pass

    notifications.send_weekly_review_alert(review, deployed, baseline)
    return {
        "status":   "completed",
        "deployed": len(deployed),
        "baseline": baseline,
        "review":   review,
        "deployed_params": deployed,
    }


def _weekly_review_thread() -> None:
    """Daemon thread: sleeps until next Sunday noon UTC, then runs the review weekly."""
    from datetime import timedelta
    while True:
        now               = datetime.now(timezone.utc)
        days_to_sunday    = (6 - now.weekday()) % 7
        next_sunday_noon  = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if days_to_sunday == 0 and now.hour >= 12:
            days_to_sunday = 7   # already past noon this Sunday — wait until next
        next_sunday_noon += timedelta(days=days_to_sunday)
        sleep_s = (next_sunday_noon - now).total_seconds()
        time.sleep(max(sleep_s, 1))
        try:
            _weekly_review_job()
        except Exception:
            pass


threading.Thread(target=_weekly_review_thread, daemon=True).start()


# ── Analysis cache ────────────────────────────────────────────────────────────
def cached_analysis(symbol: str, interval: str = "4h") -> dict:
    key = f"{symbol}_{interval}"
    now = time.time()
    with _lock:
        if key in _cache and now - _cache[key][1] < TTL:
            cached, ts = _cache[key]
            cached["cache_age"] = int(now - ts)
            return cached
    data = full_analysis(symbol, interval)
    data["cache_age"] = 0
    with _lock:
        _cache[key] = (data, now)
    return data


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analysis/<symbol>")
@app.route("/api/analysis/<symbol>/<interval>")
def analysis(symbol: str, interval: str = "4h"):
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    _known_symbols.add(sym)     # register so the scanner picks it up too
    try:
        data = cached_analysis(sym, interval)
        if "error" in data:
            return jsonify(data), 400
        _auto_close_from_data(data, sym, interval)   # TP/SL check only; scanner logs signals
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "symbol": sym}), 500


@app.route("/api/refresh/<symbol>")
@app.route("/api/refresh/<symbol>/<interval>")
def refresh(symbol: str, interval: str = "4h"):
    """Force-clear cache and re-fetch."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    with _lock:
        _cache.pop(f"{sym}_{interval}", None)
    return analysis(sym.replace("USDT", ""), interval)


@app.route("/api/chart/<symbol>/<interval>")
def chart_data(symbol, interval):
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval. Use: {VALID_INTERVALS}"}), 400
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    key = f"{sym}_{interval}"
    now = time.time()
    with _clk:
        if key in _chart_cache and now - _chart_cache[key][1] < CHART_TTL:
            return jsonify(_chart_cache[key][0])
    try:
        data = chart_for_timeframe(sym, interval)
        with _clk:
            _chart_cache[key] = (data, now)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades")
def get_trades():
    return jsonify(learning.get_trades())


@app.route("/api/trades/<trade_id>/close", methods=["POST"])
def close_trade(trade_id):
    body   = request.get_json() or {}
    status = body.get("status", "cancelled")
    price  = body.get("price")
    if status not in ("win", "loss", "cancelled"):
        return jsonify({"error": "status must be win, loss, or cancelled"}), 400
    try:
        close_px = float(price) if price else 0.0
        result   = learning.close_trade(trade_id, close_px, status)
        if result is None:
            return jsonify({"error": "trade not found or already closed"}), 404
        notifications.send_close_alert(
            result, status, result.get("close_price", close_px), result.get("roi_pct", 0.0)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning")
def get_learning():
    return jsonify(learning.get_learning_state())


@app.route("/api/symbols")
def symbols():
    return jsonify({"symbols": DEFAULT_SYMBOLS})


@app.route("/api/scanner-status")
def scanner_status():
    st = dict(_scanner_status)
    if st["last_scan_at"]:
        last = datetime.fromisoformat(st["last_scan_at"])
        st["last_scan_ago_s"] = int((datetime.now(timezone.utc) - last).total_seconds())
    return jsonify(st)


@app.route("/api/ai-chart/<symbol>/<interval>")
@app.route("/api/ai-chart/<symbol>")
def ai_chart(symbol: str, interval: str = "4h"):
    """On-demand Claude TA analysis for the currently viewed chart."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    try:
        data   = cached_analysis(sym, interval)
        result = ai_analysis.analyze_chart(sym, interval, data)
        if not result:
            return jsonify({"error": "AI unavailable — check ANTHROPIC_API_KEY"}), 503
        if "_error" in result:
            return jsonify({"error": result["_error"]}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-insights")
def get_ai_insights():
    """Ask Claude Sonnet to analyze full trade history and surface patterns."""
    try:
        history   = learning.get_trades()
        ai_acc    = learning.get_ai_accuracy()
        insights  = ai_analysis.get_performance_insights(history, ai_accuracy=ai_acc)
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/test-discord")
def test_discord():
    """Send a test Discord notification to verify the webhook is configured."""
    import notifications
    if not notifications.WEBHOOK_URL:
        return jsonify({
            "status": "error",
            "message": "DISCORD_WEBHOOK_URL environment variable is not set on Render."
        }), 500
    notifications.send_signal_alert({
        "symbol":       "BTCUSDT",
        "interval":     "15m",
        "direction":    "LONG",
        "entry":        94500.0,
        "tp":           97200.0,
        "sl":           92800.0,
        "score":        8.5,
        "reason":       "TEST — Discord webhook is working correctly ✅",
        "target_basis": "Next major resistance",
    })
    return jsonify({"status": "ok", "message": "Test alert sent to Discord."})


@app.route("/api/test-ai")
def test_ai():
    """Test ANTHROPIC_API_KEY and surface the real error if any."""
    import anthropic as _ant
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return jsonify({"status": "error", "message": "ANTHROPIC_API_KEY env var is not set."}), 500
    try:
        client = _ant.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        return jsonify({"status": "ok", "message": "API key is valid and working.",
                        "model": "claude-haiku-4-5-20251001"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ── Backtester routes ─────────────────────────────────────────────────────────

@app.route("/api/backtest/run", methods=["POST"])
def backtest_run():
    """
    Start an async backtest job.
    Body (JSON): {symbol, interval, years=2, param_grid=null}
      - param_grid=null → single run with default params
      - param_grid={...} → grid search over all combinations

    Returns: {job_id, status: "started", message}
    """
    body       = request.get_json() or {}
    symbol     = body.get("symbol", "BTCUSDT").upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    interval   = body.get("interval", "4h")
    years      = int(body.get("years", 2))
    param_grid = body.get("param_grid")   # None → single default run

    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval. Use: {VALID_INTERVALS}"}), 400

    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _backtest_jobs[job_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    def _run():
        try:
            if param_grid:
                result = backtester.grid_search(symbol, interval, param_grid, years)
            else:
                result = [backtester.run_backtest(
                    symbol, interval, backtester.DEFAULT_PARAMS, years=years
                )]
            with _bt_lock:
                _backtest_jobs[job_id] = {
                    "status":       "done",
                    "results":      result,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            with _bt_lock:
                _backtest_jobs[job_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started",
                    "message": f"Backtesting {symbol} {interval} ({years}yr). Poll /api/backtest/status/{job_id}"})


@app.route("/api/backtest/status/<job_id>")
def backtest_status(job_id: str):
    with _bt_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({"job_id": job_id, "status": job["status"]})


@app.route("/api/backtest/results/<job_id>")
def backtest_results(job_id: str):
    with _bt_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job["status"] == "running":
        return jsonify({"status": "running", "message": "Still in progress"}), 202
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job.get("error")}), 500
    return jsonify({"status": "done", "results": job.get("results", [])})


# ── MACD walk-forward backtest route ──────────────────────────────────────────

@app.route("/api/backtest/macd", methods=["POST"])
def backtest_macd():
    """
    Start an async MACD walk-forward backtest job.
    Body: {symbol, interval, years=3}
    Returns: {job_id, status: "started"}
    """
    body     = request.get_json() or {}
    symbol   = body.get("symbol", "BTCUSDT").upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    interval = body.get("interval", "1h")
    years    = int(body.get("years", 3))

    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval. Use: {VALID_INTERVALS}"}), 400

    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _backtest_jobs[job_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    def _run():
        try:
            result = backtester.run_macd_backtest(symbol, interval, years)
            with _bt_lock:
                _backtest_jobs[job_id] = {
                    "status":       "done",
                    "results":      [result],
                    "macd":         True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            with _bt_lock:
                _backtest_jobs[job_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started"})


# ── Research campaign routes ──────────────────────────────────────────────────

@app.route("/api/backtest/campaign", methods=["POST"])
def backtest_campaign():
    """
    Launch a full research campaign: grid search for each symbol × interval pair.

    Body (JSON, all optional):
        symbols   — list of symbols (default: CAMPAIGN_SYMBOLS)
        intervals — list of intervals (default: CAMPAIGN_INTERVALS)

    Returns {campaign_id, jobs: {sym_interval: {job_id, status}}}
    Each job uses CAMPAIGN_GRID and MAX_HISTORY_YEARS[interval] for data depth.
    """
    body      = request.get_json() or {}
    symbols   = body.get("symbols",   backtester.CAMPAIGN_SYMBOLS)
    intervals = body.get("intervals", backtester.CAMPAIGN_INTERVALS)

    # Normalise symbols
    symbols = [s.upper() if s.upper().endswith("USDT") else s.upper() + "USDT"
               for s in symbols]

    campaign_id = str(uuid.uuid4())[:8]
    jobs        = {}

    for sym in symbols:
        for interval in intervals:
            job_id = str(uuid.uuid4())[:8]
            years  = backtester.MAX_HISTORY_YEARS.get(interval, 2)
            key    = f"{sym}_{interval}"
            jobs[key] = {"job_id": job_id, "status": "running",
                         "symbol": sym, "interval": interval}

            with _bt_lock:
                _backtest_jobs[job_id] = {
                    "status":     "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }

            def _run(s=sym, iv=interval, jid=job_id, yr=years):
                try:
                    result = backtester.grid_search(
                        s, iv, backtester.CAMPAIGN_GRID, years=yr
                    )
                    with _bt_lock:
                        _backtest_jobs[jid] = {
                            "status":       "done",
                            "results":      result,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                except Exception as e:
                    with _bt_lock:
                        _backtest_jobs[jid] = {"status": "error", "error": str(e)}

            threading.Thread(target=_run, daemon=True).start()

    with _camp_lock:
        _campaigns[campaign_id] = {
            "jobs":       jobs,
            "symbols":    symbols,
            "intervals":  intervals,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    return jsonify({
        "campaign_id": campaign_id,
        "jobs":        jobs,
        "status":      "started",
        "message":     (f"Campaign started: {len(symbols)} symbols × "
                        f"{len(intervals)} intervals = {len(jobs)} jobs running in parallel."),
    })


@app.route("/api/backtest/campaign/status/<campaign_id>")
def campaign_status(campaign_id: str):
    """Return per-job completion status for a running campaign."""
    with _camp_lock:
        camp = _campaigns.get(campaign_id)
    if not camp:
        return jsonify({"error": "campaign not found"}), 404

    jobs_out = {}
    with _bt_lock:
        for key, meta in camp["jobs"].items():
            job = _backtest_jobs.get(meta["job_id"], {})
            jobs_out[key] = {
                "job_id":   meta["job_id"],
                "symbol":   meta["symbol"],
                "interval": meta["interval"],
                "status":   job.get("status", "unknown"),
            }

    all_done = all(v["status"] in ("done", "error") for v in jobs_out.values())
    return jsonify({
        "campaign_id": campaign_id,
        "status":      "done" if all_done else "running",
        "jobs":        jobs_out,
        "started_at":  camp["started_at"],
    })


@app.route("/api/backtest/campaign/results/<campaign_id>")
def campaign_results(campaign_id: str):
    """Return best params + top-5 results for each completed job in the campaign."""
    with _camp_lock:
        camp = _campaigns.get(campaign_id)
    if not camp:
        return jsonify({"error": "campaign not found"}), 404

    results_out = {}
    with _bt_lock:
        for key, meta in camp["jobs"].items():
            job = _backtest_jobs.get(meta["job_id"], {})
            st  = job.get("status", "unknown")
            if st == "done":
                all_r = [r for r in (job.get("results") or []) if not r.get("error")]
                best  = all_r[0] if all_r else None
                results_out[key] = {
                    "status":   "done",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                    "best":     best,
                    "top5":     all_r[:5],
                }
            elif st == "error":
                results_out[key] = {
                    "status":   "error",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                    "error":    job.get("error"),
                }
            else:
                results_out[key] = {
                    "status":   "running",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                }

    all_done = all(v["status"] in ("done", "error") for v in results_out.values())
    return jsonify({
        "campaign_id": campaign_id,
        "status":      "done" if all_done else "running",
        "results":     results_out,
    })


@app.route("/api/apply-symbol-params", methods=["POST"])
def apply_symbol_params():
    """
    Save approved backtest params for a specific symbol+interval.
    The scanner will use these on its next sweep instead of global defaults.

    Body: {symbol, interval, params: {adx_threshold, score_threshold, min_rr, ...}}
    """
    body     = request.get_json() or {}
    symbol   = body.get("symbol", "").upper()
    interval = body.get("interval", "")
    params   = body.get("params", {})

    if not symbol or not interval or not params:
        return jsonify({"error": "symbol, interval, and params are required"}), 400
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"interval must be one of {VALID_INTERVALS}"}), 400

    learning.save_symbol_params(symbol, interval, params)
    return jsonify({
        "status":   "saved",
        "symbol":   symbol,
        "interval": interval,
        "params":   params,
        "message":  f"Scanner will now use these params for {symbol} {interval}.",
    })


# ── Regime route ──────────────────────────────────────────────────────────────

@app.route("/api/regime/<symbol>")
def get_regime(symbol: str):
    """Return the most recently detected market regime for a symbol."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    stored = learning.get_regime(sym)
    if stored:
        return jsonify({"symbol": sym, **stored})
    # Compute on-demand if not yet detected
    try:
        data = cached_analysis(sym, "4h")
        detected = regime.detect_regime_from_data(data)
        learning.save_regime(sym, detected)
        return jsonify({"symbol": sym, **detected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Discovery progress ────────────────────────────────────────────────────────

@app.route("/api/discovery/progress")
def discovery_progress():
    """Read per-job progress files written by run_discovery.py workers."""
    import glob as _glob
    progress_dir = "/tmp/discovery_progress"
    files = _glob.glob(f"{progress_dir}/*.json")
    jobs = {}
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            key = os.path.basename(f).replace(".json", "")
            jobs[key] = data
        except Exception:
            pass
    total_done  = sum(j.get("done", 0)  for j in jobs.values())
    total_total = sum(j.get("total", 0) for j in jobs.values())
    all_done    = bool(jobs) and all(j.get("status") in ("done", "error") for j in jobs.values())
    return jsonify({
        "jobs":        jobs,
        "total_done":  total_done,
        "total_total": total_total,
        "overall_pct": round(total_done / total_total * 100, 1) if total_total else 0,
        "all_done":    all_done,
        "running":     bool(jobs) and not all_done,
    })


# ── Weekly review manual trigger ──────────────────────────────────────────────

@app.route("/api/weekly-review")
def weekly_review():
    """Manually trigger the weekly Claude strategy review (synchronous)."""
    try:
        result = _weekly_review_job()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Risk Engine status route ───────────────────────────────────────────────────

@app.route("/api/risk-status")
def risk_status():
    """Return current risk engine state: daily P&L, drawdown, halt status."""
    _reset_daily_risk()
    with _risk_lock:
        state = dict(_risk_state)
    state["gate_open"]        = _risk_gate_open()
    state["position_factor"]  = _position_size_factor()
    state["daily_pnl_pct"]    = round(state["daily_pnl_pct"]    * 100, 3)
    state["current_equity_pct"] = round(state["current_equity_pct"] * 100, 3)
    state["peak_equity_pct"]  = round(state["peak_equity_pct"]  * 100, 3)
    peak  = _risk_state["peak_equity_pct"]
    cur   = _risk_state["current_equity_pct"]
    dd    = (peak - cur) / (1.0 + abs(peak)) if peak > 0 else 0.0
    state["drawdown_pct"]     = round(dd * 100, 3)
    state["limits"] = {
        "daily_loss_halt_pct": DAILY_LOSS_HALT_PCT * 100,
        "drawdown_reduce_pct": DRAWDOWN_REDUCE_PCT * 100,
        "risk_per_trade_pct":  RISK_PER_TRADE_PCT  * 100,
        "fee_slippage_pct":    FEE_SLIPPAGE_PCT    * 100,
    }
    return jsonify(state)


# ── Config weights route ───────────────────────────────────────────────────────

@app.route("/api/config-weights")
def config_weights():
    """Return per-symbol/interval capital weight multipliers."""
    with _config_weights_lock:
        weights = {f"{s}_{iv}": w for (s, iv), w in _config_weights.items()}
    return jsonify({
        "weights":     weights,
        "description": "1.5× = outperforming (scale up), 1.0× = neutral, 0.25× = underperforming (scale down)",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  Crypto Analysis Dashboard")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)

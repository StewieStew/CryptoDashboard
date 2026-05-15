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

VALID_INTERVALS  = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]
DEFAULT_SYMBOLS  = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT",  "DOGEUSDT",
    "SOLUSDT", "BNBUSDT", "ADAUSDT",  "AVAXUSDT",
    "LINKUSDT","LTCUSDT", "DOTUSDT",  "NEARUSDT", "ATOMUSDT",
]

# Paper-trade mode: only these coins on 4H using discovery-validated params.
PAPER_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]
PAPER_INTERVALS = ["5m", "15m", "1h", "4h"]

# Four-tier scanning: 5m (micro-scalp), 15m (scalp), 1h (day trade), 4h (swing).
SCAN_INTERVALS   = ["5m", "15m", "1h", "4h"]

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
FEE_RATE            = 0.002  # 0.1% per side × 2 sides = 0.2% round-trip fee gate


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
        # ── 5m micro-scalp params (conservative start — learning engine adapts) ──
        ("BTCUSDT", "5m"): {
            "config":           "Structure + Volume",
            "score_threshold":  7.5,
            "min_rr":           2.5,
            "adx_threshold":    30,
            "body_ratio_min":   0.35,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "volume": 2.5, "obv": 1.0,
                "regime": 1.5, "rsi": 0.0, "adx": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("ETHUSDT", "5m"): {
            "config":           "Structure + Volume",
            "score_threshold":  8.0,
            "min_rr":           2.5,
            "adx_threshold":    30,
            "body_ratio_min":   0.25,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "volume": 2.5, "obv": 1.0,
                "regime": 1.5, "rsi": 0.0, "adx": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
            },
        },
        ("XRPUSDT", "5m"): {
            "config":           "Structure + Volume",
            "score_threshold":  7.5,
            "min_rr":           2.5,
            "adx_threshold":    20,
            "body_ratio_min":   0.25,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "volume": 2.0, "obv": 1.0,
                "regime": 2.0, "rsi": 1.0, "adx": 0.0,
                "fvg": 0.0, "fib": 0.0, "liquidity": 0.0,
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
        ("DOGEUSDT", "5m"): {
            "config":           "Structure + Volume",
            "score_threshold":  7.0,
            "min_rr":           2.5,
            "adx_threshold":    20,
            "body_ratio_min":   0.25,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.5, "sweep": 2.0, "volume": 2.0, "obv": 1.0,
                "regime": 1.5, "rsi": 1.0, "adx": 0.0,
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
        ("DOGEUSDT", "1h"): {
            "config":           "Trend + S/R + RSI",
            "score_threshold":  6.5,
            "min_rr":           2.0,
            "adx_threshold":    20,
            "body_ratio_min":   0.20,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 1.5, "rsi": 2.0, "regime": 2.0,
                "adx": 0.0, "volume": 1.0, "obv": 0.0,
                "fvg": 0.0, "fib": 0.5, "liquidity": 0.0,
            },
        },
        ("DOGEUSDT", "4h"): {
            "config":           "Trend + S/R + RSI",
            "score_threshold":  6.0,
            "min_rr":           2.0,
            "adx_threshold":    20,
            "body_ratio_min":   0.20,
            "level_touch_min":  1,
            "weights": {
                "bos": 2.0, "sweep": 1.5, "rsi": 2.0, "regime": 2.0,
                "adx": 0.0, "volume": 1.0, "obv": 0.0,
                "fvg": 0.0, "fib": 0.5, "liquidity": 0.0,
            },
        },
    }
    for (sym, interval), params in best.items():
        learning.save_symbol_params(sym, interval, params)

_apply_discovery_params()


# ── Session filter ────────────────────────────────────────────────────────────
def _in_active_session() -> bool:
    """Crypto markets are 24/7 — always return True."""
    return True


# ── Historical backfill: replay 1m bars from trade open until now ─────────────

def _backfill_trade(trade: dict) -> bool:
    """
    Fetch 1m candles from Binance from this trade's opened_at until now and
    replay bar-by-bar to detect any TP or SL hit that was missed while the bot
    was down or the trade was pending.  Returns True if the trade was closed.

    SL outcome mirrors auto_close logic:
      - breakeven not active          → loss
      - breakeven active, SL > entry  → win  (trailing SL locked in profit)
      - breakeven active, SL == entry → win  (breakeven hit, treated as managed exit)
    """
    trade_id  = trade["id"]
    direction = trade["direction"]
    tp        = float(trade["tp"])
    sl        = float(trade["sl"])   # current SL — may have been trailed
    entry     = float(trade["entry"])
    be_active = bool(trade.get("breakeven_activated", 0))
    sym       = trade["symbol"]

    try:
        # Add 60s offset so the backfill starts on the NEXT complete 1m candle
        # after entry.  The candle that was forming when the trade opened may
        # have started before the entry price was live, which could produce a
        # false TP/SL hit on the very first bar.
        since_ms = int(pd.Timestamp(trade["opened_at"]).timestamp() * 1000) + 60_000
    except Exception:
        return False

    bars = market_data.fetch_1m_bars_since(sym, since_ms)
    if not bars:
        return False

    interval      = trade["interval"]
    _let_it_run   = (interval != "5m")   # 5m closes at TP; others let it run
    tp_reached    = bool(trade.get("tp_reached", 0))
    initial_sl    = float(trade.get("initial_sl") or sl)
    risk_d_bf     = abs(entry - initial_sl)

    outcome     = None
    close_price = None
    for bar in bars:
        hi, lo = bar["high"], bar["low"]
        if direction == "LONG":
            if not tp_reached and hi >= tp:
                if _let_it_run and risk_d_bf > 0:
                    # Switch to let-it-run: lock SL at 2R and keep replaying
                    sl          = round(entry + 2.0 * risk_d_bf, 8)
                    be_active   = True
                    tp_reached  = True
                    continue
                else:
                    outcome = "win";  close_price = tp;  break
            if lo <= sl:
                close_price = sl
                if not be_active:
                    outcome = "loss"
                else:
                    outcome = "win"
                break
        else:  # SHORT
            if not tp_reached and lo <= tp:
                if _let_it_run and risk_d_bf > 0:
                    sl          = round(entry - 2.0 * risk_d_bf, 8)
                    be_active   = True
                    tp_reached  = True
                    continue
                else:
                    outcome = "win";  close_price = tp;  break
            if hi >= sl:
                close_price = sl
                if not be_active:
                    outcome = "loss"
                else:
                    outcome = "win"
                break

    if not outcome:
        print(f"[BACKFILL] {sym} {trade['interval']}: still open, handed to monitor", flush=True)
        return False

    try:
        closed = learning.close_trade(trade_id, close_price, outcome)
        if closed:
            label = "TP hit" if outcome == "win" else "SL hit"
            print(
                f"[BACKFILL] {sym} {trade['interval']}: {label} → {outcome.upper()}"
                f" @ {close_price} ({closed.get('roi_pct', 0):+.2f}%)",
                flush=True,
            )
            notifications.send_close_alert(
                closed, closed["status"], closed["close_price"], closed.get("roi_pct", 0)
            )
            _record_trade_pnl(float(closed.get("roi_pct") or 0))
            return True
    except Exception:
        pass
    return False


def _backfill_all_open_trades() -> int:
    """Replay 1m history for every open trade. Called once at bot startup."""
    trades = learning.get_open_trades()
    if not trades:
        print("[BACKFILL] no open/pending trades to backfill", flush=True)
        return 0
    print(f"[BACKFILL] checking {len(trades)} open/pending trade(s) for missed TP/SL hits...", flush=True)
    closed_count = 0
    for trade in trades:
        if _backfill_trade(trade):
            closed_count += 1
        time.sleep(0.3)   # mild pacing between Binance calls
    print(f"[BACKFILL] complete — resolved {closed_count}/{len(trades)} trade(s)", flush=True)
    return closed_count


# ── Real-time TP/SL monitor: runs every 60 s ─────────────────────────────────

def _monitor_open_trades() -> int:
    """
    Fetch the live Binance price for every open trade's symbol and immediately
    close any trade whose TP or SL has been crossed.  Returns closes count.

    Also fetches the last completed 1m candle high/low per symbol so brief
    TP/SL wicks between the 60-second polls are not missed.
    """
    trades = learning.get_open_trades()
    if not trades:
        return 0

    symbols = {t["symbol"] for t in trades}
    prices: dict[str, float] = {}
    for sym in symbols:
        px = market_data.get_live_price(sym)
        if px:
            prices[sym] = px

    if not prices:
        return 0

    # Candle extremes are now computed per-trade inside auto_close() using
    # each trade's opened_at as the since_ms filter — prevents pre-open
    # candles from falsely triggering SL/TP hits.
    checked: set[tuple[str, str]] = set()
    total_closed = 0
    for t in trades:
        sym, intv = t["symbol"], t["interval"]
        if (sym, intv) in checked:
            continue
        checked.add((sym, intv))
        px = prices.get(sym)
        if not px:
            continue
        print(
            f"[MONITOR] {sym} {intv}: live={px:.4f}  "
            f"TP={t['tp']}  SL={t['sl']}",
            flush=True,
        )
        closed, partials = learning.auto_close(sym, intv, px)
        for p in partials:
            notifications.send_partial_alert(p, p["partial_price"])
        for c in closed:
            notifications.send_close_alert(c, c["status"], c["close_price"], c.get("roi_pct", 0))
            _record_trade_pnl(float(c.get("roi_pct") or 0))
            print(
                f"[MONITOR CLOSE] {sym} {intv}: {c['direction']}"
                f" → {c['status']} @ {c['close_price']} ({c.get('roi_pct', 0):+.2f}%)",
                flush=True,
            )
        total_closed += len(closed)

    return total_closed


def _price_monitor_loop() -> None:
    """Lightweight 60-second loop: fetch live prices and close any TP/SL hits."""
    time.sleep(15)   # stagger start so scanner initialises first
    while True:
        try:
            _monitor_open_trades()
        except Exception as exc:
            print(f"[MONITOR ERROR] {exc}", flush=True)
        time.sleep(60)


# ── Price-based auto-close helper ────────────────────────────────────────────
def _auto_close_from_data(data: dict, sym: str, interval: str) -> None:
    """Auto-close trades whose TP/SL has been hit based on current price.
    Signal logging is handled ONLY by the background scanner (with bias filter).
    """
    cur_price = data.get("current_price", 0)
    if cur_price:
        learning.auto_close(sym, interval, float(cur_price))


# ── Higher-TF bias filter ─────────────────────────────────────────────────────
def _bias_agrees(signal_dir: str, htf_data: dict, strict: bool = True) -> bool:
    """
    Check whether the higher-timeframe context agrees with a signal direction.

    strict=True  (5m, 15m): requires EITHER above/below 200 EMA OR BOS on HTF.
                 These short TFs are noisy so we demand explicit HTF confirmation.

    strict=False (1h, 4h):  only blocks if HTF is actively against the trade —
                 i.e. price is clearly above 200 EMA on a SHORT, or clearly
                 below on a LONG, with NO BOS at all supporting the direction.
                 1h/4h signals have their own structural filters so we just
                 avoid trading directly against the HTF trend, not demand it.

    Returns True if the trade is allowed.
    """
    regime    = htf_data.get("regime", {})
    structure = htf_data.get("structure", {})
    above_200 = regime.get("above_200", False)
    bull_bos  = structure.get("bullish_bos", False) if structure else False
    bear_bos  = structure.get("bearish_bos", False) if structure else False

    if strict:
        # Must have explicit HTF agreement
        if signal_dir == "LONG":
            return above_200 or bull_bos
        else:
            return (not above_200) or bear_bos
    else:
        # Only block if HTF is clearly opposed AND no supporting BOS
        if signal_dir == "LONG":
            htf_opposed = (not above_200) and not bull_bos
            return not htf_opposed
        else:
            htf_opposed = above_200 and not bear_bos
            return not htf_opposed


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
    lookback = {"5m": 500, "15m": 500, "1h": 220, "4h": 120}.get(interval, 220)
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

        # Examine bars from the candle that was forming when the trade opened onward.
        # Using >= (floored to candle boundary) so the opening candle is included —
        # its high/low reflects any wick that hit SL/TP on the same candle as entry.
        _freq_map = {
            "1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
            "1h":"1h","2h":"2h","4h":"4h","1d":"1D",
        }
        _freq = _freq_map.get(trade["interval"], "1h")
        try:
            _candle_start = opened_at.floor(_freq)
        except Exception:
            _candle_start = opened_at
        future = df[df.index >= _candle_start]
        if future.empty:
            continue

        outcome     = None
        close_price = None
        for _, bar in future.iterrows():
            hi, lo = float(bar["high"]), float(bar["low"])
            if direction == "LONG":
                if hi >= tp:
                    outcome = "win";  close_price = tp;  break
                if lo <= sl:
                    outcome = "loss"; close_price = sl;  break
            else:  # SHORT
                if lo <= tp:
                    outcome = "win";  close_price = tp;  break
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

    # Backfill any trades that may have hit TP/SL while the bot was offline
    _backfill_all_open_trades()

    _warmup_complete = False   # first cycle is observation-only; no new trades logged

    while True:
        syms = PAPER_SYMBOLS   # paper-trade: locked to discovery-validated coins only
        scan_signals = 0
        scan_closes  = 0

        # ── Daily risk reset ──────────────────────────────────────────────────
        _reset_daily_risk()
        _cycle_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[SCAN CYCLE] {_cycle_ts} — scanning {len(syms) * len(PAPER_INTERVALS)} configs", flush=True)

        # ── Real-time TP/SL check before scanning for new signals ────────────
        scan_closes += _monitor_open_trades()

        for sym in syms:
            # Fetch bias TFs: 15m for 5m signals, 1h for 15m signals, 1d for 4H bias
            bias_cache: dict[str, dict] = {}
            for bias_tf in ("15m", "1h", "1d"):
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
                    data = full_analysis(sym, interval, weights=sym_weights,
                                        score_threshold=sym_p.get("score_threshold", 3.0))
                    now  = time.time()
                    data["cache_age"] = 0
                    with _lock:
                        _cache[f"{sym}_{interval}"] = (data, now)

                    # ── Detect per-symbol market regime from already-computed data ──
                    regime_label = "UNCERTAIN"
                    try:
                        detected_regime = regime.detect_regime_from_data(data)
                        _raw = detected_regime.get("regime", "transitioning")
                        regime_label = {"trending": "TRENDING", "ranging": "RANGING"}.get(_raw, "UNCERTAIN")
                        learning.save_regime(sym, detected_regime)
                        base_p = {
                            "score_threshold": sym_p.get("score_threshold", learning.get_threshold()),
                            "min_rr":          sym_p.get("min_rr",          2.0),
                            "adx_threshold":   sym_p.get("adx_threshold",   25),
                            "body_ratio_min":  sym_p.get("body_ratio_min",  0.30),
                        }
                        # 15m scalp cap: prevent elevated adaptive params from silencing
                        # all signals. VOL_SQUEEZE has hardcoded score=7.0 and internal
                        # net_rr≥1.8 — both gates must stay at/below these values to fire.
                        if interval == "15m":
                            base_p["score_threshold"] = min(base_p["score_threshold"], 7.0)
                            base_p["min_rr"]          = min(base_p["min_rr"],          1.5)
                        regime_p = regime.get_regime_params(detected_regime, base_p)
                    except Exception:
                        regime_p = {}

                    htf        = "15m" if interval == "5m" else ("1h" if interval == "15m" else "1d")
                    htf_d      = bias_cache.get(htf, {})
                    session_ok = True  # crypto is 24/7

                    # ── Build candidate signal list based on regime ───────────
                    # regime_label is TRENDING, RANGING, or UNCERTAIN
                    candidate_sigs = []

                    # (A) BOS trend-following signal — always check but only keep
                    #     in TRENDING or UNCERTAIN (with higher bar).
                    bos_sig = data.get("signal")
                    # Checklist gate: BOS + trending regime + OBV required
                    if bos_sig:
                        _factors = bos_sig.get("factors_snapshot", {})
                        _checklist = (
                            _factors.get("bos") and      # Break of structure confirmed
                            _factors.get("regime") and   # Trending/aligned market
                            _factors.get("obv")          # OBV confirming direction
                        )
                        if not _checklist:
                            _sc = bos_sig.get("score", 0)
                            _missing = [k for k in ["bos", "regime", "obv"] if not _factors.get(k)]
                            print(f"[CHECKLIST BLOCK] {sym} {interval}: score={_sc:.1f} missing={_missing}", flush=True)
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
                    if not candidate_sigs:
                        _conf  = data.get("confluence", {})
                        _sc    = _conf.get("score", 0.0)
                        _snap  = _conf.get("factors_snapshot", {})
                        _braw  = data.get("signal")   # signal that was dropped (wrong regime / threshold)
                        if _braw and regime_label not in ("TRENDING", "UNCERTAIN"):
                            print(f"[SCAN] {sym} {interval}: BOS score={_sc:.1f} — regime={regime_label} → dropped (need TRENDING/UNCERTAIN)", flush=True)
                        else:
                            print(f"[SCAN] {sym} {interval}: no signal — regime={regime_label} score={_sc:.1f} bos={_snap.get('bos')} adx={_snap.get('adx')}", flush=True)
                    for sig in candidate_sigs:
                        if not sig:
                            continue

                        sig_type = sig.get("signal_type", "")
                        # Mean-reversion signals skip the HTF bias filter
                        # (they are counter-trend by design).
                        # 5m/15m use strict bias (must have explicit HTF agreement).
                        # 1h/4h use relaxed bias (only block if HTF is clearly opposed).
                        _bias_strict = interval in ("5m", "15m")
                        if sig_type not in ("MEAN_REVERSION",) and not _bias_agrees(sig["direction"], htf_d, strict=_bias_strict):
                            print(f"[SCAN] {sym} {interval}: {sig['direction']} score={sig.get('score',0):.1f} — HTF BIAS BLOCK", flush=True)
                            continue

                        # ── Signal quality gates (Fixes 2, 4, 5, 6) ──────────
                        factors        = sig.get("factors_snapshot", {})
                        score          = sig.get("score", 0)
                        base_threshold = regime_p.get(
                            "score_threshold",
                            sym_p.get("score_threshold", learning.get_threshold()),
                        )

                        # Fix 2: Regime penalty for BOS trend signals (was hard-block) ──
                        if sig_type not in ("MEAN_REVERSION", "VOL_SQUEEZE", "MACD_EMA_VOL"):
                            if not factors.get("regime"):
                                score -= 1.5
                                print(f"[SCAN] {sym} {interval}: {sig['direction']} score→{score:.1f} — REGIME PENALTY (-1.5)", flush=True)

                        # Fix 5: OBV=False raises effective threshold by 1.0 for LONGs ──
                        effective_threshold = base_threshold
                        if sig["direction"] == "LONG" and not factors.get("obv"):
                            effective_threshold += 1.0
                            print(f"[SCAN] {sym} {interval}: OBV penalty → threshold={effective_threshold:.1f}", flush=True)

                        # TP-source self-learning: raise threshold for historically poor sources ──
                        _tp_src = sig.get("tp_source", "unknown")
                        _tp_adj = learning.tp_source_threshold_adjustment(_tp_src)
                        if _tp_adj > 0:
                            effective_threshold += _tp_adj
                            print(f"[SCAN] {sym} {interval}: tp_source='{_tp_src}' penalty +{_tp_adj:.1f} → threshold={effective_threshold:.1f}", flush=True)

                        if score < effective_threshold:
                            print(f"[SCAN] {sym} {interval}: {sig['direction']} score={score:.1f} — THRESHOLD BLOCK (need {effective_threshold:.1f})", flush=True)
                            continue

                        # Fix 4: Smart-money gate removed — score threshold is the only gate

                        # Skip if there's already an open or pending trade for
                        # this exact symbol + interval + direction
                        if learning.has_active_trade(sym, interval, sig["direction"]):
                            logger.info(
                                f"[SKIP] already have open {sym} {interval} "
                                f"{sig['direction']} — skipping"
                            )
                            continue

                        # R:R is logged but no longer used as a gate.

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
                                        print(f"[SCAN] {sym} {interval}: LONG — 1H CANDLE BLOCK (ratio={h1_ratio:.2f} bull={h1_bull})", flush=True)
                                        continue
                                    if sig["direction"] == "SHORT" and not (h1_bear and h1_ratio >= 0.40):
                                        print(f"[SCAN] {sym} {interval}: SHORT — 1H CANDLE BLOCK (ratio={h1_ratio:.2f} bear={h1_bear})", flush=True)
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
                                        print(f"[SCAN] {sym} {interval}: LONG — 1H CANDLE BLOCK (ratio={h1_ratio:.2f} bull={h1_bull})", flush=True)
                                        continue
                                    if sig["direction"] == "SHORT" and not (h1_bear and h1_ratio >= 0.35):
                                        print(f"[SCAN] {sym} {interval}: SHORT — 1H CANDLE BLOCK (ratio={h1_ratio:.2f} bear={h1_bear})", flush=True)
                                        continue

                        # ── Capital weight: skip if config is in penalty zone ──
                        cap_w = _get_config_weight(sym, interval)
                        if cap_w <= 0.25:
                            # Still log but flag as reduced-size
                            sig = dict(sig)
                            sig["reason"] = f"[size={cap_w:.2f}×] " + sig.get("reason", "")

                        if not _warmup_complete:
                            print(f"[WARMUP] {sym} {interval}: {sig['direction']} score={sig.get('score',0):.1f} — observation only, skipping log", flush=True)
                            continue

                        trade_id = f"{sym}_{interval}_{sig['direction']}_{int(time.time())}"

                        # ── Fetch live price for accurate entry recording ────
                        # sig["entry"] is computed at analysis time (candle close
                        # price), which may be minutes stale by the time all gates
                        # pass.  The actual fill price for live trading is the
                        # current market price — always use that for entry.
                        actual_px = market_data.get_live_price(sym)
                        if actual_px is None:
                            print(f"[SKIP] {sym} {interval}: live price unavailable at entry", flush=True)
                            continue

                        _sig_entry = float(sig["entry"])
                        _tp        = float(sig["target"])
                        _sl        = float(sig["stop"])
                        _dir       = sig["direction"]

                        # ── Stale signal guard ───────────────────────────────
                        # If price has already crossed the SL, the signal setup
                        # is invalidated — don't enter a trade that's already at
                        # a loss.
                        if _dir == "LONG" and actual_px <= _sl:
                            print(f"[SKIP STALE] {sym} {interval}: live {actual_px:.4f} already past SL {_sl:.4f}", flush=True)
                            continue
                        if _dir == "SHORT" and actual_px >= _sl:
                            print(f"[SKIP STALE] {sym} {interval}: live {actual_px:.4f} already past SL {_sl:.4f}", flush=True)
                            continue

                        # ── R:R gate using actual live entry ─────────────────
                        # Recalculate R:R from the actual fill price, not the
                        # stale signal entry.  TP and SL are structural levels
                        # and don't change.
                        if actual_px != _sl:
                            if _dir == "LONG":
                                _rr = (_tp - actual_px) / (actual_px - _sl)
                            else:
                                _rr = (actual_px - _tp) / (_sl - actual_px)
                            if _rr < 3.0:
                                print(f"[SKIP RR] {sym} {interval}: R:R={_rr:.2f} < 3.0 at live px {actual_px:.4f}, skipping", flush=True)
                                continue

                        # ── Fee gate using actual live entry ─────────────────
                        if actual_px > 0:
                            if _dir == "LONG":
                                _net_pct = (_tp - actual_px) / actual_px - FEE_RATE
                            else:
                                _net_pct = (actual_px - _tp) / actual_px - FEE_RATE
                            if _net_pct <= 0:
                                print(
                                    f"[SKIP FEE] {sym} {interval}: TP distance "
                                    f"{abs(_tp - actual_px) / actual_px * 100:.3f}% doesn't cover "
                                    f"fees (0.2%), skipping",
                                    flush=True,
                                )
                                continue

                        vwap_val = data.get("vwap") or 0
                        is_immediate = True   # all signals enter immediately at market price

                        trade_data = {
                            "id":               trade_id,
                            "symbol":           sym,
                            "interval":         interval,
                            "direction":        sig["direction"],
                            "entry":            actual_px,          # live price at fill time
                            "current_price":    actual_px,
                            "tp":               _tp,
                            "sl":               _sl,
                            "score":            sig["score"],
                            "effective_score":  sig["score"],
                            "reason":           sig.get("reason", ""),
                            "factors_snapshot": sig.get("factors_snapshot", {}),
                            "target_basis":     sig.get("target_basis", ""),
                            "tp_source":        sig.get("tp_source", "unknown"),
                            "opened_at":        datetime.now(timezone.utc).isoformat(),
                            "status":           "open" if is_immediate else "pending",
                            "adx_value":        data.get("adx", {}).get("value", 0),
                            "vwap_side":        ("above" if vwap_val and actual_px > vwap_val
                                                 else "below"),
                        }

                        logged = learning.log_trade(trade_data)
                        if logged:
                            scan_signals += 1
                            print(f"[SIGNAL] {sym} {interval}: {sig['direction']} score={sig['score']:.1f}"
                                  f" entry={actual_px:.4f} (signal={_sig_entry:.4f}) tp={_tp} sl={_sl}", flush=True)
                            notifications.send_signal_alert({
                                "symbol":        sym,
                                "interval":      interval,
                                "direction":     sig["direction"],
                                "entry":         actual_px,
                                "tp":            _tp,
                                "sl":            _sl,
                                "score":         sig["score"],
                                "reason":        sig.get("reason", ""),
                                "target_basis":  sig.get("target_basis", ""),
                                "ai_analysis":   {},
                                "pending":       not is_immediate,
                                "current_price": actual_px,
                            })
                            # Auto-update TradingView Pine Script clipboard
                            try:
                                import subprocess
                                subprocess.Popen(
                                    ["python3", os.path.join(os.path.dirname(__file__), "gen_pinescript.py")],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                                print(f"[PINESCRIPT] Auto-updated clipboard for {sym} {interval} trade", flush=True)
                            except Exception as _e:
                                print(f"[PINESCRIPT] Could not auto-update: {_e}", flush=True)

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
        print(f"[SCAN CYCLE] complete — {scan_signals} signal(s), {scan_closes} close(s)", flush=True)

        if not _warmup_complete:
            _warmup_complete = True
            print("[WARMUP] complete — new signals will be logged from next cycle onward", flush=True)

        # Rebalance per-config capital weights every sweep
        try:
            _update_config_weights()
        except Exception:
            pass

        time.sleep(60)          # 60s between full sweeps — matches monitor cadence,
                                # ensures 5m candle closes are caught within 1 cycle


threading.Thread(target=_background_scanner, daemon=True).start()
threading.Thread(target=_price_monitor_loop, daemon=True).start()


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


@app.route("/api/admin/clear_trades", methods=["POST"])
def admin_clear_trades():
    """Wipe all trades and adaptation log — admin use only."""
    import learning
    with learning._conn() as db:
        db.execute("DELETE FROM trades")
        db.execute("DELETE FROM adaptation_log")
    return jsonify({"status": "cleared"})


@app.route("/api/admin/reset_learning", methods=["POST"])
def admin_reset_learning():
    """Reset learning weights and log back to defaults — admin use only."""
    import learning, json as _json
    with learning._conn() as db:
        db.execute("DELETE FROM adaptation_log")
        db.execute("UPDATE config SET value=? WHERE key='weights'",
                   (_json.dumps(learning.DEFAULT_WEIGHTS),))
        db.execute("UPDATE config SET value='7.0' WHERE key='signal_threshold'")
    return jsonify({"status": "learning reset", "weights": learning.DEFAULT_WEIGHTS})


@app.route("/api/admin/import_trade", methods=["POST"])
def admin_import_trade():
    """Insert or replace a single trade record — admin/sync use only."""
    import learning as _l, json as _json
    body = request.get_json() or {}
    required = ["id", "symbol", "interval", "direction", "entry", "tp", "sl",
                "score", "opened_at", "status"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        with _l._conn() as db:
            db.execute("""
                INSERT OR REPLACE INTO trades
                (id, symbol, interval, direction, entry, tp, sl, score, effective_score,
                 reason, factors_snapshot, target_basis, tp_source, opened_at, partial_tp,
                 status, closed_at, close_price, roi_pct, breakeven_activated, partial_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                body["id"], body["symbol"], body["interval"], body["direction"],
                float(body["entry"]), float(body["tp"]), float(body["sl"]),
                float(body["score"]), float(body.get("effective_score", body["score"])),
                body.get("reason", ""),
                _json.dumps(body["factors_snapshot"]) if isinstance(body.get("factors_snapshot"), dict) else body.get("factors_snapshot", "{}"),
                body.get("target_basis", ""), body.get("tp_source", "unknown"),
                body["opened_at"], body.get("partial_tp"),
                body["status"],
                body.get("closed_at"), body.get("close_price"), body.get("roi_pct"),
                int(body.get("breakeven_activated", 0)), int(body.get("partial_hit", 0)),
            ))
        return jsonify({"status": "imported", "id": body["id"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/correct_trade", methods=["POST"])
def admin_correct_trade():
    """
    Correct the outcome of a closed trade by ID.
    Body: { "id": "...", "status": "win|loss|cancelled", "close_price": 1234.56 }
    ROI is recalculated from the corrected close price.
    """
    import learning as _l
    body = request.get_json() or {}
    trade_id    = body.get("id")
    new_status  = body.get("status")
    close_price = body.get("close_price")
    if not trade_id or not new_status or close_price is None:
        return jsonify({"error": "id, status, close_price required"}), 400
    try:
        db = _l._conn()
        row = db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            db.close()
            return jsonify({"error": "trade not found"}), 404
        t = dict(row)
        direction   = t["direction"]
        entry       = float(t["entry"])
        close_price = float(close_price)
        if direction == "LONG":
            roi = (close_price - entry) / entry * 100
        else:
            roi = (entry - close_price) / entry * 100
        # Subtract round-trip fee estimate
        roi -= 0.2
        db.execute(
            "UPDATE trades SET status=?, close_price=?, roi_pct=? WHERE id=?",
            (new_status, close_price, round(roi, 4), trade_id)
        )
        db.commit()
        db.close()
        print(f"[CORRECT] {trade_id}: {t['status']} → {new_status} @ {close_price} ({roi:+.3f}%)", flush=True)
        return jsonify({"corrected": trade_id, "status": new_status,
                        "close_price": close_price, "roi_pct": round(roi, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/set_weights", methods=["POST"])
def admin_set_weights():
    """Directly set learning weights and threshold — admin/sync use only."""
    import learning as _l, json as _json
    body = request.get_json() or {}
    weights = body.get("weights")
    threshold = body.get("threshold")
    stop_multiplier = body.get("stop_multiplier")
    if not weights and threshold is None and stop_multiplier is None:
        return jsonify({"error": "provide weights, threshold, and/or stop_multiplier"}), 400
    try:
        with _l._conn() as db:
            if weights:
                # Merge with defaults so missing keys keep their default value
                merged = dict(_l.DEFAULT_WEIGHTS)
                merged.update({k: float(v) for k, v in weights.items()})
                db.execute("UPDATE config SET value=? WHERE key='weights'",
                           (_json.dumps(merged),))
            if threshold is not None:
                db.execute("UPDATE config SET value=? WHERE key='signal_threshold'",
                           (str(float(threshold)),))
            if stop_multiplier is not None:
                db.execute("UPDATE config SET value=? WHERE key='stop_multiplier'",
                           (str(float(stop_multiplier)),))
        return jsonify({"status": "weights updated",
                        "weights": _l.get_weights(),
                        "threshold": _l.get_threshold(),
                        "stop_multiplier": _l.get_stop_multiplier()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


# ── Trade Chart Page ──────────────────────────────────────────────────────────

@app.route("/chart")
def trade_charts():
    """
    Interactive candlestick chart page — one chart per trade with entry/TP/SL lines.
    Fetches real OHLC data from Binance and renders with Plotly.js.
    """
    import urllib.request
    import urllib.error

    trades = learning.get_trades()

    # ── Fetch OHLC candles for a single trade ────────────────────────────────
    def fetch_candles(symbol: str, interval: str, opened_at_str: str, closed_at_str: str | None):
        """Return list of {time, open, high, low, close} dicts, or [] on error."""
        try:
            # Convert timestamps to ms
            opened_dt  = pd.Timestamp(opened_at_str)
            opened_ms  = int(opened_dt.timestamp() * 1000)

            if closed_at_str:
                closed_dt = pd.Timestamp(closed_at_str)
                closed_ms = int(closed_dt.timestamp() * 1000)
            else:
                closed_ms = int(time.time() * 1000)

            # Candle duration in ms for padding
            interval_ms = {
                "1m": 60_000, "5m": 300_000, "15m": 900_000,
                "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
                "1d": 86_400_000, "1w": 604_800_000,
            }.get(interval, 3_600_000)

            pad        = 10 * interval_ms
            start_ms   = opened_ms - pad
            end_ms     = closed_ms + pad

            url = (
                f"https://api.binance.com/api/v3/klines"
                f"?symbol={symbol}&interval={interval}"
                f"&startTime={start_ms}&endTime={end_ms}&limit=200"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "CryptoDashboard/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read().decode())

            candles = []
            for k in raw:
                candles.append({
                    "time":  k[0],           # open time ms
                    "open":  float(k[1]),
                    "high":  float(k[2]),
                    "low":   float(k[3]),
                    "close": float(k[4]),
                })
            return candles
        except Exception:
            return []

    # ── Build per-trade chart data ────────────────────────────────────────────
    charts_data = []
    for t in trades:
        sym      = t.get("symbol", "")
        interval = t.get("interval", "1h")
        entry    = float(t.get("entry") or 0)
        tp       = float(t.get("tp")    or 0)
        sl       = float(t.get("sl")    or 0)
        direction = t.get("direction", "")
        status    = t.get("status", "open")
        roi_pct   = t.get("roi_pct")
        opened_at = t.get("opened_at", "")
        closed_at = t.get("closed_at")

        candles = fetch_candles(sym, interval, opened_at, closed_at)

        roi_str = f"{roi_pct:+.2f}%" if roi_pct is not None else "open"
        title   = (
            f"{sym} {interval} {direction} | "
            f"Entry: {entry} | TP: {tp} | SL: {sl} | {roi_str}"
        )

        charts_data.append({
            "id":        t.get("id", ""),
            "title":     title,
            "symbol":    sym,
            "interval":  interval,
            "direction": direction,
            "status":    status,
            "entry":     entry,
            "tp":        tp,
            "sl":        sl,
            "roi_pct":   roi_pct,
            "opened_at": opened_at,
            "closed_at": closed_at or "",
            "candles":   candles,
        })

    # ── Render HTML ───────────────────────────────────────────────────────────
    charts_json = json.dumps(charts_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Charts — Crypto Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 20px;
  }}
  h1 {{
    font-size: 1.4rem;
    margin-bottom: 6px;
    color: #58a6ff;
  }}
  .subtitle {{
    color: #8b949e;
    font-size: 0.85rem;
    margin-bottom: 20px;
  }}
  .filters {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 20px;
    align-items: center;
  }}
  .filters label {{ font-size: 0.85rem; color: #8b949e; }}
  .filters select, .filters input {{
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
  }}
  .btn {{
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.15s;
  }}
  .btn:hover {{ background: #30363d; }}
  .btn.active {{ background: #1f6feb; border-color: #388bfd; }}
  .summary-bar {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 20px;
    padding: 12px 16px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    font-size: 0.85rem;
  }}
  .stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .stat-label {{ color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1rem; font-weight: 600; }}
  .win  {{ color: #3fb950; }}
  .loss {{ color: #f85149; }}
  .open {{ color: #79c0ff; }}
  .chart-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    margin-bottom: 20px;
    overflow: hidden;
  }}
  .chart-header {{
    padding: 12px 16px;
    border-bottom: 1px solid #21262d;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .chart-title {{
    font-size: 0.9rem;
    font-weight: 600;
    font-family: monospace;
  }}
  .badge {{
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
  }}
  .badge-win    {{ background: #0d4a1f; color: #3fb950; border: 1px solid #238636; }}
  .badge-loss   {{ background: #4a0d0d; color: #f85149; border: 1px solid #da3633; }}
  .badge-open   {{ background: #0d2a4a; color: #79c0ff; border: 1px solid #1f6feb; }}
  .badge-cancelled {{ background: #2d2d2d; color: #8b949e; border: 1px solid #484f58; }}
  .meta {{
    padding: 8px 16px;
    font-size: 0.78rem;
    color: #8b949e;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    border-bottom: 1px solid #21262d;
  }}
  .meta span {{ display: flex; gap: 4px; align-items: center; }}
  .plotly-container {{ width: 100%; height: 480px; }}
  .no-data {{
    padding: 40px;
    text-align: center;
    color: #8b949e;
    font-size: 0.85rem;
  }}
  .hidden {{ display: none !important; }}
  .page-nav {{
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }}
  .page-info {{ color: #8b949e; font-size: 0.85rem; margin-left: 8px; }}
</style>
</head>
<body>

<h1>📈 Trade Charts</h1>
<p class="subtitle">Candlestick charts with entry, TP &amp; SL levels for every trade</p>

<div class="filters">
  <label>Filter:
    <select id="statusFilter" onchange="applyFilters()">
      <option value="all">All trades</option>
      <option value="open">Open</option>
      <option value="win">Wins</option>
      <option value="loss">Losses</option>
      <option value="cancelled">Cancelled</option>
    </select>
  </label>
  <label>Symbol:
    <select id="symbolFilter" onchange="applyFilters()">
      <option value="all">All symbols</option>
    </select>
  </label>
  <label>Interval:
    <select id="intervalFilter" onchange="applyFilters()">
      <option value="all">All intervals</option>
    </select>
  </label>
  <button class="btn" onclick="applyFilters()">Apply</button>
  <button class="btn" onclick="resetFilters()">Reset</button>
</div>

<div class="summary-bar" id="summaryBar"></div>

<div class="page-nav" id="pageNav"></div>

<div id="chartsContainer"></div>

<script>
const ALL_TRADES = {charts_json};

const PAGE_SIZE = 10;
let currentPage = 1;
let filteredTrades = [];

function fmt(n) {{
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString(undefined, {{maximumFractionDigits: 8, minimumFractionDigits: 0}});
}}

function fmtDate(s) {{
  if (!s) return '—';
  try {{ return new Date(s).toLocaleString(); }} catch(e) {{ return s; }}
}}

function badgeClass(status) {{
  return 'badge badge-' + (status || 'open');
}}

function buildSummary(trades) {{
  const wins = trades.filter(t => t.status === 'win');
  const losses = trades.filter(t => t.status === 'loss');
  const open = trades.filter(t => t.status === 'open');
  const closed = wins.concat(losses);
  const winRate = closed.length ? (wins.length / closed.length * 100).toFixed(1) : '—';
  const totalRoi = closed.reduce((s, t) => s + (t.roi_pct || 0), 0).toFixed(2);
  const avgRoi = closed.length ? (closed.reduce((s, t) => s + (t.roi_pct || 0), 0) / closed.length).toFixed(2) : '—';

  return `
    <div class="stat"><span class="stat-label">Total</span><span class="stat-value">${{trades.length}}</span></div>
    <div class="stat"><span class="stat-label">Open</span><span class="stat-value open">${{open.length}}</span></div>
    <div class="stat"><span class="stat-label">Wins</span><span class="stat-value win">${{wins.length}}</span></div>
    <div class="stat"><span class="stat-label">Losses</span><span class="stat-value loss">${{losses.length}}</span></div>
    <div class="stat"><span class="stat-label">Win Rate</span><span class="stat-value">${{winRate}}%</span></div>
    <div class="stat"><span class="stat-label">Total ROI</span><span class="stat-value ${{parseFloat(totalRoi) >= 0 ? 'win' : 'loss'}}">${{totalRoi}}%</span></div>
    <div class="stat"><span class="stat-label">Avg ROI</span><span class="stat-value ${{parseFloat(avgRoi) >= 0 ? 'win' : 'loss'}}">${{avgRoi !== '—' ? avgRoi + '%' : '—'}}</span></div>
  `;
}}

function buildChart(trade, divId) {{
  const candles = trade.candles || [];
  if (!candles.length) return false;

  const dates = candles.map(c => new Date(c.time).toISOString());
  const open  = candles.map(c => c.open);
  const high  = candles.map(c => c.high);
  const low   = candles.map(c => c.low);
  const close = candles.map(c => c.close);

  const tFirst = dates[0];
  const tLast  = dates[dates.length - 1];

  const candleTrace = {{
    type: 'candlestick',
    x: dates, open, high, low, close,
    name: trade.symbol,
    increasing: {{ line: {{ color: '#3fb950' }}, fillcolor: '#238636' }},
    decreasing: {{ line: {{ color: '#f85149' }}, fillcolor: '#da3633' }},
  }};

  // Horizontal level lines
  const lineShapes = [
    {{ price: trade.entry, color: '#79c0ff', label: `Entry ${{fmt(trade.entry)}}`,  dash: 'dot' }},
    {{ price: trade.tp,    color: '#3fb950', label: `TP ${{fmt(trade.tp)}}`,         dash: 'dash' }},
    {{ price: trade.sl,    color: '#f85149', label: `SL ${{fmt(trade.sl)}}`,         dash: 'dash' }},
  ].filter(l => l.price > 0);

  const shapes = lineShapes.map(l => ({{
    type: 'line',
    x0: tFirst, x1: tLast,
    y0: l.price, y1: l.price,
    line: {{ color: l.color, width: 1.5, dash: l.dash }},
  }}));

  const annotations = lineShapes.map(l => ({{
    x: tLast,
    y: l.price,
    xref: 'x', yref: 'y',
    text: l.label,
    showarrow: false,
    xanchor: 'right',
    font: {{ color: l.color, size: 11 }},
    bgcolor: 'rgba(13,17,23,0.75)',
    bordercolor: l.color,
    borderwidth: 1,
    borderpad: 3,
  }}));

  const layout = {{
    paper_bgcolor: '#161b22',
    plot_bgcolor:  '#0d1117',
    font: {{ color: '#e6edf3', size: 11 }},
    margin: {{ l: 60, r: 120, t: 20, b: 40 }},
    xaxis: {{
      type: 'date',
      rangeslider: {{ visible: false }},
      gridcolor: '#21262d',
      linecolor: '#30363d',
    }},
    yaxis: {{
      gridcolor: '#21262d',
      linecolor: '#30363d',
      tickformat: '.4~f',
    }},
    shapes,
    annotations,
    hovermode: 'x unified',
    showlegend: false,
  }};

  Plotly.newPlot(divId, [candleTrace], layout, {{
    responsive: true,
    displayModeBar: false,
  }});
  return true;
}}

function renderCharts(trades) {{
  const container = document.getElementById('chartsContainer');
  container.innerHTML = '';
  if (!trades.length) {{
    container.innerHTML = '<div class="no-data">No trades match the current filter.</div>';
    return;
  }}

  // Pagination
  const totalPages = Math.ceil(trades.length / PAGE_SIZE);
  if (currentPage > totalPages) currentPage = 1;
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageTrades = trades.slice(start, start + PAGE_SIZE);

  renderPageNav(trades.length, totalPages);

  pageTrades.forEach((trade, idx) => {{
    const divId = `chart_${{start + idx}}`;
    const roiColor = trade.roi_pct === null ? '' : (trade.roi_pct >= 0 ? 'win' : 'loss');
    const roiStr = trade.roi_pct !== null ? `<span class="${{roiColor}}">${{(trade.roi_pct >= 0 ? '+' : '') + trade.roi_pct.toFixed(2)}}%</span>` : '';

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
      <div class="chart-header">
        <span class="chart-title">${{trade.title}}</span>
        <span class="${{badgeClass(trade.status)}}">${{trade.status}}</span>
      </div>
      <div class="meta">
        <span>📅 Opened: ${{fmtDate(trade.opened_at)}}</span>
        ${{trade.closed_at ? `<span>🏁 Closed: ${{fmtDate(trade.closed_at)}}</span>` : ''}}
        ${{trade.roi_pct !== null ? `<span>💰 ROI: ${{roiStr}}</span>` : ''}}
        <span>🔷 Entry: ${{fmt(trade.entry)}}</span>
        <span>✅ TP: <span class="win">${{fmt(trade.tp)}}</span></span>
        <span>🛑 SL: <span class="loss">${{fmt(trade.sl)}}</span></span>
      </div>
      ${{trade.candles && trade.candles.length
        ? `<div class="plotly-container" id="${{divId}}"></div>`
        : `<div class="no-data">⚠ No candle data available for this trade</div>`
      }}
    `;
    container.appendChild(card);

    if (trade.candles && trade.candles.length) {{
      setTimeout(() => buildChart(trade, divId), 0);
    }}
  }});
}}

function renderPageNav(total, totalPages) {{
  const nav = document.getElementById('pageNav');
  nav.innerHTML = '';
  if (totalPages <= 1) return;

  const info = document.createElement('span');
  info.className = 'page-info';
  info.textContent = `Showing ${{Math.min((currentPage-1)*PAGE_SIZE+1, total)}}–${{Math.min(currentPage*PAGE_SIZE, total)}} of ${{total}} trades`;
  nav.appendChild(info);

  const prev = document.createElement('button');
  prev.className = 'btn' + (currentPage === 1 ? ' disabled' : '');
  prev.textContent = '← Prev';
  prev.onclick = () => {{ if (currentPage > 1) {{ currentPage--; renderCharts(filteredTrades); window.scrollTo(0,0); }} }};
  nav.appendChild(prev);

  for (let p = 1; p <= totalPages; p++) {{
    const btn = document.createElement('button');
    btn.className = 'btn' + (p === currentPage ? ' active' : '');
    btn.textContent = p;
    btn.onclick = ((_p) => () => {{ currentPage = _p; renderCharts(filteredTrades); window.scrollTo(0,0); }})(p);
    nav.appendChild(btn);
  }}

  const next = document.createElement('button');
  next.className = 'btn' + (currentPage === totalPages ? ' disabled' : '');
  next.textContent = 'Next →';
  next.onclick = () => {{ if (currentPage < totalPages) {{ currentPage++; renderCharts(filteredTrades); window.scrollTo(0,0); }} }};
  nav.appendChild(next);
}}

function populateFilters() {{
  const symbols   = [...new Set(ALL_TRADES.map(t => t.symbol))].sort();
  const intervals = [...new Set(ALL_TRADES.map(t => t.interval))].sort();
  const symSel = document.getElementById('symbolFilter');
  const ivSel  = document.getElementById('intervalFilter');
  symbols.forEach(s => {{ const o = document.createElement('option'); o.value = s; o.textContent = s; symSel.appendChild(o); }});
  intervals.forEach(i => {{ const o = document.createElement('option'); o.value = i; o.textContent = i; ivSel.appendChild(o); }});
}}

function applyFilters() {{
  const st  = document.getElementById('statusFilter').value;
  const sym = document.getElementById('symbolFilter').value;
  const iv  = document.getElementById('intervalFilter').value;
  filteredTrades = ALL_TRADES.filter(t =>
    (st  === 'all' || t.status   === st)  &&
    (sym === 'all' || t.symbol   === sym) &&
    (iv  === 'all' || t.interval === iv)
  );
  currentPage = 1;
  document.getElementById('summaryBar').innerHTML = buildSummary(filteredTrades);
  renderCharts(filteredTrades);
}}

function resetFilters() {{
  document.getElementById('statusFilter').value   = 'all';
  document.getElementById('symbolFilter').value   = 'all';
  document.getElementById('intervalFilter').value = 'all';
  applyFilters();
}}

// Init
populateFilters();
applyFilters();
</script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  Crypto Analysis Dashboard")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)

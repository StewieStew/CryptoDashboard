"""
Backtester — replays the signal strategy over historical Binance OHLCV data
to find optimal parameter combinations.

Uses existing indicator functions from analysis.py (no live API calls during replay).
"""
from __future__ import annotations

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from itertools import product

import analysis
from learning import DEFAULT_WEIGHTS, DEFAULT_STOP_MULT

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"

# 0.10% per side (entry + exit) = 0.20% round-trip; expressed in R-multiples
# commission_r = COMMISSION_RT_PCT * entry_price / risk (varies per trade)
COMMISSION_RT_PCT = 0.002

# Need enough bars for EMA200, RSI(14), ADX(14) to be meaningful
_MIN_LOOKBACK = 220

# Default parameter grid used by grid_search() and weekly review validation
DEFAULT_PARAM_GRID = {
    "adx_threshold":   [20, 25, 30],
    "body_ratio_min":  [0.25, 0.30, 0.35],
    "score_threshold": [6.5, 7.0, 7.5, 8.0],
    "min_rr":          [2.5, 3.0, 3.5],
}

# Campaign: symbols + timeframes to sweep in a full research run
CAMPAIGN_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]

# Three day-trading + short-swing timeframes:
#   15m — intraday, trades close same day to next day
#   1h  — day trades, close within 1-3 days
#   4h  — short swing, close within 3-7 days (within a week)
CAMPAIGN_INTERVALS = ["15m", "1h", "4h"]

# Max history years per timeframe — balances coverage vs replay speed
# 15m: 2 yr ≈ 70k bars; 4h: 5 yr ≈ 10k bars; 1d: 10 yr ≈ 3,650 bars
MAX_HISTORY_YEARS = {
    "15m": 2,
    "30m": 3,
    "1h":  4,
    "4h":  5,
    "1d":  10,
    "1w":  10,
}

# History for discovery campaigns
# Sliding window makes each bar O(1) so we can afford more history without blowup.
CAMPAIGN_MAX_HISTORY_YEARS = {
    "15m": 2,   # ~70k bars — O(n) now, expect 60-150 trades/coin
    "1h":  3,   # ~26k bars
    "4h":  5,   # ~11k bars — full practical history
}

# Bar-skip step used during campaign grid search.
# step=2: checks every 2nd bar — 2× faster than step=1 with near-identical signal count.
# Use step=1 for final validation of the winning config.
CAMPAIGN_STEP = 2

# Campaign grid — 3×3×3 = 27 combinations per job (speed-optimised)
# body_ratio_min is held at default (0.30) to reduce runtime
CAMPAIGN_GRID = {
    "adx_threshold":   [20, 25, 30],
    "score_threshold": [6.5, 7.0, 7.5],
    "min_rr":          [2.5, 3.0, 3.5],
}

# Baseline parameters (matches current live defaults)
DEFAULT_PARAMS = {
    "adx_threshold":   25,
    "body_ratio_min":  0.15,   # relaxed from 0.30 — more candles qualify, more trades
    "score_threshold": 7.0,
    "min_rr":          3.0,
    "rsi_long_min":    35.0,   # widened from 40 — catch more pullback entries
    "rsi_long_max":    70.0,   # widened from 65
    "rsi_short_min":   30.0,   # widened from 35
    "rsi_short_max":   65.0,   # widened from 60
    "level_touch_min": 1,      # relaxed from 2 — any touched level qualifies
}

# ms duration per interval — used for pagination
_MS_PER_BAR = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4  * 60 * 60 * 1000,
    "1d":  24 * 60 * 60 * 1000,
    "1w":  7  * 24 * 60 * 60 * 1000,
}


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_historical_ohlcv(symbol: str, interval: str, years: int = 2) -> pd.DataFrame:
    """
    Fetch N years of OHLCV data from Binance, paginating in 1000-bar batches.
    Returns DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume
    """
    end_ms      = int(datetime.now(timezone.utc).timestamp() * 1000)
    ms_per_bar  = _MS_PER_BAR.get(interval, 4 * 60 * 60 * 1000)
    total_bars  = int(years * 365.25 * 24 * 3600 * 1000 / ms_per_bar)
    start_ms    = end_ms - total_bars * ms_per_bar

    all_rows      = []
    current_start = start_ms
    batch_size    = 1000

    while current_start < end_ms:
        try:
            r = requests.get(
                BINANCE_KLINES,
                params={
                    "symbol":    symbol,
                    "interval":  interval,
                    "startTime": current_start,
                    "limit":     batch_size,
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_rows.extend(data)
            current_start = int(data[-1][0]) + ms_per_bar
            time.sleep(0.25)   # respect Binance public API rate limit
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()

    cols = ["ts", "open", "high", "low", "close", "volume",
            "ct", "qv", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


# ─────────────────────────────────────────────
# SIGNAL REPLAY
# ─────────────────────────────────────────────

def _build_adx_data(df: pd.DataFrame, adx_threshold: float = 25) -> dict:
    """Compute ADX data dict with a configurable trend threshold."""
    adx_s, di_plus_s, di_minus_s = analysis.adx_indicator(df)
    adx_cur = float(adx_s.iloc[-1]) if not adx_s.empty else 0.0
    return {
        "value":    round(adx_cur, 1),
        "di_plus":  round(float(di_plus_s.iloc[-1]),  1),
        "di_minus": round(float(di_minus_s.iloc[-1]), 1),
        "trending": adx_cur > adx_threshold,
        "strong":   adx_cur > 50,
    }


def _replay_signal(df_slice: pd.DataFrame, interval: str, params: dict,
                   weights: dict | None = None) -> dict | None:
    """
    Compute all indicators for a historical DataFrame slice and apply the
    signal strategy with configurable parameters. No live API calls.

    weights — optional custom signal weights (overrides DEFAULT_WEIGHTS).
              Set a factor's weight to 0 to exclude it from scoring.
    Returns a signal dict or None.
    """
    if len(df_slice) < _MIN_LOOKBACK:
        return None

    try:
        # Regime: use lower half of the slice as a pseudo-HTF (no separate fetch)
        regime_df = df_slice.iloc[: len(df_slice) // 2]
        if len(regime_df) < 50:
            regime_df = df_slice
        regime = analysis.market_regime(regime_df)

        # Structure
        n  = analysis._swing_n(interval)
        sh, sl = analysis.detect_swings(df_slice, n=n)
        if not sh or not sl:
            return None

        structure = analysis.detect_structure(df_slice, sh, sl)
        sweeps    = analysis.detect_sweeps(df_slice, sh, sl)

        # Precision indicators: FVG, FIB, equal levels
        fvgs    = analysis.detect_fvg(df_slice)
        fib_d   = analysis.fib_analysis(df_slice, sh, sl)
        eq_lvls = analysis.detect_equal_levels(sh, sl)

        # Indicators
        vol      = analysis.volume_analysis(df_slice)
        rsi_data = analysis.rsi_analysis(df_slice)
        adx_data = _build_adx_data(df_slice, params.get("adx_threshold", 25))

        # Confluence — use custom weights if provided, else default
        w = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        confluence = analysis.confluence_score(
            regime, structure, vol, rsi_data, sweeps,
            interval, w, adx_data=adx_data,
            fvg_data=fvgs, fib_data=fib_d, eq_levels=eq_lvls,
        )

        # Risk context (FVG used for targets, FIB for precision entry)
        risk = analysis.risk_context(
            df_slice, structure, sh, sl, interval,
            stop_multiplier=DEFAULT_STOP_MULT, fvgs=fvgs, fib_data=fib_d,
        )

        # Signal with configurable gate params
        signal = analysis.generate_signal(
            confluence, structure, risk, df_slice,
            signal_threshold=params.get("score_threshold", 7.0),
            interval=interval,
            body_ratio_min=params.get("body_ratio_min", 0.30),
            min_rr=params.get("min_rr", 3.0),
            rsi_long_range=(
                params.get("rsi_long_min", 40.0),
                params.get("rsi_long_max", 65.0),
            ),
            rsi_short_range=(
                params.get("rsi_short_min", 35.0),
                params.get("rsi_short_max", 60.0),
            ),
            level_touch_min=params.get("level_touch_min", 2),
        )
        return signal

    except Exception:
        return None


# ─────────────────────────────────────────────
# TRADE SIMULATION
# ─────────────────────────────────────────────

def _simulate_trade(df_future: pd.DataFrame, signal: dict,
                    max_bars: int = 100) -> dict:
    """
    Simulate a trade by scanning future bars for TP/SL/timeout.

    Handles pending-entry logic: trade activates when price reaches entry level.
    Returns {"outcome": "win"|"loss"|"timeout", "bars_held": int, "actual_rr": float}
    """
    entry     = float(signal["entry"])
    tp        = float(signal["target"])
    sl        = float(signal["stop"])
    direction = signal["direction"]
    risk_d    = abs(entry - sl)
    comm_r    = (COMMISSION_RT_PCT * entry / risk_d) if risk_d > 0 else 0.0

    activated = False

    for i, bar in enumerate(df_future.itertuples(), 1):
        if i > max_bars:
            break

        high  = float(bar.high)
        low   = float(bar.low)

        # Activate pending trade when price retraces to entry
        if not activated:
            if direction == "LONG"  and low  <= entry:
                activated = True
            elif direction == "SHORT" and high >= entry:
                activated = True
            continue   # don't check TP/SL until activated

        if direction == "LONG":
            if high >= tp:
                actual_rr = (tp - entry) / risk_d if risk_d > 0 else 0
                return {"outcome": "win",  "bars_held": i, "actual_rr": round(actual_rr - comm_r, 2)}
            if low  <= sl:
                return {"outcome": "loss", "bars_held": i, "actual_rr": round(-1.0 - comm_r, 2)}
        else:  # SHORT
            if low  <= tp:
                actual_rr = (entry - tp) / risk_d if risk_d > 0 else 0
                return {"outcome": "win",  "bars_held": i, "actual_rr": round(actual_rr - comm_r, 2)}
            if high >= sl:
                return {"outcome": "loss", "bars_held": i, "actual_rr": round(-1.0 - comm_r, 2)}

    return {"outcome": "timeout", "bars_held": max_bars, "actual_rr": 0.0}


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(symbol: str, interval: str, params: dict,
                 df: pd.DataFrame | None = None,
                 years: int = 2,
                 step: int = 1,
                 weights: dict | None = None) -> dict:
    """
    Replay the signal strategy over historical data with given parameters.

    Returns:
        {
            symbol, interval, total_trades, wins, losses, timeouts,
            win_rate (0-1), avg_rr, max_drawdown, profit_factor, params
        }
    """
    if df is None:
        df = fetch_historical_ohlcv(symbol, interval, years)

    if df.empty or len(df) < _MIN_LOOKBACK + 10:
        return {"error": "Not enough historical data", "params": params,
                "symbol": symbol, "interval": interval}

    trades = []
    equity = [0.0]   # running equity curve: +actual_rr on win, -1 on loss

    # Sliding window: always operate on the most recent _LOOKBACK_WINDOW bars.
    # All indicators (EMA-200, RSI, ATR, OBV) converge within 400 bars, so a
    # 600-bar window gives identical results while keeping each call O(1) instead
    # of O(n) — turns the entire backtest from O(n²) → O(n), ~10-20× speedup.
    _LOOKBACK_WINDOW = 600

    i = _MIN_LOOKBACK
    while i < len(df) - 1:
        df_slice = df.iloc[max(0, i - _LOOKBACK_WINDOW): i + 1]
        signal   = _replay_signal(df_slice, interval, params, weights=weights)

        if signal:
            df_future = df.iloc[i + 1:]
            result    = _simulate_trade(df_future, signal, max_bars=100)
            outcome   = result["outcome"]

            if outcome == "win":
                equity.append(equity[-1] + result["actual_rr"])
                trades.append({"outcome": "win",     "rr": result["actual_rr"]})
            elif outcome == "loss":
                equity.append(equity[-1] - 1.0)
                trades.append({"outcome": "loss",    "rr": -1.0})
            else:
                trades.append({"outcome": "timeout", "rr": 0.0})

            # Skip ahead by bars the trade was held (avoid overlapping signals)
            i += max(result.get("bars_held", 5), 1)
        else:
            i += step

    # Metrics
    completed = [t for t in trades if t["outcome"] in ("win", "loss")]
    wins      = [t for t in completed if t["outcome"] == "win"]
    losses    = [t for t in completed if t["outcome"] == "loss"]

    win_rate      = len(wins) / len(completed) if completed else 0.0
    avg_rr        = sum(t["rr"] for t in wins) / len(wins) if wins else 0.0
    total_win_r   = sum(t["rr"] for t in wins)
    total_loss_r  = float(len(losses))
    profit_factor = (total_win_r / total_loss_r) if total_loss_r > 0 else (total_win_r or 0.0)

    # Max drawdown (peak-to-trough on equity curve)
    peak   = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    return {
        "symbol":        symbol,
        "interval":      interval,
        "total_trades":  len(completed),
        "wins":          len(wins),
        "losses":        len(losses),
        "timeouts":      len([t for t in trades if t["outcome"] == "timeout"]),
        "win_rate":      round(win_rate, 3),
        "avg_rr":        round(avg_rr, 2),
        "max_drawdown":  round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        "params":        params,
    }


# ─────────────────────────────────────────────
# MACD(5/13)+EMA50+VOL — FAST WALK-FORWARD REPLAY
# Walk-forward validated: OOS PF 1.30, OOS WR 30.2%, CI [27-33%]
# ─────────────────────────────────────────────

def _macd_precompute(df: pd.DataFrame) -> dict:
    """Precompute indicator series for the MACD strategy (O(n) replay)."""
    close = df["close"]
    m5, m5s, _ = analysis.macd(close, 5, 13, 8)
    return {
        "close":  close,
        "macd5":  m5,  "macd5_sig": m5s,
        "ema50":  analysis.ema(close, 50),
        "atr14":  analysis.atr(df, 14),
        "vol20":  df["volume"].rolling(20).mean(),
        "volume": df["volume"],
    }


def _macd_signal(pre: dict, i: int) -> dict | None:
    """MACD(5/13/8) cross + EMA50 trend + volume >1.2× at bar i."""
    if i < 1:
        return None
    m5, m5s = pre["macd5"], pre["macd5_sig"]
    prev_d = float(m5.iloc[i - 1]) - float(m5s.iloc[i - 1])
    curr_d = float(m5.iloc[i])     - float(m5s.iloc[i])
    if prev_d < 0 and curr_d >= 0:
        direction = "LONG"
    elif prev_d > 0 and curr_d <= 0:
        direction = "SHORT"
    else:
        return None
    cc  = float(pre["close"].iloc[i])
    e50 = float(pre["ema50"].iloc[i])
    if np.isnan(e50): return None
    if direction == "LONG"  and cc < e50: return None
    if direction == "SHORT" and cc > e50: return None
    avg_vol = float(pre["vol20"].iloc[i])
    cur_vol = float(pre["volume"].iloc[i])
    if not np.isnan(avg_vol) and avg_vol > 0 and cur_vol < 1.2 * avg_vol:
        return None
    atr_val = float(pre["atr14"].iloc[i])
    if atr_val <= 0 or np.isnan(atr_val): return None
    stop   = cc - atr_val if direction == "LONG" else cc + atr_val
    target = cc + 3.0 * atr_val if direction == "LONG" else cc - 3.0 * atr_val
    return {"direction": direction, "entry": cc, "stop": stop, "target": target}


def _macd_replay(df: pd.DataFrame, pre: dict,
                 min_lookback: int = 55, max_bars: int = 100) -> dict:
    """O(n) replay — immediate entry at signal bar close, no pending activation."""
    trades = []
    equity = [0.0]
    i      = min_lookback

    while i < len(df) - 1:
        try:
            sig = _macd_signal(pre, i)
        except Exception:
            sig = None

        if sig:
            entry     = float(sig["entry"])
            tp        = float(sig["target"])
            sl        = float(sig["stop"])
            direction = sig["direction"]
            risk      = abs(entry - sl)
            comm_r    = (COMMISSION_RT_PCT * entry / risk) if risk > 0 else 0.0
            resolved  = False

            for j, bar in enumerate(df.iloc[i + 1:].itertuples(), 1):
                if j > max_bars:
                    trades.append({"outcome": "timeout", "rr": 0.0})
                    i += 1; resolved = True; break
                hi, lo = float(bar.high), float(bar.low)
                if direction == "LONG":
                    if hi >= tp:
                        rr = round((tp - entry) / risk - comm_r, 2) if risk else 0
                        equity.append(equity[-1] + rr)
                        trades.append({"outcome": "win", "rr": rr})
                        i += j; resolved = True; break
                    if lo <= sl:
                        rr = round(-1.0 - comm_r, 2)
                        equity.append(equity[-1] + rr)
                        trades.append({"outcome": "loss", "rr": rr})
                        i += j; resolved = True; break
                else:
                    if lo <= tp:
                        rr = round((entry - tp) / risk - comm_r, 2) if risk else 0
                        equity.append(equity[-1] + rr)
                        trades.append({"outcome": "win", "rr": rr})
                        i += j; resolved = True; break
                    if hi >= sl:
                        rr = round(-1.0 - comm_r, 2)
                        equity.append(equity[-1] + rr)
                        trades.append({"outcome": "loss", "rr": rr})
                        i += j; resolved = True; break
            if not resolved:
                i += 1
        else:
            i += 1

    completed = [t for t in trades if t["outcome"] in ("win", "loss")]
    wins      = [t for t in completed if t["outcome"] == "win"]
    losses    = [t for t in completed if t["outcome"] == "loss"]

    if not completed:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_rr": 0.0, "max_drawdown": 0.0}

    win_rate = len(wins) / len(completed)
    total_w  = sum(t["rr"] for t in wins)
    total_l  = abs(sum(t["rr"] for t in losses))
    pf       = (total_w / total_l) if total_l > 0 else float(total_w)
    avg_rr   = total_w / len(wins) if wins else 0.0

    peak = mx_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = peak - v
        if dd > mx_dd: mx_dd = dd

    return {
        "total":         len(completed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(win_rate, 3),
        "profit_factor": round(pf, 2),
        "avg_rr":        round(avg_rr, 2),
        "max_drawdown":  round(mx_dd, 2),
    }


def _ci_normal(wr: float, n: int, z: float = 1.96) -> tuple:
    """95% normal-approximation confidence interval for a proportion."""
    if n < 5:
        return (0.0, 1.0)
    se = (wr * (1.0 - wr) / n) ** 0.5
    return (max(0.0, wr - z * se), min(1.0, wr + z * se))


def run_macd_backtest(symbol: str, interval: str, years: int = 3) -> dict:
    """
    Replay MACD(5/13)+EMA50+Vol with walk-forward split.

    Walk-forward: first 2/3 of history = in-sample (train),
                  last  1/3           = out-of-sample (test, never seen).

    Returns full-period + IS + OOS metrics with 95% CI on OOS win rate.
    """
    df = fetch_historical_ohlcv(symbol, interval, years)
    if df.empty or len(df) < 60:
        return {"error": "Not enough historical data", "symbol": symbol, "interval": interval}

    split  = int(len(df) * (2 / 3))
    df_is  = df.iloc[:split]
    df_oos = df.iloc[split:]

    def _run(data: pd.DataFrame) -> dict:
        return _macd_replay(data, _macd_precompute(data))

    full_m = _run(df)
    is_m   = _run(df_is)
    oos_m  = _run(df_oos)

    ci_lo, ci_hi = _ci_normal(oos_m["win_rate"], oos_m["total"])
    oos_holds = oos_m["profit_factor"] >= 1.05 and oos_m["win_rate"] >= is_m["win_rate"] - 0.05

    return {
        "symbol":    symbol,
        "interval":  interval,
        "years":     years,
        "strategy":  "MACD(5/13)+EMA50+Vol  ·  3:1 R:R",
        "bars":      len(df),
        "is_bars":   len(df_is),
        "oos_bars":  len(df_oos),
        **full_m,
        "is":        is_m,
        "oos":       oos_m,
        "oos_ci_lo": round(ci_lo, 3),
        "oos_ci_hi": round(ci_hi, 3),
        "oos_holds": oos_holds,
    }


def grid_search(symbol: str, interval: str,
                param_grid: dict | None = None,
                years: int = 2,
                step: int = 1,
                weights: dict | None = None) -> list:
    """
    Run backtest for every combination in param_grid.
    Fetches historical data once and reuses it for all combinations.

    Returns list of result dicts sorted by win_rate DESC (then profit_factor).

    param_grid example:
        {
            "adx_threshold":   [20, 25, 30],
            "body_ratio_min":  [0.25, 0.30, 0.35],
            "score_threshold": [6.5, 7.0, 7.5, 8.0],
            "min_rr":          [2.5, 3.0, 3.5],
        }
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    df = fetch_historical_ohlcv(symbol, interval, years)
    if df.empty:
        return [{"error": f"Failed to fetch data for {symbol} {interval}"}]

    keys         = list(param_grid.keys())
    values       = list(param_grid.values())
    combinations = list(product(*values))

    results = []
    for combo in combinations:
        params = dict(DEFAULT_PARAMS)          # start from defaults
        params.update(dict(zip(keys, combo)))  # apply grid values
        result = run_backtest(symbol, interval, params, df=df, years=years, step=step, weights=weights)
        results.append(result)

    results.sort(
        key=lambda r: (r.get("win_rate", 0), r.get("profit_factor", 0)),
        reverse=True,
    )
    return results

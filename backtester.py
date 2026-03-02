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
CAMPAIGN_INTERVALS = ["15m", "4h"]

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
    "body_ratio_min":  0.30,
    "score_threshold": 7.0,
    "min_rr":          3.0,
    "rsi_long_min":    40.0,
    "rsi_long_max":    65.0,
    "rsi_short_min":   35.0,
    "rsi_short_max":   60.0,
    "level_touch_min": 2,
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


def _replay_signal(df_slice: pd.DataFrame, interval: str, params: dict) -> dict | None:
    """
    Compute all indicators for a historical DataFrame slice and apply the
    signal strategy with configurable parameters. No live API calls.

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

        # Indicators
        vol      = analysis.volume_analysis(df_slice)
        rsi_data = analysis.rsi_analysis(df_slice)
        adx_data = _build_adx_data(df_slice, params.get("adx_threshold", 25))

        # Confluence (default weights — clean baseline)
        confluence = analysis.confluence_score(
            regime, structure, vol, rsi_data, sweeps,
            interval, dict(DEFAULT_WEIGHTS), adx_data=adx_data,
        )

        # Risk context
        risk = analysis.risk_context(
            df_slice, structure, sh, sl, interval,
            stop_multiplier=DEFAULT_STOP_MULT, fvgs=[],
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
                return {"outcome": "win",  "bars_held": i, "actual_rr": round(actual_rr, 2)}
            if low  <= sl:
                return {"outcome": "loss", "bars_held": i, "actual_rr": -1.0}
        else:  # SHORT
            if low  <= tp:
                actual_rr = (entry - tp) / risk_d if risk_d > 0 else 0
                return {"outcome": "win",  "bars_held": i, "actual_rr": round(actual_rr, 2)}
            if high >= sl:
                return {"outcome": "loss", "bars_held": i, "actual_rr": -1.0}

    return {"outcome": "timeout", "bars_held": max_bars, "actual_rr": 0.0}


# ─────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(symbol: str, interval: str, params: dict,
                 df: pd.DataFrame | None = None,
                 years: int = 2) -> dict:
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

    i = _MIN_LOOKBACK
    while i < len(df) - 1:
        df_slice = df.iloc[: i + 1]
        signal   = _replay_signal(df_slice, interval, params)

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
            i += 1

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


def grid_search(symbol: str, interval: str,
                param_grid: dict | None = None,
                years: int = 2) -> list:
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
        result = run_backtest(symbol, interval, params, df=df, years=years)
        results.append(result)

    results.sort(
        key=lambda r: (r.get("win_rate", 0), r.get("profit_factor", 0)),
        reverse=True,
    )
    return results

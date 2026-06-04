"""
Claude AI signal evaluation — uses Haiku for low latency on every signal,
and Sonnet for periodic deeper performance reviews.

Set ANTHROPIC_API_KEY in your Render environment variables.
If unset, all functions return empty dicts gracefully.
"""
from __future__ import annotations

import os
import json
import anthropic

def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def _tier(interval: str) -> str:
    return "Day" if interval in ("15m", "30m", "1h") else "Swing"


def _fmt(v) -> str:
    if v is None:
        return "—"
    n = float(v)
    if n >= 10000:  return f"{n:,.2f}"
    if n >= 100:    return f"{n:.3f}"
    if n >= 1:      return f"{n:.4f}"
    if n >= 0.01:   return f"{n:.5f}"
    return f"{n:.6f}"


# ── Signal evaluation ─────────────────────────────────────────────────────────

def analyze_signal(signal: dict, trade_history: list,
                   market_context: dict,
                   ai_accuracy: dict | None = None,
                   ohlcv_candles: list | None = None,
                   htf_context: dict | None = None) -> dict:
    """
    Ask Claude Haiku to evaluate a trade signal before it's logged.

    Now receives:
      - ohlcv_candles: last 20 candles [{time, open, high, low, close, volume}]
      - htf_context: higher-timeframe analysis dict (regime, structure, adx, rsi)

    Returns:
        {confidence, recommendation, reasoning, risks, positives,
         entry_assessment, bos_quality}
        or {} if API key not set or call fails.

    `recommendation` is one of: "strong_take" | "take" | "skip"
    `bos_quality`   is one of: "genuine" | "suspect" | "false_break"
    """
    client = _get_client()
    if not client:
        return {}

    sym          = signal.get("symbol", "")
    direction    = signal.get("direction", "")
    interval     = signal.get("interval", "")
    entry        = float(signal.get("entry") or 0)
    current_px   = float(signal.get("current_price") or entry)
    tp           = float(signal.get("tp") or signal.get("target") or 0)
    sl           = float(signal.get("sl") or signal.get("stop") or 0)
    score        = signal.get("score", 0)
    reason       = signal.get("reason", "")
    factors      = signal.get("factors_snapshot", {})
    adx          = signal.get("adx_value", 0)
    vwap_side    = signal.get("vwap_side", "")

    tp_pct  = abs((tp - entry) / entry * 100) if entry else 0
    sl_pct  = abs((sl - entry) / entry * 100) if entry else 0
    rr      = round(tp_pct / sl_pct, 2) if sl_pct else 0
    gap_pct = abs((current_px - entry) / entry * 100) if entry else 0

    # Recent symbol performance
    sym_closed = [t for t in trade_history
                  if t.get("symbol") == sym and t.get("status") in ("win", "loss")][-10:]
    sym_wins   = sum(1 for t in sym_closed if t.get("status") == "win")

    # Overall system
    all_closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
    total      = len(all_closed)
    win_rate   = (sum(1 for t in all_closed if t.get("status") == "win") / total * 100) if total else 0
    avg_roi    = (sum(t.get("roi_pct") or 0 for t in all_closed) / total) if total else 0

    # Market context lines
    fg      = market_context.get("fear_greed", {})
    btc_dom = market_context.get("btc_dominance", 0)
    funding = market_context.get("funding_rate", 0)
    oi      = market_context.get("open_interest", 0)

    ctx_parts = []
    if fg:
        ctx_parts.append(f"Fear & Greed: {fg.get('value','?')} — {fg.get('label','?')}")
    if btc_dom:
        ctx_parts.append(f"BTC Dominance: {btc_dom:.1f}%")
    if funding:
        fn = ("longs paying shorts (overheated)" if funding > 0.05 else
              "shorts paying longs (squeeze risk)" if funding < -0.05 else "neutral")
        ctx_parts.append(f"Funding Rate: {funding:+.4f}% ({fn})")
    if oi:
        ctx_parts.append(f"Open Interest: {oi:,.0f} contracts")
    if adx:
        ctx_parts.append(f"Signal TF ADX: {adx:.1f} ({'trending' if adx > 25 else 'ranging'})")
    if vwap_side:
        ctx_parts.append(f"Price vs VWAP: {vwap_side}")
    ctx_str = "\n".join(ctx_parts) if ctx_parts else "No extended market context available."

    # ── Higher-timeframe context ───────────────────────────────────────────────
    htf_parts = []
    if htf_context:
        htf_regime    = htf_context.get("regime", {})
        htf_structure = htf_context.get("structure", {})
        htf_adx       = htf_context.get("adx", {})
        htf_rsi       = htf_context.get("rsi", {})
        htf_tf        = "1H" if interval == "15m" else "1D"
        htf_parts.append(f"  {htf_tf} Regime: {htf_regime.get('regime','?')} "
                         f"({'above' if htf_regime.get('above_200') else 'below'} 200 EMA)")
        if htf_structure:
            bos_type = ("Bullish BOS" if htf_structure.get("bullish_bos") else
                        "Bearish BOS" if htf_structure.get("bearish_bos") else
                        "HH/HL" if htf_structure.get("hh_hl") else
                        "LH/LL" if htf_structure.get("lh_ll") else "No clear structure")
            htf_parts.append(f"  {htf_tf} Structure: {bos_type}")
        if htf_adx:
            htf_parts.append(f"  {htf_tf} ADX: {htf_adx.get('value',0):.1f} "
                             f"({'trending' if htf_adx.get('trending') else 'ranging'})")
        if htf_rsi:
            htf_parts.append(f"  {htf_tf} RSI: {htf_rsi.get('value',0):.1f} ({htf_rsi.get('range','?')})")
    htf_str = "\n".join(htf_parts) if htf_parts else "  Higher-TF data not available."

    # ── Recent OHLCV candle summary ────────────────────────────────────────────
    candle_str = "Not available."
    if ohlcv_candles and len(ohlcv_candles) >= 5:
        recent = ohlcv_candles[-15:]
        lines  = []
        for c in recent:
            body_pct = abs(c["close"] - c["open"]) / c["open"] * 100 if c["open"] else 0
            wick_top = (c["high"] - max(c["open"], c["close"])) / c["open"] * 100 if c["open"] else 0
            wick_bot = (min(c["open"], c["close"]) - c["low"]) / c["open"] * 100 if c["open"] else 0
            color    = "bull" if c["close"] >= c["open"] else "bear"
            lines.append(
                f"  {color} O:{_fmt(c['open'])} H:{_fmt(c['high'])} "
                f"L:{_fmt(c['low'])} C:{_fmt(c['close'])} "
                f"body:{body_pct:.2f}% topWick:{wick_top:.2f}% botWick:{wick_bot:.2f}%"
            )
        candle_str = "\n".join(lines)

    # ── Claude's own accuracy ─────────────────────────────────────────────────
    acc_parts = []
    if ai_accuracy:
        for rec in ("strong_take", "take"):
            s = ai_accuracy.get(rec)
            if s and s["total"] >= 1:
                acc_parts.append(
                    f"  {rec}: {s['total']} trades → "
                    f"{s['wins']}W / {s['losses']}L "
                    f"({s['win_rate_pct']:.0f}% WR, avg ROI {s['avg_roi_pct']:+.2f}%)"
                )
    acc_str = (
        "\n".join(acc_parts) if acc_parts
        else "  No closed AI-evaluated trades yet — be very conservative."
    )

    system_prompt = (
        "You are a professional crypto quant trader and price action analyst. "
        "Your job is to PROTECT the trading system from bad entries. "
        "Default to SKIP unless the setup is clearly high quality. "
        "Respond ONLY with valid JSON — no markdown, no text outside the JSON."
    )

    user_prompt = f"""Evaluate this PENDING RETEST trade signal. Price has broken structure and we are waiting
for a pullback to the broken level before entering. Your job: assess if this is a quality setup worth waiting for.

SIGNAL DETAILS:
- Symbol: {sym} | Direction: {direction} | Timeframe: {interval} ({_tier(interval)} trade)
- BOS candle closed at: ${_fmt(current_px)} (this is where the break happened)
- Pending entry (retest level): ${_fmt(entry)} ({gap_pct:.1f}% away from current price)
- Stop loss (structural swing {'low' if direction == 'LONG' else 'high'}): ${_fmt(sl)} (-{sl_pct:.1f}% from entry)
- Take profit (structural resistance/support): ${_fmt(tp)} (+{tp_pct:.1f}% from entry)
- Technical Score: {score}/10 | R:R: {rr}:1
- Signal reason: {reason}
- Active factors: {json.dumps({k: v for k, v in factors.items() if v})}

HIGHER TIMEFRAME CONTEXT (critical — HTF must align):
{htf_str}

MARKET CONTEXT:
{ctx_str}

LAST 15 CANDLES (most recent last — assess BOS quality and momentum):
{candle_str}

SYSTEM TRACK RECORD:
- Win rate: {win_rate:.0f}% over {total} closed trades | Avg ROI: {avg_roi:.2f}%
- {sym} recent: {sym_wins}/{len(sym_closed)} wins

YOUR PAST ACCURACY:
{acc_str}

ANALYSIS TASKS:
1. BOS quality: Is this a genuine structural break or a stop-hunt/false break?
   - Check: Is the BOS candle a strong momentum candle or a small/wick-heavy bar?
   - Check: Is volume confirming the break? (already gated by system — double-check candle quality)
   - Check: Does price action BEFORE the BOS show a proper swing structure or just noise?

2. HTF alignment: Does the higher timeframe agree with this direction?
   - A LONG in a {interval} downtrend on the HTF is very high risk.

3. Entry quality: Is the retest level (${_fmt(entry)}) a meaningful support/resistance?
   - This level must be a clear structural pivot, not a random price.

4. Risk/reward reality: Does the {rr}:1 R:R hold given the current market context?

CONSERVATIVE RULES (override everything else):
- SKIP if HTF is opposed to the signal direction
- SKIP if BOS candle is a doji, spinning top, or has wick > 2× body size
- SKIP if the last 5 candles show heavy rejection in signal direction
- SKIP if score < 8 or R:R < 3
- SKIP if system has < 10 closed trades AND this is a marginal setup
- strong_take only if: HTF aligned + genuine momentum BOS candle + 4+ factors active + clean structure

Respond with ONLY this JSON:
{{"confidence": <0-100>, "recommendation": "<strong_take|take|skip>", "bos_quality": "<genuine|suspect|false_break>", "entry_assessment": "<1 sentence on whether the retest level is meaningful>", "reasoning": "<2 sentences max>", "risks": ["<specific risk>"], "positives": ["<specific positive>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=450,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception:
        return {}


# ── On-demand chart analysis ─────────────────────────────────────────────────

def analyze_chart(symbol: str, interval: str, data: dict) -> dict:
    """
    Ask Claude Haiku for a TA read on the currently viewed chart.
    Called by /api/ai-chart when the user opens the AI card.

    Returns {bias, confidence, summary, watch_for, key_levels, risks} or {}
    """
    client = _get_client()
    if not client:
        return {}

    price     = data.get("current_price", 0)
    regime    = data.get("regime", {})
    structure = data.get("structure", {}) or {}
    rsi_d     = data.get("rsi", {}) or {}
    macd_d    = data.get("macd", {}) or {}
    vol_d     = data.get("volume", {}) or {}
    conf_d    = data.get("confluence", {}) or {}
    risk_d    = data.get("risk", {}) or {}
    adx_d     = data.get("adx", {}) or {}
    vwap      = data.get("vwap")

    above_200 = regime.get("above_200", False)
    bos = ("Bullish BOS" if structure.get("bullish_bos") else
           "Bearish BOS" if structure.get("bearish_bos") else
           "HH/HL uptrend" if structure.get("hh_hl") else
           "LH/LL downtrend" if structure.get("lh_ll") else "No confirmed BOS")

    tier = _tier(interval)

    vwap_str = (f"${_fmt(vwap)} — price {'above' if price > vwap else 'below'}"
                if vwap else "N/A")

    prompt = f"""Analyze this {tier} crypto chart and provide a trading bias.

CHART: {symbol} | {interval} timeframe | Current price: ${_fmt(price)}

TECHNICAL DATA:
- Macro regime: {'Bullish' if above_200 else 'Bearish'} (200 EMA: ${_fmt(regime.get('ema200', 0))}, price {'above' if above_200 else 'below'})
- Market structure: {bos}
- RSI(14): {rsi_d.get('value', 0):.1f} ({rsi_d.get('range', '?')})
- MACD: {'above signal line' if macd_d.get('above_signal') else 'below signal line'}, histogram {'positive' if macd_d.get('histogram_positive') else 'negative'}
- Volume: {'expanding' if vol_d.get('expanding') else 'contracting'} ({vol_d.get('trend', '?')})
- ADX: {adx_d.get('value', 0):.1f} ({'trending' if adx_d.get('trending') else 'ranging/weak trend'})
- VWAP: {vwap_str}
- Confluence score: {conf_d.get('score', 0)}/{conf_d.get('max', 12)}
- R:R setup: {risk_d.get('rr', 0)}:1 ({'favorable' if risk_d.get('favorable') else 'unfavorable'})
- Entry: ${_fmt(risk_d.get('current', 0))} | Target: ${_fmt(risk_d.get('target', 0))} | Stop: ${_fmt(risk_d.get('invalidation', 0))}

Respond with ONLY this JSON (no markdown):
{{"bias":"<LONG|SHORT|NEUTRAL>","confidence":<0-100>,"summary":"<2 sentences max — current TA read>","watch_for":"<1 sentence — key trigger to confirm or invalidate>","key_levels":["<price: what it means>"],"risks":["<specific risk>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system="You are a professional crypto technical analyst. Be direct and concise. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if Claude wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[-2] if "```" in raw[3:] else raw
            raw = raw.lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


# ── Performance review ────────────────────────────────────────────────────────

def get_performance_insights(trade_history: list,
                             ai_accuracy: dict | None = None) -> dict:
    """
    Ask Claude Sonnet to analyze closed trade history and surface patterns.
    Only runs when called explicitly (e.g. via /api/ai-insights).
    Requires at least 10 closed trades.

    Returns: {overall_assessment, patterns, recommendations, watch_out} or {}
    """
    client = _get_client()
    if not client:
        return {}

    closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
    if len(closed) < 10:
        return {"error": f"Need at least 10 closed trades (have {len(closed)})"}

    wins   = [t for t in closed if t.get("status") == "win"]
    losses = [t for t in closed if t.get("status") == "loss"]

    # Per-symbol breakdown
    by_sym = {}
    for t in closed:
        s = t.get("symbol", "?")
        if s not in by_sym:
            by_sym[s] = {"wins": 0, "losses": 0, "total_roi": 0.0}
        if t.get("status") == "win":
            by_sym[s]["wins"] += 1
        else:
            by_sym[s]["losses"] += 1
        by_sym[s]["total_roi"] += float(t.get("roi_pct") or 0)

    # Factor win rates
    factor_wins = {}
    for t in closed:
        snap = t.get("factors_snapshot") or {}
        if isinstance(snap, str):
            try:
                snap = json.loads(snap)
            except Exception:
                snap = {}
        for f, present in snap.items():
            if f not in factor_wins:
                factor_wins[f] = {"wins": 0, "losses": 0}
            if present:
                if t.get("status") == "win":
                    factor_wins[f]["wins"] += 1
                else:
                    factor_wins[f]["losses"] += 1

    summary = {
        "total_trades":  len(closed),
        "win_rate_pct":  round(len(wins) / len(closed) * 100, 1),
        "avg_roi_pct":   round(sum(t.get("roi_pct") or 0 for t in closed) / len(closed), 2),
        "best_roi":      round(max((t.get("roi_pct") or 0) for t in closed), 2),
        "worst_roi":     round(min((t.get("roi_pct") or 0) for t in closed), 2),
        "per_symbol":    by_sym,
        "factor_stats":  factor_wins,
    }

    recent_fields = ["symbol", "interval", "direction", "score", "roi_pct", "status", "reason"]
    recent = [{k: t.get(k) for k in recent_fields} for t in closed[-20:]]

    # ── Claude's own prediction accuracy ──────────────────────────────────────
    acc_parts = []
    if ai_accuracy:
        for rec in ("strong_take", "take", "skip"):
            s = ai_accuracy.get(rec)
            if s and s["total"] >= 1:
                acc_parts.append(
                    f"  {rec}: {s['total']} evaluated → "
                    f"{s['wins']}W / {s['losses']}L "
                    f"({s['win_rate_pct']:.0f}% win rate, avg ROI {s['avg_roi_pct']:+.2f}%)"
                )
    acc_str = (
        "\n".join(acc_parts)
        if acc_parts
        else "  No AI-evaluated closed trades yet."
    )

    prompt = f"""Analyze this crypto algorithmic trading system's performance. Be specific and actionable.

PERFORMANCE SUMMARY:
{json.dumps(summary, indent=2)}

RECENT TRADES (last 20):
{json.dumps(recent, indent=2)}

CLAUDE AI PREDICTION ACCURACY (your own past recommendations on these trades):
{acc_str}

Respond with this exact JSON:
{{"overall_assessment": "<1-2 sentences on system health>", "patterns": ["<observed pattern with evidence>"], "recommendations": ["<specific actionable change>"], "watch_out": ["<specific warning or risk>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(msg.content[0].text.strip())
    except Exception:
        return {}


# ── Weekly strategy review ────────────────────────────────────────────────────

def get_weekly_review(trade_history: list,
                      current_params: dict,
                      learning_state: dict | None = None) -> dict:
    """
    Weekly Claude Sonnet review: reads closed trades, identifies patterns,
    and proposes concrete parameter changes with safety caps (±20% per cycle).

    Returns:
        {
            overall_assessment: str,
            patterns: [str],
            parameter_proposals: [
                {param, current, proposed, reason, change_pct}
            ],
            recommendations: [str],
            watch_out: [str]
        }
    or {} on failure / insufficient data.
    """
    client = _get_client()
    if not client:
        return {}

    closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
    if len(closed) < 10:
        return {"error": f"Need at least 10 closed trades (have {len(closed)})"}

    wins = [t for t in closed if t.get("status") == "win"]

    # Per-symbol breakdown
    by_sym: dict = {}
    for t in closed:
        s = t.get("symbol", "?")
        if s not in by_sym:
            by_sym[s] = {"wins": 0, "losses": 0, "total_roi": 0.0}
        if t.get("status") == "win":
            by_sym[s]["wins"] += 1
        else:
            by_sym[s]["losses"] += 1
        by_sym[s]["total_roi"] += float(t.get("roi_pct") or 0)

    # Factor win rates
    factor_wins: dict = {}
    for t in closed:
        snap = t.get("factors_snapshot") or {}
        if isinstance(snap, str):
            try:
                snap = json.loads(snap)
            except Exception:
                snap = {}
        for f, present in snap.items():
            if f not in factor_wins:
                factor_wins[f] = {"wins": 0, "losses": 0}
            if present:
                if t.get("status") == "win":
                    factor_wins[f]["wins"] += 1
                else:
                    factor_wins[f]["losses"] += 1

    summary = {
        "total_trades":  len(closed),
        "win_rate_pct":  round(len(wins) / len(closed) * 100, 1),
        "avg_roi_pct":   round(sum(t.get("roi_pct") or 0 for t in closed) / len(closed), 2),
        "per_symbol":    by_sym,
        "factor_stats":  factor_wins,
    }

    recent_fields = ["symbol", "interval", "direction", "score", "roi_pct",
                     "status", "reason", "factors_snapshot"]
    recent = [{k: t.get(k) for k in recent_fields} for t in closed[-20:]]

    adapt_log = (learning_state or {}).get("adaptation_log", [])[:5]

    prompt = f"""You are a trading system optimizer. Analyze this crypto algorithmic trading system's \
performance and propose specific, measurable parameter changes backed by the data.

PERFORMANCE SUMMARY:
{json.dumps(summary, indent=2)}

CURRENT PARAMETERS:
{json.dumps(current_params, indent=2)}

RECENT TRADES (last 20, oldest first):
{json.dumps(recent, indent=2)}

RECENT ADAPTATION LOG (last 5 events):
{json.dumps(adapt_log, indent=2)}

VALID PARAMETER NAMES (use ONLY these):
  signal_threshold, stop_multiplier, adx_threshold, body_ratio_min, min_rr,
  weight_regime, weight_bos, weight_sweep, weight_volume, weight_obv, weight_rsi, weight_adx

RULES FOR PROPOSALS:
- Keep each change within ±20% of the current value (hard safety cap)
- Only propose a change when you have clear data support (≥5 trades of evidence)
- Prefer small targeted changes over sweeping overhauls

Respond with ONLY this JSON (no markdown, no text outside the JSON):
{{"overall_assessment": "<1-2 sentences on system health>",
  "patterns": ["<observed pattern with specific evidence>"],
  "parameter_proposals": [
    {{"param": "<param_name>", "current": <current_value>, "proposed": <new_value>,
      "reason": "<specific data-backed reason, ≤60 words>"}}
  ],
  "recommendations": ["<specific actionable non-parameter recommendation>"],
  "watch_out": ["<specific risk or pattern to monitor>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)

        # Enforce ±20% safety cap on every proposal
        for proposal in result.get("parameter_proposals", []):
            try:
                current  = float(proposal.get("current", 0))
                proposed = float(proposal.get("proposed", 0))
                if current > 0:
                    capped   = max(current * 0.80, min(current * 1.20, proposed))
                    proposal["proposed"]   = round(capped, 4)
                    proposal["change_pct"] = round((capped - current) / current * 100, 1)
            except Exception:
                pass

        return result
    except Exception:
        return {}


# ── Deep AI Signal Analysis ───────────────────────────────────────────────────

def analyze_signal_deep(
    signal: dict,
    candles_tf: list,          # 200 candles of signal TF
    candles_htf: list,         # 100 candles of higher TF (4h or 1d)
    candles_htf2: list,        # 50 candles of even higher TF
    trade_history: list,
    market_context: dict,
) -> dict:
    """
    Deep AI analysis using Claude Sonnet. Called before every trade entry.

    Data fed to Claude:
      - 200 candles of signal TF (OHLCV)
      - 100 candles of higher TF
      - 50 candles of highest TF
      - Binance order book depth (liquidity clusters)
      - Recent crypto news (web fetch)
      - Recent X/Twitter sentiment (web search)
      - Bot's own past AI predictions and outcomes (feedback loop)
      - Full market context (Fear & Greed, funding, dominance)

    Returns: {recommendation, confidence, reasoning, liquidity_notes,
              news_sentiment, risks, positives}
    recommendation: "strong_take" | "take" | "skip"
    """
    import requests as _req
    import numpy as _np

    client = _get_client()
    if not client:
        return {}

    sym       = signal.get("symbol", "BTCUSDT").replace("USDT", "")
    sym_full  = signal.get("symbol", "BTCUSDT")
    direction = signal.get("direction", "")
    interval  = signal.get("interval", "1h")
    entry     = float(signal.get("entry") or 0)
    tp        = float(signal.get("tp") or signal.get("target") or 0)
    sl        = float(signal.get("sl") or signal.get("stop") or 0)
    reason    = signal.get("reason", "")
    sig_type  = signal.get("signal_type", "")

    # ── 1. Binance order book — find liquidity clusters ──────────────────────
    liq_summary = "Order book unavailable."
    try:
        ob = _req.get(
            "https://api.binance.us/api/v3/depth",
            params={"symbol": sym_full, "limit": 100},
            timeout=8,
        ).json()
        bids = [(float(p), float(q)) for p, q in ob.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in ob.get("asks", [])]

        # Find large walls (top 5 by size on each side)
        top_bids = sorted(bids, key=lambda x: x[1], reverse=True)[:5]
        top_asks = sorted(asks, key=lambda x: x[1], reverse=True)[:5]

        bid_walls = ", ".join(f"${p:,.4f} ({q:,.1f})" for p, q in sorted(top_bids, key=lambda x: x[0], reverse=True))
        ask_walls = ", ".join(f"${p:,.4f} ({q:,.1f})" for p, q in sorted(top_asks, key=lambda x: x[0]))

        # Total bid/ask volume in book — imbalance ratio
        total_bid_vol = sum(q for _, q in bids[:50])
        total_ask_vol = sum(q for _, q in asks[:50])
        imbalance = total_bid_vol / total_ask_vol if total_ask_vol > 0 else 1.0
        imbalance_str = f"{imbalance:.2f}x ({'BUY' if imbalance > 1.2 else 'SELL' if imbalance < 0.8 else 'NEUTRAL'} pressure)"

        liq_summary = (
            f"Large BID walls (buy support): {bid_walls}\n"
            f"Large ASK walls (sell resistance): {ask_walls}\n"
            f"Order book imbalance (bid/ask vol ratio top-50): {imbalance_str}"
        )
    except Exception as _e:
        liq_summary = f"Order book fetch failed: {_e}"

    # ── 2. Recent news — fetch from CryptoPanic (free, no key needed) ────────
    news_summary = "News unavailable."
    try:
        news_r = _req.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params={"auth_token": "free", "currencies": sym, "public": "true", "filter": "hot"},
            timeout=8,
        )
        news_data = news_r.json()
        headlines = []
        for item in news_data.get("results", [])[:8]:
            title = item.get("title", "")
            votes = item.get("votes", {})
            sentiment = "🐂" if votes.get("positive", 0) > votes.get("negative", 0) else "🐻" if votes.get("negative", 0) > votes.get("positive", 0) else "—"
            headlines.append(f"{sentiment} {title}")
        news_summary = "\n".join(headlines) if headlines else "No recent news found."
    except Exception:
        news_summary = "News fetch failed."

    # ── 3. X/Twitter sentiment — search via Nitter or web ────────────────────
    x_summary = "X sentiment unavailable."
    try:
        # Use DuckDuckGo search for recent X posts about the coin
        search_r = _req.get(
            "https://api.duckduckgo.com/",
            params={"q": f"#{sym} crypto site:x.com OR site:twitter.com", "format": "json", "no_html": "1"},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        ddg = search_r.json()
        topics = ddg.get("RelatedTopics", [])[:5]
        snippets = [t.get("Text", "") for t in topics if t.get("Text")]
        x_summary = "\n".join(snippets[:4]) if snippets else "No X sentiment data found."
    except Exception:
        x_summary = "X sentiment fetch failed."

    # ── 4. Equal highs/lows — liquidity pool detection ────────────────────────
    liquidity_pools = []
    try:
        if candles_tf and len(candles_tf) >= 50:
            highs  = [c["high"]  for c in candles_tf[-100:]]
            lows   = [c["low"]   for c in candles_tf[-100:]]
            closes = [c["close"] for c in candles_tf[-100:]]

            # Find clusters of highs within 0.3% of each other (equal highs = resting sell stops)
            for i in range(len(highs)):
                cluster = [h for h in highs if abs(h - highs[i]) / highs[i] < 0.003]
                if len(cluster) >= 3:
                    pool_px = round(sum(cluster) / len(cluster), 6)
                    liquidity_pools.append(f"SELL STOPS cluster near ${pool_px:,.4f} ({len(cluster)} equal highs)")

            # Find clusters of lows (equal lows = resting buy stops)
            for i in range(len(lows)):
                cluster = [l for l in lows if abs(l - lows[i]) / lows[i] < 0.003]
                if len(cluster) >= 3:
                    pool_px = round(sum(cluster) / len(cluster), 6)
                    liquidity_pools.append(f"BUY STOPS cluster near ${pool_px:,.4f} ({len(cluster)} equal lows)")

            # Deduplicate
            seen = set()
            unique_pools = []
            for p in liquidity_pools:
                key = p[:40]
                if key not in seen:
                    seen.add(key)
                    unique_pools.append(p)
            liquidity_pools = unique_pools[:8]
    except Exception:
        pass

    liq_pools_str = "\n".join(liquidity_pools) if liquidity_pools else "No clear liquidity clusters detected."

    # ── 5. AI feedback loop — past predictions and outcomes ───────────────────
    ai_feedback = ""
    try:
        closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
        recent = [t for t in closed if t.get("ai_analysis")][-10:]
        if recent:
            feedback_lines = []
            for t in recent:
                try:
                    ai_prev = json.loads(t["ai_analysis"]) if isinstance(t["ai_analysis"], str) else t["ai_analysis"]
                    prev_rec = ai_prev.get("recommendation", "?")
                    outcome  = t.get("status", "?")
                    roi      = t.get("roi_pct", 0) or 0
                    correct  = (prev_rec in ("take", "strong_take") and outcome == "win") or (prev_rec == "skip" and outcome == "loss")
                    feedback_lines.append(
                        f"  {t.get('symbol')} {t.get('interval')} {t.get('direction')}: "
                        f"AI said '{prev_rec}' → outcome={outcome} ({roi:+.2f}%) {'✓' if correct else '✗'}"
                    )
                except Exception:
                    pass
            if feedback_lines:
                ai_feedback = "Past AI predictions and outcomes (learn from these):\n" + "\n".join(feedback_lines)
    except Exception:
        pass

    # ── 6. Format candle summaries ────────────────────────────────────────────
    def _fmt_candles(candles: list, label: str, limit: int = 50) -> str:
        if not candles:
            return f"{label}: no data"
        c = candles[-limit:]
        rows = []
        for i, bar in enumerate(c):
            chg = (bar["close"] - bar["open"]) / bar["open"] * 100 if bar["open"] else 0
            rows.append(
                f"  [{i+1:3d}] O={bar['open']:.4f} H={bar['high']:.4f} "
                f"L={bar['low']:.4f} C={bar['close']:.4f} "
                f"V={bar.get('volume',0):,.0f} ({chg:+.2f}%)"
            )
        return f"{label} (last {len(c)} bars):\n" + "\n".join(rows)

    candle_block   = _fmt_candles(candles_tf,   f"{sym_full} {interval}",   60)
    htf_block      = _fmt_candles(candles_htf,  f"{sym_full} 4h",           30)
    htf2_block     = _fmt_candles(candles_htf2, f"{sym_full} 1d",           14)

    # ── 7. Market context ─────────────────────────────────────────────────────
    fg      = market_context.get("fear_greed", {})
    btc_dom = market_context.get("btc_dominance", 0)
    funding = market_context.get("funding_rate", 0)
    ctx_str = f"""Fear & Greed: {fg.get('value','?')} — {fg.get('label','?')}
BTC Dominance: {btc_dom:.1f}%
Funding Rate: {funding:+.4f}% ({'longs paying' if funding > 0 else 'shorts paying'})"""

    # ── 8. Bot performance context ────────────────────────────────────────────
    closed_all  = [t for t in trade_history if t.get("status") in ("win", "loss")]
    total_trades = len(closed_all)
    wins_all    = sum(1 for t in closed_all if t.get("status") == "win")
    win_rate    = wins_all / total_trades * 100 if total_trades else 0
    recent_5    = closed_all[-5:]
    recent_str  = " ".join("W" if t.get("status") == "win" else "L" for t in recent_5)

    rr = abs((tp - entry) / (entry - sl)) if entry != sl else 0

    # ── 9. Build the prompt ───────────────────────────────────────────────────
    prompt = f"""You are a professional crypto trading analyst with deep expertise in price action, liquidity, order flow, and market microstructure. Your job is to evaluate whether this trade setup should be taken or skipped.

═══════════════════════════════════════════════════════════
TRADE SETUP
═══════════════════════════════════════════════════════════
Symbol:    {sym_full}
Direction: {direction}
Timeframe: {interval}
Signal:    {sig_type}
Reason:    {reason}
Entry:     {_fmt(entry)}
TP:        {_fmt(tp)}  ({abs((tp-entry)/entry*100):.2f}% away)
SL:        {_fmt(sl)}  ({abs((sl-entry)/entry*100):.2f}% away)
R:R ratio: {rr:.2f}:1

═══════════════════════════════════════════════════════════
MACRO MARKET CONTEXT
═══════════════════════════════════════════════════════════
{ctx_str}

Bot performance: {total_trades} closed trades, {win_rate:.0f}% win rate
Recent 5 trades: {recent_str if recent_str else 'none yet'}

═══════════════════════════════════════════════════════════
PRICE ACTION — {interval.upper()} CHART (60 bars)
═══════════════════════════════════════════════════════════
{candle_block}

═══════════════════════════════════════════════════════════
HIGHER TIMEFRAME — 4H (30 bars)
═══════════════════════════════════════════════════════════
{htf_block}

═══════════════════════════════════════════════════════════
DAILY CHART (14 bars)
═══════════════════════════════════════════════════════════
{htf2_block}

═══════════════════════════════════════════════════════════
LIQUIDITY CLUSTERS (equal highs/lows where stops rest)
═══════════════════════════════════════════════════════════
{liq_pools_str}

═══════════════════════════════════════════════════════════
ORDER BOOK DEPTH (live Binance)
═══════════════════════════════════════════════════════════
{liq_summary}

═══════════════════════════════════════════════════════════
RECENT NEWS
═══════════════════════════════════════════════════════════
{news_summary}

═══════════════════════════════════════════════════════════
X / SOCIAL SENTIMENT
═══════════════════════════════════════════════════════════
{x_summary}

═══════════════════════════════════════════════════════════
AI FEEDBACK LOOP (your past predictions vs outcomes)
═══════════════════════════════════════════════════════════
{ai_feedback if ai_feedback else 'No prior AI predictions to learn from yet.'}

═══════════════════════════════════════════════════════════
YOUR ANALYSIS TASK
═══════════════════════════════════════════════════════════
Analyze ALL of the above. Consider:

1. TREND ALIGNMENT: Do all three timeframes (signal TF, 4h, daily) agree on direction?
2. LIQUIDITY: Is this trade entering INTO a liquidity cluster (dangerous) or AWAY from one (safe)? Where are stops likely resting that could fuel a move in trade direction?
3. ORDER BOOK: Does the live order book support this direction? Any large walls that would stop the move?
4. NEWS/SENTIMENT: Any catalysts that support or oppose this trade?
5. STRUCTURE: Is the entry at a meaningful level (broken structure retest, strong S/R) or just noise?
6. RISK: What specifically could go wrong?

SKIP RULES (non-negotiable):
- Skip if daily trend opposes the signal direction
- Skip if entering directly into a large liquidity cluster (likely to reverse)
- Skip if negative news catalyst present
- Skip if Fear & Greed below 20 (extreme fear) for longs, above 80 for shorts
- Skip if order book shows massive wall directly in path of TP

Respond with ONLY this JSON (no markdown, no extra text):
{{"recommendation": "<strong_take|take|skip>", "confidence": <0-100>, "reasoning": "<3-4 sentences explaining your decision>", "liquidity_notes": "<1-2 sentences on key liquidity levels near entry/TP/SL>", "news_sentiment": "<bullish|bearish|neutral>", "trend_alignment": "<aligned|opposed|mixed>", "risks": ["<specific risk 1>", "<specific risk 2>"], "positives": ["<positive 1>", "<positive 2>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}

"""
LEARNING AGENT — runs every hour and after every trade closes
Job: Review closed trades. Write post-mortems. Identify win/loss patterns.
     Update the knowledge base. Propose concrete strategy improvements.
     This is the agent that makes the system smarter over time.
Output: post-mortems, patterns, improvement proposals
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone

import requests
import anthropic

from agents.state import (set_state, get_state, add_report, add_knowledge,
                          save_postmortem, get_postmortems, get_knowledge,
                          post_to_render)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BINANCE_BASE  = "https://api.binance.us/api/v3"
RENDER_URL    = os.environ.get("RENDER_URL", "http://localhost:8080")


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def get_all_trades() -> list:
    try:
        import time as _t
        # Cache-busting param ensures we get live data, not a CDN-cached response
        r = requests.get(f"{RENDER_URL}/api/trades?ts={int(_t.time())}", timeout=12)
        if r.status_code == 200:
            trades = r.json()
            by_status = {}
            for t in trades:
                s = t.get("status", "?")
                by_status[s] = by_status.get(s, 0) + 1
            print(f"  Fetched {len(trades)} trades: {by_status}", flush=True)
            return trades
    except Exception as e:
        print(f"  get_all_trades error: {e}", flush=True)
    return []


def get_candles_around_trade(symbol: str, opened_at: str,
                              closed_at: str, interval: str = "1h") -> list:
    """Fetch candles for the duration of a trade + context."""
    try:
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": 40},
            timeout=10,
        )
        return [{"open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]),
                 "volume": float(c[5]), "time": c[0]}
                for c in r.json()]
    except Exception:
        return []


def write_postmortem(trade: dict, candles: list,
                     macro_context: dict, analyst_context: dict) -> dict:
    """Ask Claude to write a detailed post-mortem on a single trade."""
    client = _claude()
    if not client:
        return {}

    outcome   = trade.get("status", "?")
    sym       = trade.get("symbol", "")
    direction = trade.get("direction", "")
    entry     = float(trade.get("entry", 0))
    tp        = float(trade.get("tp", 0))
    sl        = float(trade.get("sl", 0))
    close_px  = float(trade.get("close_price", 0) or 0)
    roi       = float(trade.get("roi_pct", 0) or 0)
    reason    = trade.get("reason", "")
    opened    = trade.get("opened_at", "")
    closed    = trade.get("closed_at", "")
    sig_type  = trade.get("signal_type", trade.get("tp_source", ""))

    candle_str = "\n".join(
        f"O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f}"
        for c in candles[-20:]
    )

    # Get prior knowledge for context
    prior_patterns = get_knowledge("win_patterns", 5) + get_knowledge("loss_patterns", 5)
    pattern_str    = "\n".join(p.get("pattern", "") for p in prior_patterns if p.get("pattern"))

    prompt = f"""You are the Learning Agent on a crypto trading desk. Write a detailed post-mortem on this closed trade. This is CRYPTO — everything moves with BTC, and the 15M and 1H candles tell the real story.

TRADE DETAILS:
{sym} {direction} on {trade.get('interval','')}
Signal type: {sig_type}
Entry: {entry:.6f}  |  TP: {tp:.6f}  |  SL: {sl:.6f}
Close: {close_px:.6f}  |  ROI: {roi:+.2f}%
Outcome: {outcome.upper()}
Opened: {opened}  |  Closed: {closed}
Original signal reason: {reason}

CANDLES DURING THE TRADE (1H):
{candle_str}

MACRO AT TIME OF TRADE:
Regime: {macro_context.get('regime_type','unknown')}
Risk level: {macro_context.get('risk_level','unknown')}
Coin bias: {macro_context.get('coin_bias',{}).get(sym.replace('USDT',''),'unknown')}

KNOWN PATTERNS SO FAR:
{pattern_str if pattern_str else 'Still building — no patterns yet.'}

Analyze this trade like a senior crypto trader reviewing a junior's position. Ask:
1. What was the 15M/1H structure ACTUALLY doing when this trade was entered? Was the candle pattern valid?
2. Was the entry at a real structural level or mid-range?
3. Did BTC's structure support this directional trade? Or was the bot trading against BTC?
4. What did price do after entry — did it immediately go wrong, or did it work initially then reverse?
5. What specific 15M candle signal would have confirmed OR warned against this trade?
6. What rule should the bot follow to avoid repeating this mistake (or reinforce this win)?

Respond with ONLY this JSON:
{{
  "what_happened": "<2-3 sentences: what price action actually did on the 15M/1H during this trade>",
  "why_it_{outcome}": "<2-3 sentences: specific candle/structure reason this trade won or lost>",
  "was_btc_aligned": "<yes|no|neutral — was BTC moving in the same direction as this trade?>",
  "was_entry_at_level": "<yes|no — was the entry at a real 15M/1H structural level or mid-range?>",
  "was_setup_valid": "<yes|no|marginal>",
  "should_have_been_skipped": <true|false>,
  "skip_reason": "<why it should have been skipped, or null if it was valid>",
  "key_lesson": "<the single most important thing to apply to the NEXT similar setup>",
  "pattern_identified": "<short pattern name, e.g. 'short_against_btc_uptrend', '15m_exhaustion_reversal', 'mid_range_entry_no_level'>",
  "what_to_look_for_next_time": "<specific 15M candle condition to check before taking a similar setup>",
  "improvement_suggestion": "<one concrete rule the analyst should apply going forward>"
}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  Postmortem error: {e}", flush=True)
        return {}


def identify_patterns(closed_trades: list) -> dict:
    """Analyze all closed trades and find statistical patterns."""
    client = _claude()
    if not client or len(closed_trades) < 2:
        return {}

    # Format trades for analysis
    trade_lines = []
    for t in closed_trades[-30:]:
        line = (f"{t.get('symbol')} {t.get('interval')} {t.get('direction')} "
                f"| {t.get('status')} ({t.get('roi_pct',0) or 0:+.2f}%) "
                f"| signal: {t.get('tp_source','')} "
                f"| score: {t.get('score',0)} "
                f"| reason: {t.get('reason','')[:60]}")
        trade_lines.append(line)

    wins   = [t for t in closed_trades if t.get("status") == "win"]
    losses = [t for t in closed_trades if t.get("status") == "loss"]

    win_rate  = len(wins)  / len(closed_trades) * 100
    avg_win   = sum(t.get("roi_pct",0) or 0 for t in wins)   / max(len(wins),   1)
    avg_loss  = sum(t.get("roi_pct",0) or 0 for t in losses) / max(len(losses), 1)
    best_sym  = max(set(t.get("symbol","") for t in wins),
                    key=lambda s: sum(1 for t in wins if t.get("symbol")==s), default="?")
    worst_sym = max(set(t.get("symbol","") for t in losses),
                    key=lambda s: sum(1 for t in losses if t.get("symbol")==s), default="?")

    prior_improvements = get_knowledge("improvements", 5)
    prior_str = "\n".join(p.get("suggestion","") for p in prior_improvements) or "None yet."

    prompt = f"""You are the Learning Agent reviewing all closed trades to find patterns.

STATISTICS:
Total: {len(closed_trades)} | Win rate: {win_rate:.0f}%
Avg win: {avg_win:+.2f}% | Avg loss: {avg_loss:+.2f}%
Best performing coin: {best_sym} | Worst: {worst_sym}

ALL RECENT TRADES:
{chr(10).join(trade_lines)}

PRIOR IMPROVEMENTS ALREADY SUGGESTED:
{prior_str}

Identify patterns and propose improvements. Respond with ONLY this JSON:
{{
  "win_patterns": [
    {{"pattern": "<what winning trades have in common>", "frequency": "<how often>"}}
  ],
  "loss_patterns": [
    {{"pattern": "<what losing trades have in common>", "frequency": "<how often>"}}
  ],
  "biggest_edge": "<what is this bot actually good at>",
  "biggest_weakness": "<what keeps causing losses>",
  "top_improvement": "<the single highest-impact change to make right now>",
  "secondary_improvement": "<second improvement>",
  "coins_to_focus": ["<coin1>", "<coin2>"],
  "coins_to_avoid": ["<coin1>"],
  "timeframes_to_focus": ["<tf1>"],
  "confidence": <1-10>
}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        import re as _re
        raw = _re.sub(r'//[^\n]*', '', raw)
        raw = _re.sub(r',\s*([\]}])', r'\1', raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            open_b = raw.count("{") - raw.count("}")
            open_a = raw.count("[") - raw.count("]")
            raw += "]" * max(open_a, 0) + "}" * max(open_b, 0)
            return json.loads(raw)
    except Exception as e:
        print(f"  Pattern analysis error: {e}", flush=True)
        return {}



def run() -> dict:
    """Execute learning cycle."""
    print("  Running...", flush=True)

    trades         = get_all_trades()
    closed_trades  = [t for t in trades if t.get("status") in ("win", "loss")]
    macro          = get_state("macro_regime", {})
    analyst        = get_state("analyst_ratings", {})

    if not closed_trades:
        print("  No closed trades yet.", flush=True)
        return {}

    # ── Write post-mortems for trades not yet analyzed ─────────────────────
    existing_pms = {p["trade_id"] for p in get_postmortems(200)}
    new_trades   = [t for t in closed_trades if t.get("id") not in existing_pms]

    postmortems_written = 0
    for trade in new_trades[:5]:  # max 5 per cycle
        sym     = trade.get("symbol", "BTCUSDT")
        candles = get_candles_around_trade(sym, trade.get("opened_at",""),
                                           trade.get("closed_at",""))
        print(f"    Writing post-mortem: {trade.get('id','')[:20]}...", flush=True)
        pm = write_postmortem(trade, candles, macro, analyst)

        if pm:
            save_postmortem(
                trade_id  = trade.get("id",""),
                symbol    = sym,
                direction = trade.get("direction",""),
                outcome   = trade.get("status",""),
                roi       = float(trade.get("roi_pct",0) or 0),
                analysis  = pm.get("what_happened","") + " " + pm.get(f"why_it_{trade.get('status','')}",""),
                lessons   = pm.get("key_lesson",""),
            )

            # Add to knowledge base
            category = "win_patterns" if trade.get("status") == "win" else "loss_patterns"
            add_knowledge(category, {
                "pattern":   pm.get("pattern_identified",""),
                "lesson":    pm.get("key_lesson",""),
                "check":     pm.get("what_to_look_for_next_time",""),
                "symbol":    sym,
                "direction": trade.get("direction",""),
            })

            if pm.get("improvement_suggestion"):
                add_knowledge("improvements", {"suggestion": pm.get("improvement_suggestion")})

            # Post post-mortem to Render
            post_to_render("/api/agent/report", {
                "type":      "postmortem",
                "agent":     "learning",
                "trade_id":  trade.get("id",""),
                "symbol":    sym,
                "outcome":   trade.get("status",""),
                "roi":       trade.get("roi_pct"),
                "analysis":  pm,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            postmortems_written += 1

    # ── Pattern analysis — observations only, no blocking rules ───────────────
    # The bot never restricts which coins or directions it can trade.
    # A loss on ETH SHORT doesn't mean avoid ETH SHORT — it means look harder
    # at the next ETH SHORT setup. The analyst reads these as context, not rules.
    patterns = {}
    if len(closed_trades) >= 2:
        print(f"    Analyzing patterns across {len(closed_trades)} trade(s)...", flush=True)
        patterns = identify_patterns(closed_trades)

        if patterns:
            add_knowledge("strategy_insights", patterns)
            if patterns.get("biggest_weakness"):
                print(f"    Weakness to watch: {patterns.get('biggest_weakness','')[:80]}", flush=True)
            if patterns.get("top_improvement"):
                print(f"    Key lesson: {patterns.get('top_improvement','')[:80]}", flush=True)

            post_to_render("/api/agent/report", {
                "type":               "strategy_improvement",
                "agent":              "learning",
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "biggest_edge":       patterns.get("biggest_edge"),
                "biggest_weakness":   patterns.get("biggest_weakness"),
                "top_improvement":    patterns.get("top_improvement"),
                "win_patterns":       patterns.get("win_patterns"),
                "loss_patterns":      patterns.get("loss_patterns"),
            })

    # Clear any old blocking rules — we don't use them
    set_state("strategy_rules", [])

    result = {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "closed_trades":       len(closed_trades),
        "postmortems_written": postmortems_written,
        "patterns":            patterns,
    }

    set_state("learning_status", result)
    add_report("learning", "cycle_complete", result)

    if postmortems_written:
        print(f"  Wrote {postmortems_written} new trade post-mortem(s).", flush=True)
    elif closed_trades:
        print(f"  {len(closed_trades)} closed trade(s) reviewed.", flush=True)
    return result

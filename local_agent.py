#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  LOCAL INTELLIGENCE AGENT — runs 24/7 on Mac Mini
  Continuously gathers market data, news, sentiment, and improves the bot.
═══════════════════════════════════════════════════════════════════════════════

What this agent does every 15 minutes:
  1. Fetches candles, order book, funding rates for all 5 coins
  2. Scrapes crypto news and X/social sentiment
  3. Detects liquidity clusters, regime shifts, key levels
  4. Reviews open and closed trades on Render
  5. Posts intelligence reports back to Render
  6. Generates post-mortems on closed trades (why it won/lost)
  7. Proposes strategy improvements and posts them to the dashboard
  8. Builds a local knowledge base that gets richer over time

Set environment variables:
  ANTHROPIC_API_KEY   — Claude API key
  RENDER_URL          — https://cryptodashboard-nuf5.onrender.com
  DISCORD_WEBHOOK_URL — Discord webhook for agent reports (optional)

Run: python3 local_agent.py
Auto-start: use the included com.cryptobot.agent.plist LaunchAgent
"""
from __future__ import annotations

import os, json, time, sqlite3, traceback, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
RENDER_URL    = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_URL   = os.environ.get("DISCORD_WEBHOOK_URL", "")
COINS         = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT"]
BINANCE_BASE  = "https://api.binance.us/api/v3"
SCAN_INTERVAL = 15 * 60   # 15 minutes

# Local knowledge base
KB_DIR  = Path.home() / "CryptoDashboard" / "agent_kb"
KB_DIR.mkdir(parents=True, exist_ok=True)
KB_FILE = KB_DIR / "knowledge_base.json"
LOG_FILE = KB_DIR / "agent.log"


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Knowledge base ────────────────────────────────────────────────────────────
def load_kb() -> dict:
    try:
        if KB_FILE.exists():
            return json.loads(KB_FILE.read_text())
    except Exception:
        pass
    return {
        "market_memory":    [],   # list of key market observations
        "pattern_library":  [],   # patterns that preceded wins/losses
        "regime_history":   {},   # coin → list of regime observations
        "news_seen":        [],   # hashes of news already processed
        "postmortems":      [],   # trade post-mortems
        "improvements":     [],   # proposed strategy changes
        "last_scan":        None,
    }


def save_kb(kb: dict) -> None:
    try:
        KB_FILE.write_text(json.dumps(kb, indent=2, default=str))
    except Exception as e:
        log(f"KB save error: {e}")


def kb_add_memory(kb: dict, category: str, entry: dict) -> None:
    """Add to knowledge base, keep last 500 per category."""
    lst = kb.setdefault(category, [])
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    lst.append(entry)
    kb[category] = lst[-500:]


# ── Market data ───────────────────────────────────────────────────────────────
def fetch_candles(symbol: str, interval: str, limit: int = 200) -> list:
    try:
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        return [
            {
                "time":   row[0],
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            }
            for row in r.json()
        ]
    except Exception as e:
        log(f"Candle fetch {symbol} {interval}: {e}")
        return []


def fetch_orderbook(symbol: str) -> dict:
    try:
        r = requests.get(
            f"{BINANCE_BASE}/depth",
            params={"symbol": symbol, "limit": 100},
            timeout=8,
        )
        return r.json()
    except Exception:
        return {}


def fetch_funding_rate(symbol: str) -> float:
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 1},
            timeout=8,
        )
        data = r.json()
        if data:
            return float(data[-1].get("fundingRate", 0))
    except Exception:
        pass
    return 0.0


def fetch_fear_greed() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        return {}


def fetch_news(symbol: str) -> list:
    coin = symbol.replace("USDT", "")
    headlines = []
    try:
        r = requests.get(
            "https://cryptopanic.com/api/free/v1/posts/",
            params={"auth_token": "free", "currencies": coin, "public": "true", "filter": "hot"},
            timeout=8,
        )
        for item in r.json().get("results", [])[:10]:
            title   = item.get("title", "")
            votes   = item.get("votes", {})
            pos     = votes.get("positive", 0)
            neg     = votes.get("negative", 0)
            sentiment = "bullish" if pos > neg else "bearish" if neg > pos else "neutral"
            headlines.append({"title": title, "sentiment": sentiment, "url": item.get("url", "")})
    except Exception:
        pass
    return headlines


def detect_liquidity_clusters(candles: list, entry: float) -> list:
    """Identify equal highs/lows near entry price — where stops are resting."""
    clusters = []
    if not candles or len(candles) < 20:
        return clusters
    try:
        highs  = [c["high"]  for c in candles[-100:]]
        lows   = [c["low"]   for c in candles[-100:]]

        seen_h, seen_l = set(), set()
        for h in highs:
            if any(abs(h - h2) / h < 0.003 for h2 in highs if h2 != h):
                key = round(h, 4)
                if key not in seen_h:
                    seen_h.add(key)
                    dist = abs(h - entry) / entry * 100
                    count = sum(1 for hh in highs if abs(hh - h) / h < 0.003)
                    if dist < 8 and count >= 2:
                        clusters.append({"type": "sell_stops", "price": round(h, 6),
                                         "count": count, "dist_pct": round(dist, 2)})

        for l in lows:
            if any(abs(l - l2) / l < 0.003 for l2 in lows if l2 != l):
                key = round(l, 4)
                if key not in seen_l:
                    seen_l.add(key)
                    dist = abs(l - entry) / entry * 100
                    count = sum(1 for ll in lows if abs(ll - l) / l < 0.003)
                    if dist < 8 and count >= 2:
                        clusters.append({"type": "buy_stops", "price": round(l, 6),
                                         "count": count, "dist_pct": round(dist, 2)})

        clusters.sort(key=lambda x: x["dist_pct"])
    except Exception:
        pass
    return clusters[:8]


# ── Render API ────────────────────────────────────────────────────────────────
def get_render_trades() -> list:
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def post_agent_insight(insight: dict) -> bool:
    """Post intelligence report to Render dashboard."""
    try:
        r = requests.post(
            f"{RENDER_URL}/api/agent/insight",
            json=insight,
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


def post_agent_report(report: dict) -> bool:
    """Post periodic strategy report to Render."""
    try:
        r = requests.post(
            f"{RENDER_URL}/api/agent/report",
            json=report,
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


# ── Discord ───────────────────────────────────────────────────────────────────
def discord_post(content: str, title: str = "", color: int = 0x5865F2) -> None:
    if not DISCORD_URL:
        return
    try:
        payload = {
            "embeds": [{
                "title":       title or "Agent Report",
                "description": content[:3900],
                "color":       color,
                "footer":      {"text": f"Mac Mini Agent • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
            }]
        }
        requests.post(DISCORD_URL, json=payload, timeout=8)
    except Exception:
        pass


# ── Claude AI ─────────────────────────────────────────────────────────────────
def get_claude_client():
    if not ANTHROPIC_KEY:
        return None
    try:
        return anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    except Exception:
        return None


def claude_market_analysis(coin_data: dict, kb: dict, trades: list) -> dict:
    """
    Deep market analysis for all coins. Returns structured intelligence.
    Called every 15 minutes. Uses Claude Sonnet.
    """
    client = get_claude_client()
    if not client:
        return {}

    # Build coin summaries
    coin_blocks = []
    for sym, data in coin_data.items():
        candles_1h = data.get("candles_1h", [])
        candles_4h = data.get("candles_4h", [])
        candles_1d = data.get("candles_1d", [])
        ob         = data.get("orderbook", {})
        funding    = data.get("funding", 0)
        news       = data.get("news", [])
        clusters   = data.get("liquidity_clusters", [])

        cur = candles_1h[-1]["close"] if candles_1h else 0
        change_24h = ((cur - candles_1h[-25]["close"]) / candles_1h[-25]["close"] * 100) if len(candles_1h) >= 25 else 0
        change_7d  = ((cur - candles_1d[-8]["close"])  / candles_1d[-8]["close"]  * 100) if len(candles_1d) >= 8  else 0

        # Order book imbalance
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid_vol = sum(float(q) for _, q in bids[:30])
        ask_vol = sum(float(q) for _, q in asks[:30])
        ob_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0

        # Recent 1h candles (last 10)
        recent_candles = ""
        for c in candles_1h[-10:]:
            chg = (c["close"] - c["open"]) / c["open"] * 100
            recent_candles += f"  O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f} ({chg:+.2f}%)\n"

        # 4h last 5
        htf_candles = ""
        for c in candles_4h[-5:]:
            chg = (c["close"] - c["open"]) / c["open"] * 100
            htf_candles += f"  4H: O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f} ({chg:+.2f}%)\n"

        news_str = "\n".join(f"  [{n['sentiment']}] {n['title']}" for n in news[:5]) or "  No news."
        liq_str  = "\n".join(f"  {c['type']} @ {c['price']} ({c['count']}x, {c['dist_pct']:.1f}% away)"
                             for c in clusters[:4]) or "  No clear clusters."

        coin_blocks.append(f"""
── {sym} ──────────────────────────────────────────
Price: {cur:.6f}  |  24h: {change_24h:+.2f}%  |  7d: {change_7d:+.2f}%
Funding: {funding:+.4f}%  |  OrderBook ratio (bid/ask): {ob_ratio:.2f}x

Recent 1H candles:
{recent_candles}
4H context:
{htf_candles}
Liquidity clusters:
{liq_str}
News:
{news_str}""")

    # Recent trade context
    closed = [t for t in trades if t.get("status") in ("win", "loss")][-15:]
    open_  = [t for t in trades if t.get("status") == "open"]
    trade_summary = f"Open: {len(open_)} | Recent closed: {len(closed)}"
    if closed:
        wins = sum(1 for t in closed if t.get("status") == "win")
        trade_summary += f" | Win rate: {wins/len(closed)*100:.0f}%"
    recent_trades_str = "\n".join(
        f"  {t.get('symbol')} {t.get('interval')} {t.get('direction')} → {t.get('status')} ({t.get('roi_pct',0) or 0:+.2f}%)"
        for t in closed[-8:]
    )

    # Knowledge base context
    recent_patterns = kb.get("pattern_library", [])[-5:]
    pattern_str = "\n".join(f"  {p.get('summary','')}" for p in recent_patterns) or "  Building pattern library..."

    fg = coin_data.get("BTCUSDT", {}).get("fear_greed", {})

    prompt = f"""You are a professional crypto quant trader and market analyst. Analyze all 5 coins and the overall market.

═══════════════════════════════════════════════════════
MARKET OVERVIEW
═══════════════════════════════════════════════════════
Fear & Greed: {fg.get('value','?')} — {fg.get('label','?')}
Bot trade summary: {trade_summary}
Recent trades:
{recent_trades_str}

═══════════════════════════════════════════════════════
COIN DATA
═══════════════════════════════════════════════════════
{''.join(coin_blocks)}

═══════════════════════════════════════════════════════
PATTERNS LEARNED SO FAR
═══════════════════════════════════════════════════════
{pattern_str}

═══════════════════════════════════════════════════════
YOUR TASKS
═══════════════════════════════════════════════════════
1. MARKET REGIME: What is the current macro regime? (risk-on/risk-off, trending/ranging, accumulation/distribution)

2. BEST SETUPS: Which of the 5 coins has the highest probability setup RIGHT NOW and why? Rate each coin 1-5.

3. LIQUIDITY INSIGHT: Where is smart money likely hunting stops? What levels should the bot target or avoid?

4. NEWS IMPACT: Any news that should change the bot's bias on specific coins?

5. PATTERN RECOGNITION: Have you spotted any patterns in the recent trades that explain wins/losses?

6. IMPROVEMENT: One specific concrete thing the bot should do differently based on what you see.

Respond with ONLY this JSON:
{{
  "market_regime": "<1-2 sentences>",
  "regime_type": "<trending_up|trending_down|ranging|uncertain>",
  "coin_ratings": {{"BTCUSDT": <1-5>, "ETHUSDT": <1-5>, "XRPUSDT": <1-5>, "DOGEUSDT": <1-5>, "SOLUSDT": <1-5>}},
  "best_setup": "<which coin and why, 2-3 sentences>",
  "liquidity_insight": "<where stops are resting, what levels matter, 2-3 sentences>",
  "news_impact": "<any news changing bias, or none>",
  "pattern_found": "<pattern in recent win/loss data, or still building>",
  "improvement": "<one specific concrete improvement>",
  "avoid_trades": ["<coin+direction to avoid right now>"],
  "watch_for": ["<setup to watch for in next few hours>"]
}}"""

    try:
        client_obj = get_claude_client()
        msg = client_obj.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        log(f"Claude market analysis error: {e}")
        return {}


def claude_trade_postmortem(trade: dict, candles: list, analysis: dict) -> str:
    """
    Generate a detailed post-mortem on a closed trade.
    Explains what happened and what to learn.
    """
    client = get_claude_client()
    if not client:
        return ""

    outcome  = trade.get("status", "?")
    sym      = trade.get("symbol", "")
    interval = trade.get("interval", "")
    direction = trade.get("direction", "")
    entry    = trade.get("entry", 0)
    tp       = trade.get("tp", 0)
    sl       = trade.get("sl", 0)
    close_px = trade.get("close_price", 0)
    roi      = trade.get("roi_pct", 0) or 0
    reason   = trade.get("reason", "")
    opened   = trade.get("opened_at", "")
    closed   = trade.get("closed_at", "")

    candle_summary = ""
    for c in candles[-20:]:
        chg = (c["close"] - c["open"]) / c["open"] * 100
        candle_summary += f"O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} C={c['close']:.4f} ({chg:+.2f}%)\n"

    prompt = f"""You are reviewing a closed crypto trade to extract lessons.

TRADE:
{sym} {interval} {direction}
Entry: {entry} | TP: {tp} | SL: {sl}
Opened: {opened} | Closed: {closed}
Close price: {close_px} | ROI: {roi:+.2f}%
Outcome: {outcome.upper()}
Signal reason: {reason}

CANDLES AROUND THE TRADE (20 bars):
{candle_summary}

CURRENT MARKET ANALYSIS:
Regime: {analysis.get('market_regime', 'unknown')}
Coin rating: {analysis.get('coin_ratings', {}).get(sym, '?')}/5

Write a concise post-mortem (3-4 sentences) covering:
1. Why the trade {'won' if outcome == 'win' else 'lost'}
2. Was this a good setup or should it have been skipped?
3. What specific condition to look for (or avoid) next time

Be direct and specific. No fluff."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return ""


# ── Main scan cycle ───────────────────────────────────────────────────────────
def run_scan_cycle(kb: dict) -> None:
    log("Starting scan cycle...")

    # ── Gather data for all coins ──────────────────────────────────────────
    coin_data: dict = {}
    fg = fetch_fear_greed()

    for sym in COINS:
        log(f"  Fetching {sym}...")
        candles_1h = fetch_candles(sym, "1h", 200)
        candles_4h = fetch_candles(sym, "4h", 100)
        candles_1d = fetch_candles(sym, "1d",  60)
        ob         = fetch_orderbook(sym)
        funding    = fetch_funding_rate(sym)
        news       = fetch_news(sym)
        entry_px   = candles_1h[-1]["close"] if candles_1h else 0
        clusters   = detect_liquidity_clusters(candles_1h, entry_px)

        coin_data[sym] = {
            "candles_1h":         candles_1h,
            "candles_4h":         candles_4h,
            "candles_1d":         candles_1d,
            "orderbook":          ob,
            "funding":            funding,
            "news":               news,
            "liquidity_clusters": clusters,
            "fear_greed":         fg,
        }
        time.sleep(1)

    # ── Get current trades from Render ─────────────────────────────────────
    trades = get_render_trades()
    open_trades  = [t for t in trades if t.get("status") == "open"]
    new_closed   = []

    # Track which trades we've already post-mortemed
    seen_ids = set(p.get("trade_id") for p in kb.get("postmortems", []))
    recently_closed = [
        t for t in trades
        if t.get("status") in ("win", "loss")
        and t.get("id") not in seen_ids
    ]

    # ── Claude deep market analysis ────────────────────────────────────────
    log("  Running Claude market analysis...")
    analysis = claude_market_analysis(coin_data, kb, trades)

    if analysis:
        log(f"  Regime: {analysis.get('regime_type')} | Best: {analysis.get('best_setup','?')[:60]}")

        # Store in knowledge base
        kb_add_memory(kb, "market_memory", {
            "regime":        analysis.get("regime_type"),
            "summary":       analysis.get("market_regime"),
            "coin_ratings":  analysis.get("coin_ratings"),
            "avoid_trades":  analysis.get("avoid_trades"),
            "watch_for":     analysis.get("watch_for"),
        })

        if analysis.get("pattern_found") and "building" not in analysis.get("pattern_found", "").lower():
            kb_add_memory(kb, "pattern_library", {
                "summary":     analysis.get("pattern_found"),
                "improvement": analysis.get("improvement"),
            })

        # Post intelligence report to Render
        insight = {
            "type":             "market_analysis",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "regime":           analysis.get("regime_type"),
            "market_summary":   analysis.get("market_regime"),
            "coin_ratings":     analysis.get("coin_ratings"),
            "best_setup":       analysis.get("best_setup"),
            "liquidity_insight": analysis.get("liquidity_insight"),
            "news_impact":      analysis.get("news_impact"),
            "avoid_trades":     analysis.get("avoid_trades"),
            "watch_for":        analysis.get("watch_for"),
            "improvement":      analysis.get("improvement"),
        }
        posted = post_agent_insight(insight)
        log(f"  Intelligence posted to Render: {posted}")

    # ── Post-mortems on newly closed trades ────────────────────────────────
    for trade in recently_closed[:3]:  # max 3 per cycle
        sym   = trade.get("symbol", "BTCUSDT")
        candles = coin_data.get(sym, {}).get("candles_1h", [])
        log(f"  Writing post-mortem: {trade.get('id')}")
        pm = claude_trade_postmortem(trade, candles, analysis)
        if pm:
            kb_add_memory(kb, "postmortems", {
                "trade_id":  trade.get("id"),
                "symbol":    sym,
                "direction": trade.get("direction"),
                "outcome":   trade.get("status"),
                "roi":       trade.get("roi_pct"),
                "postmortem": pm,
            })
            # Post back to Render
            post_agent_report({
                "type":      "postmortem",
                "trade_id":  trade.get("id"),
                "symbol":    sym,
                "outcome":   trade.get("status"),
                "roi":       trade.get("roi_pct"),
                "analysis":  pm,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            log(f"  Post-mortem for {trade.get('id')[:20]}:\n    {pm[:150]}")

    # ── Periodic improvement report (every 4 cycles = 1 hour) ─────────────
    scan_count = kb.get("scan_count", 0) + 1
    kb["scan_count"] = scan_count

    if scan_count % 4 == 0 and analysis.get("improvement"):
        improvement_report = {
            "type":        "improvement",
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "suggestion":  analysis.get("improvement"),
            "regime":      analysis.get("regime_type"),
            "watch_for":   analysis.get("watch_for"),
            "avoid":       analysis.get("avoid_trades"),
            "scan_count":  scan_count,
        }
        post_agent_report(improvement_report)

        # Discord notification
        if DISCORD_URL:
            msg = (
                f"**Regime:** {analysis.get('regime_type')}\n"
                f"**Best setup:** {analysis.get('best_setup','')}\n"
                f"**Avoid:** {', '.join(analysis.get('avoid_trades',[]))}\n"
                f"**Watch for:** {', '.join(analysis.get('watch_for',[]))}\n"
                f"**Improvement:** {analysis.get('improvement','')}"
            )
            discord_post(msg, "🧠 Agent Intelligence Report", color=0x00b4d8)

    kb["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_kb(kb)
    log(f"Scan cycle complete. KB has {len(kb.get('market_memory',[]))} memories, "
        f"{len(kb.get('postmortems',[]))} post-mortems.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log("═" * 60)
    log("  CRYPTO LOCAL INTELLIGENCE AGENT — starting up")
    log(f"  Render: {RENDER_URL}")
    log(f"  Claude: {'configured' if ANTHROPIC_KEY else 'NOT SET — set ANTHROPIC_API_KEY'}")
    log(f"  Discord: {'configured' if DISCORD_URL else 'not configured'}")
    log(f"  Scan interval: {SCAN_INTERVAL // 60} minutes")
    log("═" * 60)

    if not ANTHROPIC_KEY:
        log("ERROR: ANTHROPIC_API_KEY not set. Exiting.")
        return

    kb = load_kb()
    log(f"  Knowledge base loaded: {len(kb.get('market_memory',[]))} memories")

    while True:
        try:
            run_scan_cycle(kb)
        except Exception as e:
            log(f"ERROR in scan cycle: {e}")
            log(traceback.format_exc())
        log(f"Sleeping {SCAN_INTERVAL // 60} minutes until next scan...")
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()

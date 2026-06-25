"""
MACRO AGENT — runs every 30 minutes
Job: Read the big picture. News, sentiment, Fear & Greed, BTC dominance,
     funding rates. Determine overall market regime and coin biases.
Output: regime, coin biases, news highlights, avoid list
"""
from __future__ import annotations
import json, os, time
from datetime import datetime, timezone

import requests
import anthropic

from agents.state import set_state, get_state, add_report, add_knowledge, post_to_render

COINS = ["BTC", "ETH", "XRP", "DOGE", "SOL"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def fetch_fear_greed() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=3", timeout=8)
        data = r.json()["data"]
        return {
            "current":   {"value": int(data[0]["value"]), "label": data[0]["value_classification"]},
            "yesterday": {"value": int(data[1]["value"]), "label": data[1]["value_classification"]},
        }
    except Exception:
        return {}


def fetch_btc_dominance() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        return float(r.json()["data"]["market_cap_percentage"].get("btc", 0))
    except Exception:
        return 0.0


def fetch_funding_rates() -> dict:
    rates = {}
    for coin in COINS:
        sym = f"{coin}USDT"
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": sym, "limit": 1},
                timeout=8,
            )
            d = r.json()
            if d:
                rates[sym] = float(d[-1].get("fundingRate", 0))
        except Exception:
            rates[sym] = 0.0
        time.sleep(0.3)
    return rates


def fetch_news_all() -> dict:
    """Fetch news for all coins at once."""
    news_by_coin = {}
    for coin in COINS:
        try:
            r = requests.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={"auth_token": "free", "currencies": coin,
                        "public": "true", "filter": "hot"},
                timeout=8,
            )
            items = r.json().get("results", [])[:6]
            news_by_coin[coin] = [
                {
                    "title": item.get("title", ""),
                    "sentiment": (
                        "bullish" if item.get("votes", {}).get("positive", 0) >
                                     item.get("votes", {}).get("negative", 0)
                        else "bearish" if item.get("votes", {}).get("negative", 0) >
                                          item.get("votes", {}).get("positive", 0)
                        else "neutral"
                    ),
                }
                for item in items
            ]
            time.sleep(0.5)
        except Exception:
            news_by_coin[coin] = []
    return news_by_coin


def fetch_btc_price_context() -> dict:
    """Get BTC 1d candles to understand macro trend."""
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 14},
            timeout=10,
        )
        candles = r.json()
        closes = [float(c[4]) for c in candles]
        cur    = closes[-1]
        week   = closes[-7]
        month  = closes[0]
        return {
            "current":      cur,
            "change_7d":    (cur - week)  / week  * 100,
            "change_14d":   (cur - month) / month * 100,
            "above_ema20":  cur > sum(closes) / len(closes),
            "trend":        "up" if closes[-1] > closes[-3] > closes[-7] else
                            "down" if closes[-1] < closes[-3] < closes[-7] else "sideways",
        }
    except Exception:
        return {}


# ── On-chain data fetchers ────────────────────────────────────────────────────

def fetch_btc_onchain() -> dict:
    """
    BTC on-chain metrics from Blockchain.com public API.
    No API key required.
    """
    data = {}
    try:
        # 24h transaction volume
        r = requests.get("https://api.blockchain.info/stats", timeout=8)
        s = r.json()
        data["tx_count_24h"]        = s.get("n_tx", 0)
        data["btc_sent_24h"]        = round(s.get("total_btc_sent", 0) / 1e8, 2)
        data["avg_block_size"]      = round(s.get("blocks_size", 0) / 1024, 1)
        data["mempool_size"]        = s.get("mempool_size", 0)
        data["difficulty"]          = s.get("difficulty", 0)
        data["hash_rate_gh"]        = round(s.get("hash_rate", 0), 0)
    except Exception:
        pass

    try:
        # BTC held on exchanges (proxy: large wallet movements)
        r2 = requests.get(
            "https://api.blockchain.info/charts/estimated-transaction-volume-usd",
            params={"timespan": "2days", "format": "json", "sampled": "true"},
            timeout=8,
        )
        vals = r2.json().get("values", [])
        if vals:
            data["tx_volume_usd_24h"] = round(vals[-1].get("y", 0) / 1e9, 2)  # in billions
    except Exception:
        pass

    try:
        # Active addresses (network health)
        r3 = requests.get(
            "https://api.blockchain.info/charts/n-unique-addresses",
            params={"timespan": "2days", "format": "json", "sampled": "true"},
            timeout=8,
        )
        vals = r3.json().get("values", [])
        if len(vals) >= 2:
            data["active_addresses"] = int(vals[-1].get("y", 0))
            data["active_addr_change"] = round(
                (vals[-1]["y"] - vals[-2]["y"]) / max(vals[-2]["y"], 1) * 100, 1
            )
    except Exception:
        pass

    return data


def fetch_eth_onchain() -> dict:
    """ETH network stats from Etherscan (no key needed for basic stats)."""
    data = {}
    try:
        r = requests.get(
            "https://api.etherscan.io/api",
            params={"module": "stats", "action": "ethsupply"},
            timeout=8,
        )
        data["eth_supply"] = int(r.json().get("result", 0)) / 1e18
    except Exception:
        pass

    try:
        # ETH gas price (high gas = high network activity)
        r2 = requests.get(
            "https://api.etherscan.io/api",
            params={"module": "gastracker", "action": "gasoracle"},
            timeout=8,
        )
        gas = r2.json().get("result", {})
        data["gas_fast_gwei"]   = int(gas.get("FastGasPrice", 0))
        data["gas_safe_gwei"]   = int(gas.get("SafeGasPrice", 0))
        data["gas_suggest"]     = int(gas.get("suggestBaseFee", 0))
    except Exception:
        pass

    return data


def fetch_defi_tvl() -> dict:
    """
    Total Value Locked across DeFi from DefiLlama (free, no key).
    High TVL = more risk appetite. Sharp TVL drop = risk-off signal.
    """
    data = {}
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=8)
        history = r.json()
        if history and len(history) >= 3:
            latest  = history[-1]
            prev24h = history[-2]
            prev7d  = history[-8] if len(history) >= 8 else history[0]
            tvl     = latest.get("tvl", 0)
            data["total_tvl_bn"]  = round(tvl / 1e9, 2)
            data["tvl_change_24h"] = round((tvl - prev24h["tvl"]) / prev24h["tvl"] * 100, 2) if prev24h["tvl"] else 0
            data["tvl_change_7d"]  = round((tvl - prev7d["tvl"])  / prev7d["tvl"]  * 100, 2) if prev7d["tvl"] else 0
            data["tvl_signal"]     = ("risk_on" if data["tvl_change_24h"] > 1 else
                                      "risk_off" if data["tvl_change_24h"] < -2 else "neutral")
    except Exception:
        pass

    return data


def fetch_whale_alert() -> list:
    """
    Large transactions from Whale Alert API.
    Requires WHALE_ALERT_KEY env var. Returns empty list if not set.
    Free tier at whale-alert.io: 10 requests/min, last 1h of transactions.
    """
    key = os.environ.get("WHALE_ALERT_KEY", "")
    if not key:
        return []
    txs = []
    try:
        r = requests.get(
            "https://api.whale-alert.io/v1/transactions",
            params={
                "api_key":   key,
                "min_value": 1_000_000,   # $1M+ only
                "limit":     20,
            },
            timeout=8,
        )
        for tx in r.json().get("transactions", []):
            from_type = tx.get("from", {}).get("owner_type", "unknown")
            to_type   = tx.get("to",   {}).get("owner_type", "unknown")
            symbol    = tx.get("symbol", "").upper()
            amount    = tx.get("amount_usd", 0)
            # Flag exchange inflows/outflows (most significant for trading)
            if symbol in ("BTC", "ETH", "XRP", "DOGE", "SOL", "USDT", "USDC"):
                txs.append({
                    "symbol":    symbol,
                    "amount_usd": round(amount / 1e6, 1),  # in millions
                    "from":      from_type,
                    "to":        to_type,
                    "signal":    (
                        "BEARISH_INFLOW"  if to_type   == "exchange" else
                        "BULLISH_OUTFLOW" if from_type == "exchange" else
                        "TRANSFER"
                    ),
                })
    except Exception:
        pass
    return txs[:10]


def fetch_stablecoin_flows() -> dict:
    """
    USDT/USDC supply changes from DefiLlama.
    Rising stablecoin supply = dry powder to buy = bullish.
    Falling = money leaving crypto = bearish.
    """
    data = {}
    try:
        for stable in ["tether", "usd-coin"]:
            r = requests.get(f"https://api.llama.fi/stablecoin/{stable}", timeout=8)
            d = r.json()
            chains = d.get("chainCirculating", {})
            total  = sum(
                v.get("current", {}).get("peggedUSD", 0)
                for v in chains.values()
            )
            data[stable.replace("-", "_")] = round(total / 1e9, 2)  # billions
    except Exception:
        pass
    return data


def run() -> dict:
    """Execute macro analysis. Returns findings dict."""
    print("  Pulling market data...", flush=True)

    fg         = fetch_fear_greed()
    btc_dom    = fetch_btc_dominance()
    funding    = fetch_funding_rates()
    news       = fetch_news_all()
    btc_ctx    = fetch_btc_price_context()
    btc_chain  = fetch_btc_onchain()
    eth_chain  = fetch_eth_onchain()
    defi_tvl   = fetch_defi_tvl()
    whales     = fetch_whale_alert()
    stables    = fetch_stablecoin_flows()

    # ── Print everything we found so the terminal tells the full story ─────────
    fg_val   = fg.get("current", {}).get("value", "?")
    fg_label = fg.get("current", {}).get("label", "?")
    fg_yday  = fg.get("yesterday", {}).get("value", "?")
    print(f"  Fear & Greed: {fg_val} ({fg_label})  ←  was {fg_yday} yesterday", flush=True)

    btc_price = btc_ctx.get("current", 0)
    btc_7d    = btc_ctx.get("change_7d", 0)
    btc_14d   = btc_ctx.get("change_14d", 0)
    btc_trend = btc_ctx.get("trend", "?")
    print(f"  BTC: ${btc_price:,.0f}  |  7d: {btc_7d:+.1f}%  |  14d: {btc_14d:+.1f}%  |  Trend: {btc_trend}  |  Dominance: {btc_dom:.1f}%", flush=True)

    # Funding rates — tells us who is overleveraged
    print("  Funding rates (who's paying who):", flush=True)
    for sym, rate in funding.items():
        coin = sym.replace("USDT", "")
        who  = "longs paying → crowded long" if rate > 0 else "shorts paying → crowded short"
        print(f"    {coin}: {rate:+.4f}%  ({who})", flush=True)

    # On-chain
    if btc_chain:
        addrs   = btc_chain.get("active_addresses", 0)
        addr_ch = btc_chain.get("active_addr_change", 0)
        txvol   = btc_chain.get("tx_volume_usd_24h", 0)
        mempool = btc_chain.get("mempool_size", 0)
        print(f"  BTC on-chain: {addrs:,} active addrs ({addr_ch:+.1f}% vs yday)  |  ${txvol:.1f}B tx volume  |  {mempool:,} mempool", flush=True)

    if eth_chain:
        gas = eth_chain.get("gas_fast_gwei", 0)
        gas_note = "HIGH — heavy usage" if gas > 50 else "LOW — quiet" if gas < 10 else "normal"
        print(f"  ETH gas: {gas} gwei ({gas_note})", flush=True)

    if defi_tvl:
        tvl    = defi_tvl.get("total_tvl_bn", 0)
        tvl24  = defi_tvl.get("tvl_change_24h", 0)
        tvl7   = defi_tvl.get("tvl_change_7d", 0)
        signal = defi_tvl.get("tvl_signal", "neutral").upper()
        print(f"  DeFi TVL: ${tvl:.1f}B  |  24h: {tvl24:+.1f}%  |  7d: {tvl7:+.1f}%  →  {signal}", flush=True)

    if stables:
        usdt = stables.get("tether", 0)
        usdc = stables.get("usd_coin", 0)
        print(f"  Stablecoins: USDT ${usdt:.1f}B  |  USDC ${usdc:.1f}B", flush=True)

    if whales:
        print(f"  Whale moves (last 1h, $1M+):", flush=True)
        for tx in whales[:5]:
            arrow = "→ EXCHANGE (bearish inflow)" if tx["signal"] == "BEARISH_INFLOW" else \
                    "← FROM EXCHANGE (bullish outflow)" if tx["signal"] == "BULLISH_OUTFLOW" else "transfer"
            print(f"    ${tx['amount_usd']}M {tx['symbol']}  {tx['from']} {arrow}", flush=True)

    # News headlines
    print("  Latest news:", flush=True)
    for coin, items in news.items():
        if items:
            for item in items[:2]:
                sentiment_tag = "▲" if item["sentiment"] == "bullish" else "▼" if item["sentiment"] == "bearish" else "–"
                print(f"    {sentiment_tag} {coin}: {item['title'][:90]}", flush=True)

    # Format for Claude
    fg_str = f"Fear & Greed: {fg.get('current',{}).get('value','?')} ({fg.get('current',{}).get('label','?')}) | Yesterday: {fg.get('yesterday',{}).get('value','?')}"
    fund_str = "\n".join(f"  {sym}: {rate:+.4f}% ({'longs paying' if rate>0 else 'shorts paying'})"
                         for sym, rate in funding.items())
    news_str = ""
    for coin, items in news.items():
        if items:
            news_str += f"\n{coin}:\n"
            news_str += "\n".join(f"  [{i['sentiment']}] {i['title']}" for i in items[:4])

    btc_str = (f"BTC ${btc_ctx.get('current',0):,.0f} | "
               f"7d: {btc_ctx.get('change_7d',0):+.1f}% | "
               f"14d: {btc_ctx.get('change_14d',0):+.1f}% | "
               f"Trend: {btc_ctx.get('trend','?')}")

    # Format on-chain data
    onchain_str = ""
    if btc_chain:
        onchain_str += f"\nBTC ON-CHAIN:\n"
        onchain_str += f"  24h tx count: {btc_chain.get('tx_count_24h',0):,}\n"
        onchain_str += f"  24h tx volume: ${btc_chain.get('tx_volume_usd_24h',0):.1f}B\n"
        onchain_str += f"  Active addresses: {btc_chain.get('active_addresses',0):,} "
        onchain_str += f"({btc_chain.get('active_addr_change',0):+.1f}% vs yesterday)\n"
        onchain_str += f"  Mempool: {btc_chain.get('mempool_size',0):,} pending txs\n"

    if eth_chain:
        onchain_str += f"\nETH ON-CHAIN:\n"
        onchain_str += f"  Gas (fast): {eth_chain.get('gas_fast_gwei',0)} gwei "
        if eth_chain.get("gas_fast_gwei", 0) > 50:
            onchain_str += "(HIGH — heavy network usage)\n"
        elif eth_chain.get("gas_fast_gwei", 0) < 10:
            onchain_str += "(LOW — quiet network)\n"
        else:
            onchain_str += "(normal)\n"

    if defi_tvl:
        onchain_str += f"\nDeFi TVL: ${defi_tvl.get('total_tvl_bn',0):.1f}B "
        onchain_str += f"| 24h: {defi_tvl.get('tvl_change_24h',0):+.1f}% "
        onchain_str += f"| 7d: {defi_tvl.get('tvl_change_7d',0):+.1f}% "
        onchain_str += f"→ {defi_tvl.get('tvl_signal','neutral').upper()}\n"

    if stables:
        onchain_str += f"\nStablecoin Supply: "
        onchain_str += f"USDT ${stables.get('tether',0):.1f}B | "
        onchain_str += f"USDC ${stables.get('usd_coin',0):.1f}B\n"

    if whales:
        onchain_str += f"\nWHALE TRANSACTIONS (last 1h, $1M+):\n"
        for tx in whales[:6]:
            signal_emoji = "🔴" if tx["signal"] == "BEARISH_INFLOW" else "🟢" if tx["signal"] == "BULLISH_OUTFLOW" else "⚪"
            onchain_str += f"  {signal_emoji} ${tx['amount_usd']}M {tx['symbol']} {tx['from']} → {tx['to']} ({tx['signal']})\n"
    else:
        onchain_str += "\nWhale Alert: not configured (add WHALE_ALERT_KEY for whale tracking)\n"

    # Get prior regime for context
    prior = get_state("macro_regime", {})
    prior_str = f"Prior regime: {prior.get('regime_type','unknown')} ({prior.get('timestamp','')})" if prior else "First run."

    prompt = f"""You are the Macro Analyst on a crypto trading desk. Your job is to assess the big-picture market environment using ALL available data including on-chain signals.

{fg_str}
BTC Dominance: {btc_dom:.1f}%
{btc_str}

Funding Rates:
{fund_str}

ON-CHAIN DATA:
{onchain_str}

Recent News:
{news_str}

{prior_str}

Assess the macro environment and output ONLY this JSON:
{{
  "regime_type": "<bull_trending|bear_trending|ranging|uncertain>",
  "regime_strength": <1-10>,
  "macro_summary": "<2-3 sentences on overall market state>",
  "coin_bias": {{
    "BTC": "<long|short|neutral>",
    "ETH": "<long|short|neutral>",
    "XRP": "<long|short|neutral>",
    "DOGE": "<long|short|neutral>",
    "SOL": "<long|short|neutral>"
  }},
  "coin_bias_reasons": {{
    "BTC": "<one reason>",
    "ETH": "<one reason>",
    "XRP": "<one reason>",
    "DOGE": "<one reason>",
    "SOL": "<one reason>"
  }},
  "avoid_longs": ["<coin if applicable>"],
  "avoid_shorts": ["<coin if applicable>"],
  "key_news": "<most impactful news item, or none>",
  "risk_level": "<low|medium|high>",
  "trading_advice": "<one sentence — what the trading desk should do right now>"
}}"""

    client = _claude()
    result = {}
    if client:
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[MACRO AGENT] Claude error: {e}", flush=True)

    result["timestamp"]   = datetime.now(timezone.utc).isoformat()
    result["fear_greed"]  = fg
    result["btc_dom"]     = btc_dom
    result["funding"]     = funding
    result["btc_price"]   = btc_ctx
    result["btc_onchain"] = btc_chain
    result["eth_onchain"] = eth_chain
    result["defi_tvl"]    = defi_tvl
    result["whales"]      = whales
    result["stablecoins"] = stables

    # Store in shared state
    set_state("macro_regime", result)
    add_report("macro", "regime_analysis", result)
    add_knowledge("macro_snapshots", {
        "regime":   result.get("regime_type"),
        "strength": result.get("regime_strength"),
        "summary":  result.get("macro_summary"),
    })

    # Post to Render
    post_to_render("/api/agent/insight", {
        "type":      "macro_analysis",
        "agent":     "macro",
        "timestamp": result["timestamp"],
        "regime":    result.get("regime_type"),
        "strength":  result.get("regime_strength"),
        "summary":   result.get("macro_summary"),
        "coin_bias": result.get("coin_bias"),
        "risk_level": result.get("risk_level"),
        "advice":    result.get("trading_advice"),
    })

    # ── Print the AI's conclusions ─────────────────────────────────────────────
    regime   = result.get("regime_type", "?")
    strength = result.get("regime_strength", "?")
    risk     = result.get("risk_level", "?")
    summary  = result.get("macro_summary", "")
    advice   = result.get("trading_advice", "")
    key_news = result.get("key_news", "")

    print(f"", flush=True)
    print(f"  ── AI MACRO READ ──────────────────────────────────────", flush=True)
    print(f"  Regime: {regime.upper()} (strength {strength}/10)  |  Risk: {risk.upper()}", flush=True)
    if summary:
        print(f"  {summary}", flush=True)
    if advice:
        print(f"  Desk action: {advice}", flush=True)
    if key_news:
        print(f"  Key news: {key_news}", flush=True)

    # Coin biases
    biases  = result.get("coin_bias", {})
    reasons = result.get("coin_bias_reasons", {})
    if biases:
        print(f"  Coin biases:", flush=True)
        for coin in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
            bias   = biases.get(coin, "neutral").upper()
            reason = reasons.get(coin, "")
            arrow  = "▲" if bias == "LONG" else "▼" if bias == "SHORT" else "–"
            print(f"    {arrow} {coin}: {bias}  —  {reason[:70]}", flush=True)

    return result

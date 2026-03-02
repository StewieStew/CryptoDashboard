"""
Research Campaign — run this locally, NOT on Render.

Usage:
    python run_campaign.py

Runs a grid search for BTC, ETH, XRP, DOGE × 15m + 4H using maximum
available history. Prints a ranked results table and saves the best params
per symbol+interval to trades.db so the live scanner picks them up.

Then push to GitHub so Render restarts with the new params active.
"""

import json
import threading
import time
from datetime import datetime

import backtester
import learning

SYMBOLS   = backtester.CAMPAIGN_SYMBOLS    # BTC ETH XRP DOGE
INTERVALS = backtester.CAMPAIGN_INTERVALS  # 15m 4H
GRID      = backtester.CAMPAIGN_GRID       # ADX × threshold × min_rr (27 combos)


def _color(text, code):
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _color(t, 32)
def red(t):    return _color(t, 31)
def yellow(t): return _color(t, 33)
def bold(t):   return _color(t, 1)
def dim(t):    return _color(t, 2)


def run_job(sym, interval, results_store, lock):
    years = backtester.MAX_HISTORY_YEARS.get(interval, 2)
    label = f"{sym.replace('USDT','')} {interval.upper()}"
    print(f"  ⏳ {label} — fetching {years}yr of data…")
    try:
        all_results = backtester.grid_search(sym, interval, GRID, years=years)
        valid = [r for r in all_results if not r.get("error")]
        with lock:
            results_store[f"{sym}_{interval}"] = {
                "symbol":   sym,
                "interval": interval,
                "best":     valid[0] if valid else None,
                "top5":     valid[:5],
                "error":    None,
            }
        best = valid[0] if valid else {}
        wr   = f"{best.get('win_rate',0)*100:.1f}%" if best else "—"
        pf   = f"{best.get('profit_factor',0):.2f}" if best else "—"
        print(f"  ✓ {label} done — best win rate {wr}, profit factor {pf}")
    except Exception as e:
        with lock:
            results_store[f"{sym}_{interval}"] = {
                "symbol": sym, "interval": interval,
                "best": None, "top5": [], "error": str(e),
            }
        print(f"  ✗ {label} error: {e}")


def print_results(results):
    print()
    print(bold("═" * 80))
    print(bold("  RESEARCH CAMPAIGN RESULTS"))
    print(bold("═" * 80))
    print(f"  {'Symbol':<10} {'TF':<6} {'Trades':<8} {'Win %':<8} {'P.Factor':<10} {'Drawdown':<12} {'Best Params'}")
    print("  " + "─" * 76)

    for key in sorted(results):
        r = results[key]
        sym = r["symbol"].replace("USDT", "")
        iv  = r["interval"].upper()

        if r["error"]:
            print(f"  {sym:<10} {iv:<6} {red('ERROR: ' + r['error'])}")
            continue

        b = r["best"]
        if not b:
            print(f"  {sym:<10} {iv:<6} {dim('No results')}")
            continue

        wr   = b.get("win_rate", 0)
        pf   = b.get("profit_factor", 0)
        dd   = b.get("max_drawdown", 0)
        tot  = b.get("total_trades", 0)
        p    = b.get("params", {})

        wr_str = f"{wr*100:.1f}%"
        wr_col = green(wr_str) if wr >= 0.55 else (yellow(wr_str) if wr >= 0.40 else red(wr_str))
        pf_str = f"{pf:.2f}"
        pf_col = green(pf_str) if pf >= 1.5 else (yellow(pf_str) if pf >= 1.0 else red(pf_str))
        dd_str = f"{dd:.2f}R"

        params_str = (f"ADX:{p.get('adx_threshold','?')}  "
                      f"Thr:{p.get('score_threshold','?')}  "
                      f"R:R:{p.get('min_rr','?')}")

        print(f"  {bold(sym):<10} {iv:<6} {tot:<8} {wr_col:<8} {pf_col:<10} {dd_str:<12} {dim(params_str)}")

    print()


def prompt_apply(results):
    print(bold("─" * 80))
    print("  Apply best params to live scanner?")
    print("  These will be saved to trades.db. Push to GitHub to activate on Render.\n")

    applied = []
    for key in sorted(results):
        r = results[key]
        if not r["best"]:
            continue
        sym  = r["symbol"]
        iv   = r["interval"]
        p    = r["best"]["params"]
        wr   = r["best"].get("win_rate", 0)
        label = f"{sym.replace('USDT','')} {iv.upper()} (win rate {wr*100:.1f}%)"

        ans = input(f"  Apply {label}? [y/N]: ").strip().lower()
        if ans == "y":
            learning.save_symbol_params(sym, iv, p)
            applied.append(label)
            print(f"  {green('✓')} Saved {label}")

    print()
    if applied:
        print(green(f"  {len(applied)} param set(s) saved to trades.db."))
        print("  Push to GitHub so Render restarts with the new params active:")
        print(dim("    git add trades.db && git commit -m 'Apply campaign params' && git push"))
    else:
        print(dim("  Nothing applied."))
    print()


def main():
    print()
    print(bold("🔬 Research Campaign"))
    print(f"   Symbols:   {', '.join(s.replace('USDT','') for s in SYMBOLS)}")
    print(f"   Intervals: {', '.join(INTERVALS)}")
    print(f"   Grid:      {sum(len(v) for v in GRID.values())} values → "
          f"{len(list(__import__('itertools').product(*GRID.values())))} combinations per job")
    print(f"   Jobs:      {len(SYMBOLS) * len(INTERVALS)} (running in parallel)")
    print(f"   Started:   {datetime.now().strftime('%H:%M:%S')}")
    print()

    results = {}
    lock    = threading.Lock()
    threads = []

    for sym in SYMBOLS:
        for interval in INTERVALS:
            t = threading.Thread(
                target=run_job,
                args=(sym, interval, results, lock),
                daemon=True,
            )
            t.start()
            threads.append(t)
            time.sleep(0.5)   # stagger starts slightly to avoid API burst

    for t in threads:
        t.join()

    print(f"\n   Finished: {datetime.now().strftime('%H:%M:%S')}")
    print_results(results)
    prompt_apply(results)


if __name__ == "__main__":
    main()

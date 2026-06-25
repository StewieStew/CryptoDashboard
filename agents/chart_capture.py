"""
TRADINGVIEW CHART CAPTURE
Uses Playwright (headless Chromium) to screenshot real TradingView charts.
Captures 4H, 1H, and 15M for each coin and stitches them into one tall image.

Requires: pip install playwright && python3 -m playwright install chromium
Falls back silently if Playwright is not installed.
"""
from __future__ import annotations
import base64, time
from io import BytesIO

# TradingView widget URL — dark theme, candlestick style, volume + EMA overlays
# Works without login. Uses Binance data (same price as Binance US for charting).
_TV_URL = (
    "https://www.tradingview.com/widgetembed/"
    "?symbol=BINANCE:{symbol}"
    "&interval={interval}"
    "&theme=dark"
    "&style=1"               # 1 = candlesticks
    "&locale=en"
    "&timezone=Etc%2FUTC"
    "&hidesidetoolbar=0"
    "&hidetoptoolbar=0"
    "&allow_symbol_change=0"
    "&save_image=0"
    "&studies=MAExp%40tv-basicstudies%1FMAExp%40tv-basicstudies%1FVolume%40tv-basicstudies%1FRSI%40tv-basicstudies"
    "&overrides=%7B%22mainSeriesProperties.candleStyle.upColor%22%3A%22%2326a69a%22%2C%22mainSeriesProperties.candleStyle.downColor%22%3A%22%23ef5350%22%7D"
)

# interval strings for TradingView
_INTERVALS = {
    "4h":  "240",
    "1h":  "60",
    "15m": "15",
    "1d":  "D",
}

# viewport per chart panel
_W = 1800
_H = 560


def _screenshot_url(page, url: str, wait_ms: int = 5000) -> bytes | None:
    """Navigate to URL and return screenshot bytes."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30_000)
        page.wait_for_timeout(wait_ms)   # let the chart fully render
        return page.screenshot()
    except Exception as e:
        print(f"  [TV] screenshot failed: {e}", flush=True)
        return None


def capture_coin(symbol: str,
                 timeframes: list[str] | None = None) -> str | None:
    """
    Capture TradingView charts for one coin across multiple timeframes.
    Stacks them vertically into a single PNG and returns as base64.

    symbol     — e.g. "BTCUSDT"
    timeframes — list from ["4h", "1h", "15m", "1d"]  (default: 4h, 1h, 15m)
    """
    if timeframes is None:
        timeframes = ["4h", "1h", "15m"]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None   # Playwright not installed — caller falls back to matplotlib

    try:
        from PIL import Image
    except ImportError:
        # PIL not available — return just the first screenshot
        _pil = False
    else:
        _pil = True

    screenshots: list[bytes] = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": _W, "height": _H})

            for tf in timeframes:
                interval = _INTERVALS.get(tf, "60")
                url = _TV_URL.format(symbol=symbol, interval=interval)
                page = context.new_page()
                shot = _screenshot_url(page, url, wait_ms=5000)
                page.close()
                if shot:
                    screenshots.append(shot)
                    print(f"  [TV] {symbol} {tf} ✓", flush=True)
                else:
                    print(f"  [TV] {symbol} {tf} ✗ (skipped)", flush=True)

            browser.close()

    except Exception as e:
        print(f"  [TV] capture failed for {symbol}: {e}", flush=True)
        return None

    if not screenshots:
        return None

    if not _pil or len(screenshots) == 1:
        return base64.b64encode(screenshots[0]).decode()

    # Stitch screenshots vertically with PIL
    try:
        images = [Image.open(BytesIO(s)) for s in screenshots]
        total_h = sum(img.height for img in images)
        canvas = Image.new("RGB", (_W, total_h), color=(13, 13, 26))
        y = 0
        for img in images:
            canvas.paste(img, (0, y))
            y += img.height
        buf = BytesIO()
        canvas.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print(f"  [TV] stitch failed: {e}", flush=True)
        return base64.b64encode(screenshots[0]).decode()


def is_available() -> bool:
    """Return True if Playwright is installed and ready."""
    try:
        from playwright.sync_api import sync_playwright  # noqa
        return True
    except ImportError:
        return False

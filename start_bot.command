#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CryptoBot — Start / Restart
#  Double-click to start the bot. Also works as a manual restart.
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")"
BOT_DIR="$(pwd)"

# Load .env
if [ -f "$BOT_DIR/.env" ]; then
  export $(grep -v '^#' "$BOT_DIR/.env" | xargs)
fi

export DB_PATH="${DB_PATH:-$BOT_DIR/trades.db}"
export PORT="${PORT:-5000}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CryptoBot — Starting                   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Dashboard → http://localhost:$PORT"
echo "  DB        → $DB_PATH"
echo "  Log       → $BOT_DIR/bot.log"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

source "$BOT_DIR/venv/bin/activate"
python "$BOT_DIR/app.py" 2>&1 | tee -a "$BOT_DIR/bot.log"

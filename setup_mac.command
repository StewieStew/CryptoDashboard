#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CryptoBot — Mac Mini one-time setup
#  Double-click this ONCE to install everything.
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")"
BOT_DIR="$(pwd)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CryptoBot — Mac Mini Setup             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check Python 3 ────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Install it from https://python.org then re-run."
  read -p "Press Enter to exit..."; exit 1
fi
PYTHON=$(command -v python3)
echo "✅  Python: $($PYTHON --version)"

# ── 2. Create virtualenv ─────────────────────────────────────
if [ ! -d "$BOT_DIR/venv" ]; then
  echo "→  Creating virtual environment..."
  $PYTHON -m venv "$BOT_DIR/venv"
fi
source "$BOT_DIR/venv/bin/activate"

# ── 3. Install dependencies ──────────────────────────────────
echo "→  Installing Python packages..."
pip install --upgrade pip -q
pip install -r "$BOT_DIR/requirements.txt" -q
echo "✅  Packages installed"

# ── 4. Create .env if missing ────────────────────────────────
if [ ! -f "$BOT_DIR/.env" ]; then
  cat > "$BOT_DIR/.env" <<EOF
# CryptoBot local environment — fill in your keys
ANTHROPIC_API_KEY=
DISCORD_WEBHOOK_URL=
DB_PATH=$BOT_DIR/trades.db
PORT=5000
EOF
  echo "✅  Created .env — open it and add your API keys before starting the bot"
  open "$BOT_DIR/.env"
else
  echo "✅  .env already exists"
fi

# ── 5. Migrate DB from Render (optional) ─────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo "  OPTIONAL: Migrate your trade history from Render"
echo "  If you have a trades.db backup from Render, copy it to:"
echo "  $BOT_DIR/trades.db"
echo "  (skip this if starting fresh)"
echo "──────────────────────────────────────────────"
echo ""

# ── 6. Install LaunchAgent for auto-start on login ───────────
PLIST_SRC="$BOT_DIR/com.cryptobot.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.cryptobot.plist"

# Write the plist with the correct paths baked in
cat > "$PLIST_SRC" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.cryptobot</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BOT_DIR/venv/bin/python</string>
    <string>$BOT_DIR/app.py</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$BOT_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>DB_PATH</key>
    <string>$BOT_DIR/trades.db</string>
    <key>PORT</key>
    <string>5000</string>
  </dict>
  <key>KeepAlive</key>
  <true/>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$BOT_DIR/bot.log</string>
  <key>StandardErrorPath</key>
  <string>$BOT_DIR/bot.log</string>
  <key>ThrottleInterval</key>
  <integer>10</integer>
</dict>
</plist>
EOF

cp "$PLIST_SRC" "$PLIST_DEST"
launchctl unload "$PLIST_DEST" 2>/dev/null
launchctl load "$PLIST_DEST"
echo "✅  LaunchAgent installed — bot will start automatically on login"

# ── 7. Install cloudflared for remote dashboard ──────────────
echo ""
echo "→  Checking for cloudflared (remote dashboard tunnel)..."
if ! command -v cloudflared &>/dev/null; then
  if command -v brew &>/dev/null; then
    echo "→  Installing cloudflared via Homebrew..."
    brew install cloudflared
  else
    echo "⚠️   Homebrew not found. Install cloudflared manually:"
    echo "    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  fi
else
  echo "✅  cloudflared already installed"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete!                        ║"
echo "║                                          ║"
echo "║   Next steps:                            ║"
echo "║   1. Add your API keys to .env           ║"
echo "║   2. Double-click start_bot.command      ║"
echo "║   3. Open http://localhost:5000           ║"
echo "║   4. Run tunnel.command for remote URL   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
read -p "Press Enter to close..."

#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  CryptoBot — Cloudflare Tunnel
#  Double-click to get a public URL for your dashboard.
#  Keep this window open while you want remote access.
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi
PORT="${PORT:-5000}"

if ! command -v cloudflared &>/dev/null; then
  echo "❌  cloudflared not installed."
  echo "    Run setup_mac.command first, or install manually:"
  echo "    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  read -p "Press Enter to exit..."; exit 1
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CryptoBot — Remote Access Tunnel       ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Starting tunnel to http://localhost:$PORT ..."
echo "  Your public URL will appear below."
echo "  Keep this window open for remote access."
echo ""

cloudflared tunnel --url "http://localhost:$PORT"

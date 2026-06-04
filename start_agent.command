#!/bin/bash
# ── Start the Trading Desk Agent System ──────────────────────────────────────
cd ~/Desktop/CryptoDashboard

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "Add it to ~/.zshrc:"
    echo "  echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.zshrc && source ~/.zshrc"
    echo ""
    read -p "Or enter it now: " KEY
    export ANTHROPIC_API_KEY=$KEY
fi

export RENDER_URL="https://cryptodashboard-nuf5.onrender.com"

pip3 install anthropic requests numpy --quiet --break-system-packages 2>/dev/null

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  CRYPTO TRADING DESK — MULTI-AGENT SYSTEM"
echo "  4 agents running. Press Ctrl+C to stop."
echo "═══════════════════════════════════════════════════════"
echo ""

python3 orchestrator.py

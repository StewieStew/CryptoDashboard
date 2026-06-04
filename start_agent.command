#!/bin/bash
# ── Start the local intelligence agent ──────────────────────────────────────
# Run this once to start the agent. It will keep running until you close
# the terminal or stop it with Ctrl+C.
#
# For permanent auto-start on boot, run: install_agent_autostart.command

cd ~/Desktop/CryptoDashboard

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "ERROR: ANTHROPIC_API_KEY not set."
    echo "Set it by running:"
    echo "  export ANTHROPIC_API_KEY=your-key-here"
    echo ""
    echo "Or add it to ~/.zshrc:"
    echo "  echo 'export ANTHROPIC_API_KEY=your-key-here' >> ~/.zshrc"
    echo ""
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  CRYPTO LOCAL INTELLIGENCE AGENT"
echo "  Scanning every 15 minutes. Press Ctrl+C to stop."
echo "═══════════════════════════════════════════════════════"
echo ""

export RENDER_URL="https://cryptodashboard-nuf5.onrender.com"

pip3 install anthropic requests --quiet --break-system-packages 2>/dev/null

python3 local_agent.py

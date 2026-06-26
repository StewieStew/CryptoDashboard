#!/bin/bash
# Push analyst changes to GitHub and sync ~/CryptoDashboard/ for LaunchAgents
cd "$(dirname "$0")"

echo "=== Committing analyst + executor changes ==="
git add agents/analyst_agent.py app.py orchestrator.py
git status
git commit -m "Add daily minimum trade rule + market entry type + fix LaunchAgent .env handling

- _trades_today(): count trades opened today via Render API
- After 18:00 UTC with zero trades: append DAILY MINIMUM NOT MET warning to analyst prompt,
  force entry_type=market so setup executes immediately instead of waiting for pullback
- app.py executor: entry_type=market bypasses limit-order wait logic, executes at live price
- orchestrator.py: graceful fallback when .env in Desktop is blocked by macOS TCC

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

echo ""
echo "=== Pushing to GitHub (Render will auto-deploy) ==="
git push origin main

echo ""
echo "=== Syncing to ~/CryptoDashboard/ for LaunchAgents ==="
cp orchestrator.py ~/CryptoDashboard/orchestrator.py
cp local_agent.py ~/CryptoDashboard/local_agent.py
rsync -a --delete agents/ ~/CryptoDashboard/agents/
echo "Sync complete."

echo ""
echo "=== Reloading LaunchAgents ==="
launchctl unload ~/Library/LaunchAgents/com.cryptobot.tradingdesk.plist 2>/dev/null
launchctl load  ~/Library/LaunchAgents/com.cryptobot.tradingdesk.plist
launchctl unload ~/Library/LaunchAgents/com.cryptobot.agent.plist 2>/dev/null
launchctl load  ~/Library/LaunchAgents/com.cryptobot.agent.plist
echo "LaunchAgents reloaded."

echo ""
echo "=== Done. Render deploy in ~2 min, agents live now. ==="

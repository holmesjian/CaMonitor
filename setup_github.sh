#!/bin/bash
# CaMonitor — GitHub setup and push script
# Run once from ~/Documents/camonitor/

REPO_URL="https://github.com/YOUR_USERNAME/camonitor.git"

echo "=== CaMonitor GitHub Setup ==="

if [ ! -d .git ]; then
    git init
    echo "Git repo initialised"
fi

# ── .gitignore ──────────────────────────────────────────────────
cat > .gitignore << 'GITIGNORE'
# ── Credentials — NEVER commit ─────────────────────
scripts/email_config.yaml

# ── YOLO model weights — auto-downloaded by ultralytics on first run ──
*.pt
*.onnx

# ── Alert images — may contain personal/child images ────────────
data/alerts/

# ── Logs — exclude raw data, keep folder structure ──────────────
logs/*.csv
logs/monitor.log
!logs/.gitkeep

# ── Python ──────────────────────────────────────────────────────
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/

# ── OS ──────────────────────────────────────────────────────────
.DS_Store
Thumbs.db
GITIGNORE

echo ".gitignore written"

# Create placeholder files to preserve empty dirs
mkdir -p logs data/alerts docs/screenshots
touch logs/.gitkeep
touch data/alerts/.gitkeep

# Stage all
git add .
git status

echo ""
echo "=== Files staged. Review above, then run: ==="
echo ""
echo "  1. Create repo at: https://github.com/new"
echo "     Name: camonitor"
echo "     Visibility: Public"
echo "     Do NOT initialise with README (no checkbox)"
echo ""
echo "  2. Then commit and push:"
echo "     git commit -m 'Initial commit — CaMonitor edge AI nursery guard'"
echo "     git remote add origin $REPO_URL"
echo "     git branch -M main"
echo "     git push -u origin main"
echo ""
echo "  3. Replace YOUR_USERNAME in the remote URL above with your GitHub username."

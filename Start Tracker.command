#!/bin/zsh
# Double-click to start the drone tracker dashboard. Keep this window open while you use it.
cd "$(dirname "$0")" || { echo "Project folder not found."; read; exit 1; }
pkill -f "dashboard/app.py" 2>/dev/null; sleep 1
(sleep 5; open http://localhost:8000) &
echo "Drone tracker dashboard: http://localhost:8000"
echo "Keep this window open. Close it or press Ctrl-C to stop."
exec .venv/bin/python dashboard/app.py

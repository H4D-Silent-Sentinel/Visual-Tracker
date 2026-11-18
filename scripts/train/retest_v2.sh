#!/bin/zsh
# Waits for the bird/kite retraining to finish, then re-runs Test 1, Test 2 and the held-out
# drone photo test on the new model (v2), and restarts the dashboard so v2 becomes the default.
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "name=dut_neg_n960" > /dev/null; do sleep 30; done
W=runs/train/dut_neg_n960/weights/best.pt
echo "training finished $(date)"
.venv/bin/python -u scripts/evaluate/test_false_alarms.py --model $W --out runs/false_alarms_v2
.venv/bin/python -u scripts/evaluate/test_tracking.py --model $W --out runs/tracking_test_v2 --no-mp4
.venv/bin/yolo detect val model=$W data=$HOME/yolo-datasets/dut-anti-uav/dut.yaml split=test imgsz=960 device=mps project=$PWD/runs/val name=dut_neg_n960_test exist_ok=True
pkill -f "dashboard/app.py"; sleep 2
nohup .venv/bin/python -u dashboard/app.py > runs/dashboard.log 2>&1 &
echo "ALL DONE $(date)"

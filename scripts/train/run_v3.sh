#!/bin/zsh
# Queued after scripts/train/retest_v2.sh: (1) test the "hold" tracker settings with v2, (2) train v3 with
# cluttered-background frames, (3) test v3. Tracking scores for v3 only count on HELD_OUT videos.
cd "$(dirname "$0")/../.." || exit 1
while pgrep -f "retest_v2.sh" > /dev/null || pgrep -f "name=dut_neg_n960" > /dev/null; do sleep 30; done
V2=runs/train/dut_neg_n960/weights/best.pt
echo "== v2 + hold tracker $(date)"
.venv/bin/python -u scripts/evaluate/test_tracking.py --model $V2 --out runs/tracking_test_v2_hold --no-mp4 --tracker bytetrack_hold.yaml --det-conf 0.08
echo "== train v3 $(date)"
.venv/bin/yolo detect train model=$V2 data=$HOME/yolo-datasets/dut-anti-uav/dut_v3.yaml imgsz=960 epochs=10 close_mosaic=3 warmup_epochs=1 batch=16 device=mps workers=8 patience=5 project=$PWD/runs/train name=dut_v3_n960 exist_ok=True > runs/train_dut_v3_n960.log 2>&1
V3=runs/train/dut_v3_n960/weights/best.pt
echo "== test v3 $(date)"
.venv/bin/python -u scripts/evaluate/test_false_alarms.py --model $V3 --out runs/false_alarms_v3
.venv/bin/python -u scripts/evaluate/test_tracking.py --model $V3 --out runs/tracking_test_v3 --no-mp4
.venv/bin/python -u scripts/evaluate/test_tracking.py --model $V3 --out runs/tracking_test_v3_hold --no-mp4 --tracker bytetrack_hold.yaml --det-conf 0.08
.venv/bin/yolo detect val model=$V3 data=$HOME/yolo-datasets/dut-anti-uav/dut.yaml split=test imgsz=960 device=mps project=$PWD/runs/val name=dut_v3_n960_test exist_ok=True
echo "ALL DONE $(date)"

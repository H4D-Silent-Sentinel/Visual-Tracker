#!/bin/zsh
# v4 = v3 trained further with drone types (ratio1 REAL: propeller + fixed-wing drones; planes, helicopters,
# birds as "no drone") and motion-blurred copies. Steps: wait for the download, build data, score v3 on the
# drone-type test (baseline), train v4, run every test on v4.
cd "$(dirname "$0")/../.." || exit 1
until grep -q "DOWNLOAD DONE" /tmp/dl_ratio1.log 2>/dev/null; do sleep 30; done
echo "== prepare data $(date)"
.venv/bin/python -u scripts/data/prepare_ratio1.py || exit 1
.venv/bin/python -u scripts/data/prepare_blur.py || exit 1
D=$HOME/yolo-datasets/dut-anti-uav
cat > $D/dut_v4.yaml <<EOF
path: $D/yolo
train: [images/train, images/train_neg, images/train_trk, images/train_r1, images/train_blur]
val: [images/val, images/val_neg, images/val_r1]
test: images/test
names:
  0: drone
EOF
V3=runs/train/dut_v3_n960/weights/best.pt
echo "== v3 baseline on drone types $(date)"
.venv/bin/python -u scripts/evaluate/test_drone_types.py --model $V3 --out runs/drone_types_v3.json
echo "== train v4 $(date)"
.venv/bin/yolo detect train model=$V3 data=$D/dut_v4.yaml imgsz=960 epochs=10 close_mosaic=3 warmup_epochs=1 batch=16 device=mps workers=8 patience=5 project=$PWD/runs/train name=dut_v4_n960 exist_ok=True > runs/train_dut_v4_n960.log 2>&1
V4=runs/train/dut_v4_n960/weights/best.pt
[ -f $V4 ] || { echo "TRAINING FAILED"; exit 1; }
echo "== test v4 $(date)"
.venv/bin/python -u scripts/evaluate/test_drone_types.py --model $V4 --out runs/drone_types_v4.json
.venv/bin/python -u scripts/evaluate/test_false_alarms.py --model $V4 --out runs/false_alarms_v4
.venv/bin/python -u scripts/evaluate/test_tracking.py --model $V4 --out runs/tracking_test_v4 --no-mp4
.venv/bin/yolo detect val model=$V4 data=$D/dut.yaml split=test imgsz=960 device=mps project=$PWD/runs/val name=dut_v4_n960_test exist_ok=True
echo "ALL DONE $(date)"

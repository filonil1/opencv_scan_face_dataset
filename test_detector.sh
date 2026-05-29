bash test_detector.sh#!/usr/bin/env bash
# Quick diagnostics — run from project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

echo "=== 1. Python + mediapipe version ==="
$PYTHON -c "import mediapipe as mp, cv2; print('mediapipe', mp.__version__); print('opencv-python', cv2.__version__)"

echo ""
echo "=== 2. Capture one webcam frame and test detection ==="
$PYTHON - << 'PYEOF'
import cv2, mediapipe as mp, json, sys

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open webcam"); sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Warm up — skip first few frames (camera exposure settling)
for _ in range(10):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: blank frame"); sys.exit(1)

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")
cv2.imwrite("/tmp/fdet_test_frame.jpg", frame)
print("Saved test frame → /tmp/fdet_test_frame.jpg")

# Try both models
for model_sel in (0, 1):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_detection.FaceDetection(
            model_selection=model_sel,
            min_detection_confidence=0.3) as det:
        res = det.process(rgb)
    n = len(res.detections) if res.detections else 0
    print(f"model_selection={model_sel}  min_conf=0.3  → {n} face(s) detected")
    if res.detections:
        for d in res.detections:
            print(f"  score={d.score[0]:.3f}  bbox={d.location_data.relative_bounding_box}")
PYEOF

echo ""
echo "=== 3. Check saved test frame ==="
ls -lh /tmp/fdet_test_frame.jpg 2>/dev/null || echo "frame not saved"

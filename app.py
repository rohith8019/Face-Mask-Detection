"""
Flask Web Application for YOLOv8 Face Mask Detection
Provides an interactive modern web UI supporting both single image and batch folder inference.
"""

import base64
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "models" / "best.pt"

if not WEIGHTS_PATH.exists():
    WEIGHTS_PATH = ROOT / "runs" / "face_mask" / "baseline_416_10ep" / "weights" / "best.pt"

print(f"[INFO] Initializing YOLOv8 Face Mask Model from: {WEIGHTS_PATH}")
model = YOLO(str(WEIGHTS_PATH))

app = Flask(__name__, template_folder="templates", static_folder="static")

CLASS_COLORS = {
    0: (0, 200, 0),      # Wearing Mask: Green
    1: (0, 215, 255),    # Wore Mask Incorrectly: Gold / Amber
    2: (0, 0, 230)       # No Mask: Crimson Red
}

CLASS_NAMES = {
    0: "Wearing Mask",
    1: "Wore Mask Incorrectly",
    2: "No Mask"
}


def draw_styled_box(frame, box, label, conf, color):
    """Draw a clean bounding box with corner accents and tag."""
    x1, y1, x2, y2 = map(int, box)

    # Box outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Corner brackets
    corner_len = min(15, max(4, (x2 - x1) // 5), max(4, (y2 - y1) // 5))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

    # Label text & tag banner
    text = f"{label} {conf:.1%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

    tag_y1 = max(0, y1 - th - 8)
    tag_y2 = y1
    tag_x2 = min(frame.shape[1], x1 + tw + 10)

    cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
    text_color = (0, 0, 0) if color == (0, 215, 255) else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (x1 + 5, tag_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        text_color,
        1,
        cv2.LINE_AA
    )


def process_image(img_bgr, conf=0.40, iou=0.45, imgsz=416):
    """Run detection and return annotated image with summary."""
    t0 = time.time()
    results = model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False)
    latency_ms = (time.time() - t0) * 1000

    annotated = img_bgr.copy()
    counts = {name: 0 for name in CLASS_NAMES.values()}
    detections = []

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()

                label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                counts[label] = counts.get(label, 0) + 1

                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "box": [round(float(v), 1) for v in xyxy]
                })

                draw_styled_box(annotated, xyxy, label, confidence, color)

    # Top summary banner
    h, w = annotated.shape[:2]
    banner_h = 32
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 24, 33)
    summary_text = f"Wearing Mask: {counts['Wearing Mask']} | Incorrect: {counts['Wore Mask Incorrectly']} | No Mask: {counts['No Mask']} | {latency_ms:.1f}ms"
    cv2.putText(banner, summary_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
    annotated = np.vstack([banner, annotated])

    # Encode to base64
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_str}"

    return {
        "annotated_data_uri": data_uri,
        "total_faces": sum(counts.values()),
        "counts": counts,
        "detections": detections,
        "latency_ms": round(latency_ms, 1)
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "No files provided in request"}), 400

    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "Empty file list"}), 400

    try:
        conf = float(request.form.get("conf", 0.40))
        iou = float(request.form.get("iou", 0.45))
    except ValueError:
        conf = 0.40
        iou = 0.45

    results_list = []
    total_faces_agg = 0
    wearing_mask_agg = 0
    incorrect_mask_agg = 0
    no_mask_agg = 0
    total_latency_ms = 0.0

    for file_obj in files:
        filename = file_obj.filename
        file_bytes = np.frombuffer(file_obj.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            continue

        res = process_image(img, conf=conf, iou=iou)
        total_faces_agg += res["total_faces"]
        wearing_mask_agg += res["counts"]["Wearing Mask"]
        incorrect_mask_agg += res["counts"]["Wore Mask Incorrectly"]
        no_mask_agg += res["counts"]["No Mask"]
        total_latency_ms += res["latency_ms"]

        results_list.append({
            "filename": filename,
            "image_base64": res["annotated_data_uri"],
            "total_faces": res["total_faces"],
            "counts": res["counts"],
            "detections": res["detections"],
            "latency_ms": res["latency_ms"]
        })

    avg_latency = (total_latency_ms / len(results_list)) if results_list else 0.0

    return jsonify({
        "results": results_list,
        "aggregate": {
            "total_images": len(results_list),
            "total_faces": total_faces_agg,
            "wearing_mask": wearing_mask_agg,
            "incorrect_mask": incorrect_mask_agg,
            "no_mask": no_mask_agg,
            "avg_latency_ms": round(avg_latency, 1)
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🚀 Face Mask Detection Studio UI Starting...")
    print("👉 Open your browser at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

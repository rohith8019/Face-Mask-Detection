"""
YOLOv8 Face Mask Detection — Inference Pipeline
Supports single image and batch directory inference on CPU with custom bounding boxes.

Color tags:
  - Green   : Class 0 (Wearing Mask)
  - Yellow  : Class 1 (Wore Mask Incorrectly)
  - Red     : Class 2 (No Mask)
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Class colors in BGR
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


def find_best_weights():
    """Locate the best available trained model checkpoint."""
    candidates = [
        ROOT / "models" / "best.pt",
        ROOT / "runs" / "face_mask" / "baseline_416_10ep" / "weights" / "best.pt",
        ROOT / "runs" / "face_mask" / "quick_416" / "weights" / "best.pt",
        ROOT / "yolov8n.pt"
    ]
    for c in candidates:
        if c.exists():
            return c
    return ROOT / "yolov8n.pt"


def draw_styled_box(frame, box, label, conf, color):
    """Draw a clean, aesthetic bounding box with contrasting tag banner."""
    x1, y1, x2, y2 = map(int, box)

    # Bounding box border
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Subtle corner brackets
    corner_len = min(15, max(4, (x2 - x1) // 5), max(4, (y2 - y1) // 5))
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 - corner_len), color, 3)
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


def predict_image(model, image_path, output_path=None, conf=0.40, iou=0.45, imgsz=416):
    """
    Run mask detection on an uploaded image.
    Returns: (annotated_image, detection_summary_dict)
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image file from: {image_path}. Ensure it is a valid JPG/PNG file.")

    t0 = time.time()
    results = model.predict(source=img, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False)
    inference_ms = (time.time() - t0) * 1000

    annotated = img.copy()
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
                    "confidence": round(confidence, 3),
                    "box": [round(float(v), 1) for v in xyxy]
                })

                draw_styled_box(annotated, xyxy, label, confidence, color)

    # Banner overlay
    h, w = annotated.shape[:2]
    banner_h = 32
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    summary_text = f"Wearing Mask: {counts['Wearing Mask']} | Incorrect: {counts['Wore Mask Incorrectly']} | No Mask: {counts['No Mask']} | Latency: {inference_ms:.1f}ms"
    cv2.putText(banner, summary_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
    annotated = np.vstack([banner, annotated])

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)

    summary = {
        "image": str(img_path),
        "output": str(output_path) if output_path else None,
        "total_faces": sum(counts.values()),
        "counts": counts,
        "detections": detections,
        "latency_ms": round(inference_ms, 1)
    }
    return annotated, summary


def predict_batch(model, input_dir, output_dir, conf=0.40, iou=0.45, imgsz=416, limit=None):
    """Run batch inference over a folder of images."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in valid_exts]

    if limit and limit > 0:
        images = images[:limit]

    print(f"\nProcessing {len(images)} images from: {in_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Confidence threshold: {conf} | NMS IoU threshold: {iou} | Image resolution: {imgsz}")
    print("-" * 60)

    results = []
    agg_counts = {name: 0 for name in CLASS_NAMES.values()}
    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        out_file = out_dir / f"pred_{img_path.name}"
        _, summary = predict_image(model, img_path, out_file, conf=conf, iou=iou, imgsz=imgsz)
        results.append(summary)
        for k, v in summary["counts"].items():
            agg_counts[k] += v
        print(f"[{idx:02d}/{len(images):02d}] {img_path.name} -> {summary['total_faces']} faces ({summary['counts']}) [{summary['latency_ms']}ms]")

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("BATCH INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Images Processed : {len(images)}")
    print(f"Total Faces      : {sum(agg_counts.values())}")
    for k, v in agg_counts.items():
        print(f"  {k:<22}: {v}")
    print(f"Total Time       : {total_time:.2f}s (Avg {total_time/len(images)*1000:.1f}ms/image)")
    print(f"Saved to         : {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Face Mask Image Detection")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Optional positional path to an uploaded image file"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to an uploaded image file, a folder of images, or 'test' for default test dataset"
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45 to eliminate overlapping boxes)")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference resolution (default: 416)")
    parser.add_argument("--output", type=str, default="runs/face_mask/predictions", help="Output directory")
    parser.add_argument("--limit", type=int, default=20, help="Max images in batch mode")

    args = parser.parse_args()

    # Determine input source
    source_arg = args.image_path if args.image_path else (args.source if args.source else "test")

    weights_file = Path(args.weights) if args.weights else find_best_weights()
    if not weights_file.exists():
        print(f"[ERROR] Model weights not found: {weights_file}")
        sys.exit(1)

    print(f"[INFO] Using model weights: {weights_file}")
    detector = YOLO(str(weights_file))

    if source_arg == "test":
        src = ROOT / "dataset" / "test" / "images"
    else:
        src = Path(source_arg)

    if not src.exists():
        print(f"\n[ERROR] Provided path does not exist: {src}")
        print("Please check the file or directory path and try again.")
        sys.exit(1)

    if src.is_dir():
        predict_batch(detector, src, args.output, conf=args.conf, iou=args.iou, imgsz=args.imgsz, limit=args.limit)
    elif src.is_file():
        out_file = Path(args.output) / f"pred_{src.name}"
        _, summary = predict_image(detector, src, out_file, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Input Image : {summary['image']}")
        print(f"Output Image: {summary['output']}")
        print(f"Total Faces : {summary['total_faces']}")
        print(f"Breakdown   : {summary['counts']}")
        print(f"Latency     : {summary['latency_ms']}ms")

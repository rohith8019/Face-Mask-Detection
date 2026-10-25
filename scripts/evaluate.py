"""
YOLOv8 Face Mask Detection — Dedicated Evaluation Pipeline
Evaluates a specified weights file on validation or test splits with per-class and overall metrics.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def find_eval_weights():
    candidates = [
        ROOT / "models" / "best.pt",
        ROOT / "runs" / "face_mask" / "baseline_416_10ep" / "weights" / "best.pt",
        ROOT / "runs" / "face_mask" / "quick_416" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def evaluate(weights=None, split="val", imgsz=416, batch=4, plots=True):
    weights_path = Path(weights) if weights else find_eval_weights()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    print("=" * 70)
    print(f"EVALUATING MODEL: {weights_path}")
    print(f"Split: {split.upper()} | Image Resolution: {imgsz}x{imgsz} | Batch Size: {batch}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        plots=plots,
        project=str(ROOT / "runs" / "face_mask"),
        name=f"eval_{split}_{weights_path.stem}",
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(f"EVALUATION METRICS SUMMARY ({split.upper()} SPLIT)")
    print("=" * 70)
    print(f"Overall Precision : {metrics.box.mp * 100:.2f}%")
    print(f"Overall Recall    : {metrics.box.mr * 100:.2f}%")
    print(f"Overall mAP@50    : {metrics.box.map50 * 100:.2f}%")
    print(f"Overall mAP@50-95 : {metrics.box.map * 100:.2f}%")

    print("\nPer-class Metrics:")
    for cid, cname in model.names.items():
        p = metrics.box.p[cid] if cid < len(metrics.box.p) else 0.0
        r = metrics.box.r[cid] if cid < len(metrics.box.r) else 0.0
        ap50 = metrics.box.ap50[cid] if cid < len(metrics.box.ap50) else 0.0
        ap = metrics.box.ap[cid] if cid < len(metrics.box.ap) else 0.0
        print(f"  Class {cid} ({cname:<22}): P={p*100:>5.2f}%, R={r*100:>5.2f}%, mAP50={ap50*100:>5.2f}%, mAP50-95={ap*100:>5.2f}%")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Face Mask Model")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=416, help="Image resolution (default: 416)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate evaluation plots")

    args = parser.parse_args()
    evaluate(
        weights=args.weights,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=args.plots
    )

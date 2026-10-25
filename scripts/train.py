"""
YOLOv8 Face Mask Detection — Training Pipeline
Step 3: Pretrained Weights Loading, Smoke Test, Baseline Training, and Model Evaluation
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Project configuration
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "data.yaml"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset YAML not found at: {DATA}")


def run_smoke_test():
    """Step 4: Run a short 2-epoch smoke test at 320px to verify pipeline integrity."""
    print("\n" + "=" * 60)
    print("STEP 4: RUNNING SMOKE TEST (2 Epochs, imgsz=320, batch=4)")
    print("=" * 60)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA),
        epochs=2,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs" / "face_mask"),
        name="smoke_test",
        seed=42,
        plots=True,
        exist_ok=True
    )
    print("\n[SUCCESS] Smoke test completed successfully!")
    print(f"Results saved to: {ROOT / 'runs' / 'face_mask' / 'smoke_test'}")
    return results


def run_baseline_training(epochs=10, imgsz=640, batch=4, name="baseline", workers=2):
    """Step 5: Train the YOLOv8n model on CPU with custom parameters."""
    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {name} ({epochs} Epochs, imgsz={imgsz}, batch={batch})")
    print("=" * 60)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=workers,
        project=str(ROOT / "runs" / "face_mask"),
        name=name,
        seed=42,
        plots=True,
        exist_ok=True
    )
    print(f"\n[SUCCESS] Training {name} completed successfully!")
    print(f"Results saved to: {ROOT / 'runs' / 'face_mask' / name}")
    return results


def evaluate_model(weights_path=None, split="val", imgsz=640, batch=4):
    """Step 6: Evaluate trained weights and print overall and per-class AP."""
    print("\n" + "=" * 60)
    print(f"STEP 6: EVALUATING MODEL ON {split.upper()} SET")
    print("=" * 60)

    if weights_path is None or not Path(weights_path).exists():
        candidates = [
            ROOT / "runs" / "face_mask" / "baseline_640" / "weights" / "best.pt",
            ROOT / "runs" / "face_mask" / "quick_416" / "weights" / "best.pt",
            ROOT / "runs" / "face_mask" / "baseline" / "weights" / "best.pt",
            ROOT / "runs" / "face_mask" / "smoke_test" / "weights" / "best.pt",
        ]
        for c in candidates:
            if c.exists():
                weights_path = c
                break
        if weights_path is None or not Path(weights_path).exists():
            raise FileNotFoundError("No checkpoint weights found in runs/face_mask/")

    print(f"Evaluating checkpoint: {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATA),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        plots=True
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({split.upper()} SPLIT)")
    print("=" * 60)
    print(f"mAP@50    : {metrics.box.map50 * 100:.2f}%")
    print(f"mAP@50-95 : {metrics.box.map * 100:.2f}%")
    print(f"Precision : {metrics.box.mp * 100:.2f}%")
    print(f"Recall    : {metrics.box.mr * 100:.2f}%")
    print("\nPer-class AP@50 breakdown:")
    for idx, cname in model.names.items():
        ap50 = metrics.box.maps[idx] if idx < len(metrics.box.maps) else 0.0
        print(f"  Class {idx} ({cname:<22}): {ap50 * 100:>6.2f}%")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Face Mask Training & Evaluation")
    parser.add_argument(
        "--mode",
        choices=["smoke_test", "baseline", "evaluate"],
        default="smoke_test",
        help="Pipeline stage: 'smoke_test', 'baseline', or 'evaluate'"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for training")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--name", type=str, default=None, help="Custom run name under runs/face_mask/")
    parser.add_argument("--weights", type=str, default=None, help="Weights path for evaluation")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split for evaluation")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default 2)")

    args = parser.parse_args()

    if args.mode == "smoke_test":
        run_smoke_test()
    elif args.mode == "baseline":
        run_name = args.name if args.name else ("quick_416" if args.imgsz == 416 else "baseline")
        run_baseline_training(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=run_name,
            workers=args.workers
        )
    elif args.mode == "evaluate":
        evaluate_model(
            weights_path=args.weights,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch
        )

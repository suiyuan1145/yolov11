#!/usr/bin/env python3
"""Train a YOLO11 detector on the local PlantDoc YOLO dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLO  # noqa: E402
from yolo_custom import register_custom_modules  # noqa: E402

register_custom_modules()

DEFAULT_DATA = ROOT / "dataset.yaml"
DEFAULT_MODEL = ROOT / "configs" / "yolo11n_single_channel_spatial.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on PlantDoc.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to YOLO data yaml.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO11 model yaml or checkpoint.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=24, help="Batch size. Use -1 for auto batch.")
    parser.add_argument("--device", default="", help="Device, for example 0, 0,1, cpu, or empty for auto.")
    parser.add_argument("--workers", type=int, default=32, help="Dataloader workers.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs", help="Output project directory.")
    parser.add_argument("--name", default="plantdoc_yolo11", help="Run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest run if possible.")
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"), help="Split used for final metrics.")
    parser.add_argument("--fps-warmup", type=int, default=5, help="Warmup images before FPS timing.")
    return parser.parse_args()


def resolve_split_images(data_yaml: Path, split: str) -> Path | None:
    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    split_path = data_cfg.get(split)
    if split_path is None:
        return None

    base = Path(data_cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    split_images = Path(split_path)
    if not split_images.is_absolute():
        split_images = base / split_images
    return split_images.resolve()


def metric_value(metrics: object, name: str) -> float | None:
    value = getattr(metrics.box, name, None)
    return float(value) if value is not None else None


def compute_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def measure_predict_fps(
    model: YOLO,
    source: Path,
    imgsz: int,
    device: str,
    warmup: int,
) -> dict[str, float | int | str | None]:
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for path in source.glob(suffix)
    )
    if not image_paths:
        return {"predict_images": 0, "predict_fps": None, "predict_seconds": None}

    warmup_paths = [str(path) for path in image_paths[: min(warmup, len(image_paths))]]
    if warmup_paths:
        list(model.predict(source=warmup_paths, imgsz=imgsz, device=device, verbose=False))

    start = time.perf_counter()
    results = list(model.predict(source=[str(path) for path in image_paths], imgsz=imgsz, device=device, verbose=False))
    elapsed = time.perf_counter() - start
    count = len(results)
    fps = count / elapsed if elapsed > 0 else None
    return {"predict_images": count, "predict_fps": fps, "predict_seconds": elapsed}


def save_experiment_summary(
    save_dir: Path,
    args: argparse.Namespace,
    weights_path: Path,
    metrics: object,
    fps_metrics: dict[str, float | int | str | None],
) -> None:
    precision = metric_value(metrics, "mp")
    recall = metric_value(metrics, "mr")
    map50 = metric_value(metrics, "map50")
    map50_95 = metric_value(metrics, "map")
    f1 = compute_f1(precision, recall)

    speed = getattr(metrics, "speed", {}) or {}
    inference_ms = speed.get("inference")
    postprocess_ms = speed.get("postprocess")
    val_fps = None
    if inference_ms is not None and postprocess_ms is not None and inference_ms + postprocess_ms > 0:
        val_fps = 1000.0 / (inference_ms + postprocess_ms)

    summary = {
        "task": "PlantDoc YOLO11 object detection",
        "data": str(args.data),
        "model": str(args.model),
        "weights": str(weights_path),
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device or "auto",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "val_speed_ms_per_image": speed,
        "val_fps_inference_postprocess": val_fps,
        **fps_metrics,
    }

    json_path = save_dir / "experiment_summary.json"
    txt_path = save_dir / "experiment_summary.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "PlantDoc YOLO11 Experiment Summary",
        f"Data: {summary['data']}",
        f"Model: {summary['model']}",
        f"Weights: {summary['weights']}",
        f"Eval split: {summary['eval_split']}",
        f"Epochs: {summary['epochs']}",
        f"Image size: {summary['imgsz']}",
        f"Batch: {summary['batch']}",
        f"Device: {summary['device']}",
        f"Precision: {precision:.6f}" if precision is not None else "Precision: N/A",
        f"Recall: {recall:.6f}" if recall is not None else "Recall: N/A",
        f"F1: {f1:.6f}" if f1 is not None else "F1: N/A",
        f"mAP50: {map50:.6f}" if map50 is not None else "mAP50: N/A",
        f"mAP50-95: {map50_95:.6f}" if map50_95 is not None else "mAP50-95: N/A",
        f"Val FPS(inference+postprocess): {val_fps:.2f}" if val_fps is not None else "Val FPS(inference+postprocess): N/A",
        (
            f"Predict FPS(end-to-end): {fps_metrics['predict_fps']:.2f}"
            if fps_metrics.get("predict_fps") is not None
            else "Predict FPS(end-to-end): N/A"
        ),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = args.data.expanduser().resolve()

    if not data.exists():
        raise FileNotFoundError(f"Data yaml not found: {data}")

    model_path = args.model.expanduser().resolve() if isinstance(args.model, Path) else args.model
    if isinstance(model_path, Path) and not model_path.exists():
        raise FileNotFoundError(f"Model yaml/checkpoint not found: {model_path}")

    project = args.project.expanduser().resolve()

    model = YOLO(str(model_path))
    train_results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )

    save_dir = Path(getattr(train_results, "save_dir", model.trainer.save_dir)).resolve()
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"
    weights_path = best_weights if best_weights.exists() else last_weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found in: {save_dir / 'weights'}")

    eval_model = YOLO(str(weights_path))
    metrics = eval_model.val(
        data=str(data),
        split=args.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(save_dir),
        name=f"{args.eval_split}_metrics",
    )

    split_images = resolve_split_images(data, args.eval_split)
    fps_metrics = {"predict_images": 0, "predict_fps": None, "predict_seconds": None}
    if split_images is not None and split_images.exists():
        fps_metrics = measure_predict_fps(
            eval_model,
            source=split_images,
            imgsz=args.imgsz,
            device=args.device,
            warmup=args.fps_warmup,
        )

    save_experiment_summary(save_dir, args, weights_path, metrics, fps_metrics)
    print(f"Experiment summary saved to: {save_dir / 'experiment_summary.txt'}")


if __name__ == "__main__":
    main()

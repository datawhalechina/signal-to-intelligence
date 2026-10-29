"""Run a reproducible YOLO vs DETR comparison on a COCO person subset."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from statistics import mean, median
from typing import Callable

from .data import CocoImage, load_person_subset
from .detr_runner import DETR_MODEL_NAME, DETR_MODEL_REVISION, DetrPersonDetector
from .error_analysis import render_error_cases
from .evaluate import (
    Detection,
    best_f1_point,
    evaluate_detections,
    point_at_recall,
    precision_recall_curve,
)
from .report import write_comparison_plots
from .yolo_runner import YOLO_IMAGE_SIZE, YOLO_MODEL_NAME, YoloPersonDetector


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DetectorFactory = Callable[[str, float], object]


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without an extra dependency."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(latencies),
        "median_ms": median(latencies),
        "p95_ms": _percentile(latencies, 0.95),
    }


def _point_payload(point) -> dict | None:
    if point is None:
        return None
    return {**vars(point), "f1": point.f1}


def _console_summary(result: dict) -> dict:
    """Keep terminal output readable while the full PR curve stays in metrics.json."""
    return {
        "config": {
            "images": result["config"]["images"],
            "device": result["config"]["device"],
            "score_threshold": result["config"]["score_threshold"],
            "iou_threshold": result["config"]["iou_threshold"],
        },
        "models": {
            name: {
                "metrics": model_result["metrics"],
                "latency": model_result["latency"],
                "operating_points": model_result["operating_points"],
            }
            for name, model_result in result["models"].items()
        },
        "artifacts": result.get("artifacts", {}),
    }


def _default_detector_factories() -> dict[str, DetectorFactory]:
    return {
        "yolov8n": lambda device, threshold: YoloPersonDetector(
            device=device, score_threshold=threshold
        ),
        "detr_resnet50": lambda device, threshold: DetrPersonDetector(
            device=device, score_threshold=threshold
        ),
    }


def _validate_images(subset: list[CocoImage], images_dir: Path) -> None:
    missing = [image.filename for image in subset if not (images_dir / image.filename).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Missing {len(missing)} selected images, including: {preview}")


def run_comparison(
    annotation_path: Path,
    images_dir: Path,
    limit: int,
    device: str,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    warmup: bool = True,
    error_case_limit: int = 3,
    output_dir: Path | None = None,
    detector_factories: dict[str, DetectorFactory] | None = None,
) -> dict:
    """Run all detectors under one data, timing, and metric contract."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    if error_case_limit < 0:
        raise ValueError("error_case_limit must be non-negative")

    subset = load_person_subset(annotation_path, limit)
    if not subset:
        raise ValueError("No non-crowd person annotations found in the supplied COCO file")
    _validate_images(subset, images_dir)

    factories = detector_factories or _default_detector_factories()
    ground_truth = [box for image in subset for box in image.ground_truth]
    model_results = {}
    for name, factory in factories.items():
        detector = factory(device, score_threshold)
        if warmup:
            detector.predict(images_dir / subset[0].filename, subset[0].image_id)

        predictions: list[Detection] = []
        predictions_by_image: dict[str, list[Detection]] = {}
        latencies = []
        for image in subset:
            image_predictions, latency_ms = detector.predict(
                images_dir / image.filename, image.image_id
            )
            predictions.extend(image_predictions)
            predictions_by_image[image.image_id] = image_predictions
            latencies.append(latency_ms)

        metrics = evaluate_detections(predictions, ground_truth, iou_threshold)
        curve = precision_recall_curve(predictions, ground_truth, iou_threshold)
        best_point = best_f1_point(curve)
        error_cases = []
        if output_dir is not None and error_case_limit:
            error_predictions = {
                image_id: [
                    prediction
                    for prediction in image_predictions
                    if best_point is None or prediction.score >= best_point.threshold
                ]
                for image_id, image_predictions in predictions_by_image.items()
            }
            error_cases = render_error_cases(
                model_name=name,
                images_dir=images_dir,
                subset=subset,
                predictions_by_image=error_predictions,
                output_dir=output_dir / "error_cases",
                iou_threshold=iou_threshold,
                limit=error_case_limit,
            )
        model_results[name] = {
            "metrics": vars(metrics),
            "latency": _latency_summary(latencies),
            "prediction_count": len(predictions),
            "pr_curve": [_point_payload(point) for point in curve],
            "operating_points": {
                "best_f1": _point_payload(best_point),
                "recall_0_50": _point_payload(point_at_recall(curve, 0.5)),
                "recall_0_80": _point_payload(point_at_recall(curve, 0.8)),
            },
            "error_case_threshold": best_point.threshold if best_point else score_threshold,
            "error_cases": error_cases,
        }

    result = {
        "schema_version": 1,
        "config": {
            "annotations": str(annotation_path),
            "images_dir": str(images_dir),
            "images": len(subset),
            "selected_image_ids": [image.image_id for image in subset],
            "category": "person",
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "device": device,
            "warmup": warmup,
            "models": {
                "yolov8n": YOLO_MODEL_NAME,
                "detr_resnet50": f"{DETR_MODEL_NAME}@{DETR_MODEL_REVISION}",
            },
            "preprocessing": {
                "yolo_image_size": YOLO_IMAGE_SIZE,
                "detr_processor": "checkpoint preprocessor_config.json (slow implementation)",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "ultralytics": _package_version("ultralytics"),
        },
        "models": model_results,
    }
    if output_dir is not None:
        result["artifacts"] = write_comparison_plots(result, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare YOLOv8n and DETR on the same COCO person images."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="instances_val*.json path")
    parser.add_argument(
        "--images-dir", type=Path, required=True, help="COCO validation image directory"
    )
    parser.add_argument("--limit", type=int, default=50, help="Number of person-containing images")
    parser.add_argument(
        "--device", default="cpu", help="Torch/Ultralytics device, e.g. cpu or cuda:0"
    )
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--error-cases", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = run_comparison(
        annotation_path=args.annotations,
        images_dir=args.images_dir,
        limit=args.limit,
        device=args.device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        warmup=not args.no_warmup,
        error_case_limit=args.error_cases,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(_console_summary(result), indent=2))
    print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()

"""Rank and render model errors for the vision detection comparison."""

from pathlib import Path

from .data import CocoImage
from .evaluate import Detection, evaluate_detections


def render_error_cases(
    model_name: str,
    images_dir: Path,
    subset: list[CocoImage],
    predictions_by_image: dict[str, list[Detection]],
    output_dir: Path,
    iou_threshold: float,
    limit: int,
) -> list[dict]:
    """Save the highest-error images with ground-truth and prediction overlays."""
    from PIL import Image, ImageDraw

    ranked_cases = []
    for image in subset:
        predictions = predictions_by_image.get(image.image_id, [])
        metrics = evaluate_detections(
            predictions,
            list(image.ground_truth),
            iou_threshold=iou_threshold,
        )
        error_count = metrics.false_negatives + metrics.false_positives
        if error_count:
            ranked_cases.append((error_count, metrics.false_negatives, image, predictions, metrics))
    ranked_cases.sort(key=lambda item: (item[0], item[1]), reverse=True)

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for _, _, image_info, predictions, metrics in ranked_cases[:limit]:
        with Image.open(images_dir / image_info.filename) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for truth in image_info.ground_truth:
            draw.rectangle((truth.x1, truth.y1, truth.x2, truth.y2), outline="#16a34a", width=4)
        for prediction in predictions:
            draw.rectangle(
                (prediction.x1, prediction.y1, prediction.x2, prediction.y2),
                outline="#f97316",
                width=3,
            )
            draw.text(
                (prediction.x1 + 3, prediction.y1 + 3),
                f"{prediction.score:.2f}",
                fill="#f97316",
            )

        output_path = model_output_dir / f"{image_info.image_id}.jpg"
        image.save(output_path, quality=90)
        summaries.append(
            {
                "image_id": image_info.image_id,
                "filename": image_info.filename,
                "output": str(output_path),
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "legend": "green=ground truth, orange=prediction",
            }
        )
    return summaries

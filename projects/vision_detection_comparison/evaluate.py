"""Model-agnostic object-detection evaluation for the YOLO/DETR comparison.

The module evaluates already-produced predictions. It intentionally does not
download models or datasets, so both runners can use the exact same metrics.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """One bounding box in ``(x1, y1, x2, y2)`` image coordinates."""

    image_id: str
    category_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection(BoundingBox):
    """A predicted bounding box with a confidence score."""

    score: float


@dataclass(frozen=True)
class DetectionMetrics:
    """Summary metrics at one IoU threshold."""

    average_precision: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class PrecisionRecallPoint:
    """Metrics after accepting predictions down to one score threshold."""

    threshold: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 0.0 if denominator == 0 else 2 * self.precision * self.recall / denominator


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Compute IoU; boxes from different images never overlap."""
    if first.image_id != second.image_id:
        return 0.0
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    union = first.area() + second.area() - intersection
    return 0.0 if union <= 0 else intersection / union


def evaluate_detections(
    predictions: list[Detection],
    ground_truth: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Evaluate predictions with one-to-one, same-class matching at a fixed IoU."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered_predictions, true_positive_flags = _match_predictions(
        predictions,
        ground_truth,
        iou_threshold,
    )
    true_positives = int(sum(true_positive_flags))
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    average_precision = _average_precision(true_positive_flags, len(ground_truth))
    return DetectionMetrics(
        average_precision=average_precision,
        precision=precision,
        recall=recall,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def precision_recall_curve(
    predictions: list[Detection],
    ground_truth: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> list[PrecisionRecallPoint]:
    """Return one PR point for each distinct prediction score."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered_predictions, true_positive_flags = _match_predictions(
        predictions,
        ground_truth,
        iou_threshold,
    )
    if not ordered_predictions:
        return []

    cumulative_true_positives = np.cumsum(true_positive_flags)
    points = []
    for index, prediction in enumerate(ordered_predictions):
        is_last_at_score = (
            index == len(ordered_predictions) - 1
            or ordered_predictions[index + 1].score != prediction.score
        )
        if not is_last_at_score:
            continue
        accepted = index + 1
        true_positives = int(cumulative_true_positives[index])
        false_positives = accepted - true_positives
        points.append(
            PrecisionRecallPoint(
                threshold=prediction.score,
                precision=true_positives / accepted,
                recall=true_positives / len(ground_truth) if ground_truth else 0.0,
                true_positives=true_positives,
                false_positives=false_positives,
            )
        )
    return points


def best_f1_point(points: list[PrecisionRecallPoint]) -> PrecisionRecallPoint | None:
    """Choose the highest-F1 point, preferring the higher threshold on ties."""
    if not points:
        return None
    return max(points, key=lambda point: (point.f1, point.threshold))


def point_at_recall(
    points: list[PrecisionRecallPoint], target_recall: float
) -> PrecisionRecallPoint | None:
    """Choose the highest-precision point that reaches a requested recall."""
    if not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be in [0, 1]")
    eligible = [point for point in points if point.recall >= target_recall]
    if not eligible:
        return None
    return max(eligible, key=lambda point: (point.precision, point.threshold))


def _match_predictions(
    predictions: list[Detection],
    ground_truth: list[BoundingBox],
    iou_threshold: float,
) -> tuple[list[Detection], list[float]]:
    """Return score-ordered predictions and one-to-one match flags."""
    ordered_predictions = sorted(predictions, key=lambda item: item.score, reverse=True)
    matched_truth_indices: set[int] = set()
    true_positive_flags: list[float] = []

    for prediction in ordered_predictions:
        candidates = [
            (index, truth)
            for index, truth in enumerate(ground_truth)
            if index not in matched_truth_indices
            and truth.image_id == prediction.image_id
            and truth.category_id == prediction.category_id
        ]
        best_index, best_iou = None, 0.0
        for index, truth in candidates:
            current_iou = intersection_over_union(prediction, truth)
            if current_iou > best_iou:
                best_index, best_iou = index, current_iou
        if best_index is not None and best_iou >= iou_threshold:
            matched_truth_indices.add(best_index)
            true_positive_flags.append(1.0)
        else:
            true_positive_flags.append(0.0)
    return ordered_predictions, true_positive_flags


def _average_precision(true_positive_flags: list[float], number_of_truths: int) -> float:
    """Compute interpolated AP from score-ordered true/false-positive flags."""
    if number_of_truths == 0:
        return 0.0
    if not true_positive_flags:
        return 0.0
    flags = np.asarray(true_positive_flags)
    cumulative_true_positives = np.cumsum(flags)
    recalls = cumulative_true_positives / number_of_truths
    precisions = cumulative_true_positives / np.arange(1, len(flags) + 1)
    interpolated_precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    recall_points = np.concatenate(([0.0], recalls))
    precision_points = np.concatenate(([interpolated_precisions[0]], interpolated_precisions))
    return float(np.sum(np.diff(recall_points) * precision_points[1:]))

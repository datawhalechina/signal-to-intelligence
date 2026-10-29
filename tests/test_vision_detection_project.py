"""Tests for the model-neutral evaluation layer of the vision project."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "projects" / "vision_detection_comparison"


def load_evaluate_module():
    path = PROJECT_DIR / "evaluate.py"
    spec = importlib.util.spec_from_file_location("vision_project_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def load_main_module():
    return importlib.import_module("projects.vision_detection_comparison.main")


def test_iou_is_one_for_identical_boxes_and_zero_across_images():
    evaluate = load_evaluate_module()
    first = evaluate.BoundingBox("image-1", 1, 0, 0, 10, 10)
    same = evaluate.BoundingBox("image-1", 1, 0, 0, 10, 10)
    other_image = evaluate.BoundingBox("image-2", 1, 0, 0, 10, 10)

    assert evaluate.intersection_over_union(first, same) == 1.0
    assert evaluate.intersection_over_union(first, other_image) == 0.0


def test_duplicate_predictions_count_as_false_positives_after_first_match():
    evaluate = load_evaluate_module()
    truth = [evaluate.BoundingBox("image-1", 1, 0, 0, 10, 10)]
    predictions = [
        evaluate.Detection("image-1", 1, 0, 0, 10, 10, score=0.9),
        evaluate.Detection("image-1", 1, 0, 0, 10, 10, score=0.8),
    ]

    metrics = evaluate.evaluate_detections(predictions, truth)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == 0.5


def test_wrong_class_and_low_iou_predictions_are_not_matches():
    evaluate = load_evaluate_module()
    truth = [evaluate.BoundingBox("image-1", 1, 0, 0, 10, 10)]
    predictions = [
        evaluate.Detection("image-1", 2, 0, 0, 10, 10, score=0.9),
        evaluate.Detection("image-1", 1, 20, 20, 30, 30, score=0.8),
    ]

    metrics = evaluate.evaluate_detections(predictions, truth)

    assert metrics.true_positives == 0
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.average_precision == 0.0


def test_precision_recall_curve_exposes_comparable_operating_points():
    evaluate = load_evaluate_module()
    truths = [
        evaluate.BoundingBox("image-1", 1, 0, 0, 10, 10),
        evaluate.BoundingBox("image-1", 1, 20, 20, 30, 30),
    ]
    predictions = [
        evaluate.Detection("image-1", 1, 0, 0, 10, 10, score=0.9),
        evaluate.Detection("image-1", 1, 40, 40, 50, 50, score=0.8),
        evaluate.Detection("image-1", 1, 20, 20, 30, 30, score=0.7),
    ]

    points = evaluate.precision_recall_curve(predictions, truths)
    best = evaluate.best_f1_point(points)
    high_recall = evaluate.point_at_recall(points, 0.8)

    assert [point.recall for point in points] == [0.5, 0.5, 1.0]
    assert best.threshold == 0.7
    assert high_recall.threshold == 0.7


def test_comparison_loads_detector_once_and_excludes_warmup_from_latency(tmp_path):
    main = load_main_module()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    annotations = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 32, "height": 32},
            {"id": 2, "file_name": "two.jpg", "width": 32, "height": 32},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 10, 10]},
        ],
    }
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    for filename in ("one.jpg", "two.jpg"):
        (images_dir / filename).write_bytes(b"test fixture")

    created_detectors = []

    class FakeDetector:
        def __init__(self):
            self.calls = []

        def predict(self, image_path, image_id):
            self.calls.append(image_id)
            offset = 0 if image_id == "1" else 5
            prediction = main.Detection(image_id, 1, offset, offset, offset + 10, offset + 10, 0.9)
            latency = 10.0 if image_id == "1" else 20.0
            return [prediction], latency

    def factory(device, threshold):
        detector = FakeDetector()
        created_detectors.append(detector)
        return detector

    result = main.run_comparison(
        annotation_path=annotation_path,
        images_dir=images_dir,
        limit=2,
        device="cpu",
        detector_factories={"fake": factory},
    )

    assert len(created_detectors) == 1
    assert created_detectors[0].calls == ["1", "1", "2"]
    assert result["models"]["fake"]["metrics"]["average_precision"] == 1.0
    assert result["models"]["fake"]["latency"]["mean_ms"] == 15.0
    assert result["models"]["fake"]["error_case_threshold"] == 0.9


def test_coco8_conversion_keeps_person_boxes_and_pixel_coordinates(tmp_path):
    from PIL import Image

    dataset_dir = tmp_path / "coco8"
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(image_dir / "000000000009.jpg")
    (label_dir / "000000000009.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n2 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )

    path = PROJECT_DIR / "prepare_coco8.py"
    spec = importlib.util.spec_from_file_location("vision_project_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotation_path, images_dir = module.prepare_coco8(dataset_dir, tmp_path / "prepared")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))

    assert images_dir.joinpath("000000000009.jpg").is_file()
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["bbox"] == [30.0, 15.0, 40.0, 20.0]


def test_coco_val_preparation_selects_fixed_person_subset(tmp_path):
    from PIL import Image

    prepare = importlib.import_module("projects.vision_detection_comparison.prepare_coco_val")
    labels_dir = tmp_path / "labels"
    source_images = tmp_path / "source-images"
    labels_dir.mkdir()
    source_images.mkdir()
    (labels_dir / "000000000003.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels_dir / "000000000002.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (labels_dir / "000000000001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    for image_id in ("000000000001", "000000000002"):
        Image.new("RGB", (100, 50)).save(source_images / f"{image_id}.jpg")

    annotation_path, images_dir = prepare.prepare_coco_val(
        labels_dir=labels_dir,
        output_dir=tmp_path / "prepared",
        limit=2,
        workers=1,
        image_base_url=source_images.as_uri(),
    )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))

    assert [image["id"] for image in payload["images"]] == [1, 2]
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][0]["bbox"] == [40.0, 15.0, 20.0, 20.0]
    assert len(list(images_dir.glob("*.jpg"))) == 2


def test_report_writes_pr_and_latency_plots(tmp_path):
    report = importlib.import_module("projects.vision_detection_comparison.report")
    point = {
        "threshold": 0.5,
        "precision": 0.75,
        "recall": 0.6,
        "true_positives": 3,
        "false_positives": 1,
        "f1": 2 / 3,
    }
    result = {
        "config": {"images": 5},
        "models": {
            "fake": {
                "metrics": {"average_precision": 0.7},
                "latency": {"mean_ms": 10.0, "p95_ms": 12.0},
                "pr_curve": [point],
                "operating_points": {"best_f1": point},
            }
        },
    }

    artifacts = report.write_comparison_plots(result, tmp_path)

    assert Path(artifacts["precision_recall_curve"]).is_file()
    assert Path(artifacts["latency_comparison"]).is_file()

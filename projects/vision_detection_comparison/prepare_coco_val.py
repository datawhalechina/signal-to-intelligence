"""Prepare a deterministic person subset from Ultralytics COCO val labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from PIL import Image


DEFAULT_IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"


def _person_rows(label_path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label row in {label_path}: {line!r}")
        class_id, center_x, center_y, width, height = map(float, values)
        if int(class_id) == 0:
            rows.append((center_x, center_y, width, height))
    return rows


def select_person_labels(labels_dir: Path, limit: int) -> list[Path]:
    """Select the lowest image IDs that contain at least one person label."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if _person_rows(label_path):
            selected.append(label_path)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Found only {len(selected)} person images; requested {limit}")
    return selected


def _download_image(image_id: str, images_dir: Path, image_base_url: str) -> Path:
    destination = images_dir / f"{image_id}.jpg"
    if destination.is_file():
        return destination
    source = f"{image_base_url.rstrip('/')}/{image_id}.jpg"
    temporary = destination.with_suffix(".jpg.part")
    try:
        with urlopen(source, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_coco_val(
    labels_dir: Path,
    output_dir: Path,
    limit: int = 50,
    workers: int = 8,
    image_base_url: str = DEFAULT_IMAGE_BASE_URL,
) -> tuple[Path, Path]:
    """Download selected val images and create a minimal COCO person annotation file."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    selected = select_person_labels(labels_dir, limit)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [path.stem for path in selected]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        image_paths = list(
            executor.map(
                lambda image_id: _download_image(image_id, images_dir, image_base_url),
                image_ids,
            )
        )

    images, annotations = [], []
    annotation_id = 1
    for label_path, image_path in zip(selected, image_paths):
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        image_id = int(label_path.stem)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for center_x, center_y, box_width, box_height in _person_rows(label_path):
            pixel_width = box_width * width
            pixel_height = box_height * height
            x = max(0.0, center_x * width - pixel_width / 2)
            y = max(0.0, center_y * height - pixel_height / 2)
            pixel_width = min(pixel_width, width - x)
            pixel_height = min(pixel_height, height - y)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    annotation_path = output_dir / "instances_val2017_person_subset.json"
    annotation_path.write_text(
        json.dumps(
            {
                "info": {
                    "description": (
                        "Deterministic COCO val2017 person subset from Ultralytics YOLO labels"
                    ),
                    "selection": f"lowest {limit} image IDs containing class 0",
                },
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return annotation_path, images_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed person subset from Ultralytics COCO val2017 labels."
    )
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    args = parser.parse_args()
    annotation_path, images_dir = prepare_coco_val(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
        image_base_url=args.image_base_url,
    )
    print(f"Annotations: {annotation_path}")
    print(f"Images: {images_dir}")


if __name__ == "__main__":
    main()

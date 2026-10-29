"""COCO subset loading utilities for a reproducible one-class comparison."""

import json
from dataclasses import dataclass
from pathlib import Path

from .evaluate import BoundingBox


PERSON_CATEGORY_ID = 1


@dataclass(frozen=True)
class CocoImage:
    """One selected COCO image and its person annotations."""

    image_id: str
    filename: str
    width: int
    height: int
    ground_truth: tuple[BoundingBox, ...]


def load_person_subset(annotation_path: Path, limit: int) -> list[CocoImage]:
    """Load up to ``limit`` COCO validation images containing at least one person."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    by_image: dict[int, list[BoundingBox]] = {}
    for annotation in payload["annotations"]:
        if annotation["category_id"] != PERSON_CATEGORY_ID or annotation.get("iscrowd", 0):
            continue
        x, y, width, height = annotation["bbox"]
        box = BoundingBox(
            image_id=str(annotation["image_id"]),
            category_id=PERSON_CATEGORY_ID,
            x1=float(x),
            y1=float(y),
            x2=float(x + width),
            y2=float(y + height),
        )
        by_image.setdefault(annotation["image_id"], []).append(box)

    subset = []
    for image_id in sorted(by_image):
        image = images[image_id]
        subset.append(
            CocoImage(
                image_id=str(image_id),
                filename=image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
                ground_truth=tuple(by_image[image_id]),
            )
        )
        if len(subset) == limit:
            break
    return subset

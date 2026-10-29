"""Convert Ultralytics COCO8 labels into this project's COCO person contract."""

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image


def prepare_coco8(dataset_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Collect person-containing COCO8 images and write a minimal COCO annotation file."""
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    annotation_id = 1

    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            with Image.open(image_path) as image:
                width, height = image.size

            person_boxes = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id, center_x, center_y, box_width, box_height = map(float, line.split())
                if int(class_id) != 0:
                    continue
                pixel_width = box_width * width
                pixel_height = box_height * height
                x = max(0.0, center_x * width - pixel_width / 2)
                y = max(0.0, center_y * height - pixel_height / 2)
                pixel_width = min(pixel_width, width - x)
                pixel_height = min(pixel_height, height - y)
                person_boxes.append((x, y, pixel_width, pixel_height))
            if not person_boxes:
                continue

            image_id = int(image_path.stem)
            shutil.copy2(image_path, images_output_dir / image_path.name)
            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                }
            )
            for x, y, box_width, box_height in person_boxes:
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    if not images:
        raise ValueError("No person labels found in the supplied COCO8 directory")
    annotation_path = output_dir / "instances_coco8_person.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return annotation_path, images_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare COCO8 for the person detector smoke test."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    annotation_path, images_dir = prepare_coco8(args.dataset_dir, args.output_dir)
    print(f"Annotations: {annotation_path}")
    print(f"Images: {images_dir}")


if __name__ == "__main__":
    main()

"""Optional Ultralytics YOLO adapter for the shared evaluation contract."""

import os
from pathlib import Path
from time import perf_counter

from .evaluate import Detection


YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_IMAGE_SIZE = 640


def default_model_path() -> Path:
    """Keep downloaded weights out of the repository working tree."""
    configured_cache = os.environ.get("STI_MODEL_CACHE") or os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(configured_cache) if configured_cache else Path.home() / ".cache"
    model_dir = cache_root / "signal-to-intelligence" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / YOLO_MODEL_NAME


def _synchronize_cuda(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


class YoloPersonDetector:
    """Load YOLO once and expose person detections through the shared contract."""

    def __init__(
        self,
        device: str = "cpu",
        score_threshold: float = 0.05,
        model_name: str = YOLO_MODEL_NAME,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO adapter requires `pip install ultralytics`. "
                "The model weight is downloaded by Ultralytics on first use."
            ) from exc
        self.device = device
        self.score_threshold = score_threshold
        model_path = Path(model_name)
        if model_name == YOLO_MODEL_NAME:
            model_path = default_model_path()
        self.model = YOLO(str(model_path))

    def predict(self, image_path: Path, image_id: str) -> tuple[list[Detection], float]:
        """Measure preprocessing, inference, and postprocessing, excluding disk I/O."""
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        _synchronize_cuda(self.device)
        started_at = perf_counter()
        result = self.model.predict(
            source=image,
            device=self.device,
            conf=self.score_threshold,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
        )[0]
        _synchronize_cuda(self.device)
        latency_ms = (perf_counter() - started_at) * 1000

        detections = []
        for box in result.boxes:
            if int(box.cls.item()) != 0:  # Ultralytics COCO index 0 is person.
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(image_id, 1, x1, y1, x2, y2, float(box.conf.item()))
            )
        return detections, latency_ms

"""Optional Hugging Face DETR adapter for the shared evaluation contract."""

from pathlib import Path
from time import perf_counter

from .evaluate import Detection


DETR_MODEL_NAME = "facebook/detr-resnet-50"
DETR_MODEL_REVISION = "main"


class DetrPersonDetector:
    """Load DETR once and expose person detections through the shared contract."""

    def __init__(
        self,
        device: str = "cpu",
        score_threshold: float = 0.05,
        model_name: str = DETR_MODEL_NAME,
        revision: str = DETR_MODEL_REVISION,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, DetrConfig, DetrForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "DETR adapter requires compatible torch, Pillow, and transformers packages."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            revision=revision,
            use_fast=False,
        )
        config = DetrConfig.from_pretrained(model_name, revision=revision)
        config.use_pretrained_backbone = False
        self.model = (
            DetrForObjectDetection.from_pretrained(
                model_name,
                revision=revision,
                config=config,
                use_safetensors=True,
            )
            .to(self.device)
            .eval()
        )

    def _synchronize_cuda(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, image_path: Path, image_id: str) -> tuple[list[Detection], float]:
        """Measure preprocessing, inference, and postprocessing, excluding disk I/O."""
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        self._synchronize_cuda()
        started_at = perf_counter()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]
        self._synchronize_cuda()
        latency_ms = (perf_counter() - started_at) * 1000

        detections = []
        for score, label, box in zip(
            processed["scores"], processed["labels"], processed["boxes"]
        ):
            if int(label.item()) != 1:  # DETR's COCO category id 1 is person.
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(image_id, 1, x1, y1, x2, y2, float(score.item())))
        return detections, latency_ms

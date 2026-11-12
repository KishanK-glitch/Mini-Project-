"""Object detection: YOLO11 via ultralytics, with an OpenVINO runtime for Intel CPU/GPU/NPU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def _intel_device() -> str:
    """Best OpenVINO device: Arc GPU if present, else CPU."""
    from openvino import Core

    devices = Core().available_devices  # e.g. ['CPU', 'GPU', 'NPU']
    log.info("OpenVINO devices: %s", devices)
    return "intel:gpu" if any(d.startswith("GPU") for d in devices) else "intel:cpu"


class YoloDetector:
    def __init__(
        self,
        model: str = "yolo11s.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str = "auto",
        classes: Optional[Sequence[str]] = None,
        runtime: str = "auto",
    ):
        from ultralytics import YOLO  # lazy: heavy import

        self._YOLO = YOLO
        self._model_name = model
        self.confidence, self.iou, self.imgsz = confidence, iou, imgsz
        self.runtime = "torch"
        self.model = None
        self.device = None

        wants_torch_device = str(device).startswith(("cuda", "mps"))
        if runtime in ("auto", "openvino") and not wants_torch_device:
            try:
                self.model, self.device = self._load_openvino(model, device)
                self.runtime = "openvino"
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenVINO unavailable (%s) - using PyTorch instead", exc)
        if self.model is None:
            self._load_torch(device)

        self.names = dict(self.model.names)  # {id: name}
        self.class_ids: Optional[List[int]] = None
        if classes:
            by_name = {n: i for i, n in self.names.items()}
            unknown = [c for c in classes if c not in by_name]
            if unknown:
                raise ValueError(f"Unknown YOLO classes {unknown}. Available: {sorted(by_name)}")
            self.class_ids = [by_name[c] for c in classes]
        log.info("YOLO ready: %s via %s (device=%s)", model, self.runtime, self.device or "auto")

    # ---------------------------------------------------------------- loading
    def _load_torch(self, device: str) -> None:
        self.model = self._YOLO(self._model_name)
        self.device = None if device in (None, "", "auto") or str(device).startswith("intel") else device
        self.runtime = "torch"

    def _load_openvino(self, model: str, device: str):
        dev = _intel_device() if device in (None, "", "auto") else device
        ov_dir = Path(model).with_name(f"{Path(model).stem}_openvino_model")
        if not ov_dir.exists():
            log.info("Exporting %s to OpenVINO (one-time, may take a minute)...", model)
            base = self._YOLO(model)
            exported = base.export(format="openvino", imgsz=self.imgsz, half=False)
            ov_dir = Path(str(exported))
        return self._YOLO(str(ov_dir), task="detect"), dev

    # -------------------------------------------------------------- inference
    def _predict(self, frame_bgr: np.ndarray):
        return self.model.predict(
            frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.class_ids,
            device=self.device,
            verbose=False,
        )[0]

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        try:
            result = self._predict(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            if self.runtime != "openvino":
                raise
            log.warning("OpenVINO inference failed (%s) - switching to PyTorch", exc)
            self._load_torch("auto")
            result = self._predict(frame_bgr)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(tuple(map(float, b)), float(c), int(k), self.names[int(k)])
            for b, c, k in zip(xyxy, conf, cls)
        ]


def build_detector(cfg: SimpleNamespace) -> YoloDetector:
    d = cfg.detection
    if d.backend != "yolo":
        raise ValueError(f"Unsupported detection backend: {d.backend}")
    return YoloDetector(
        model=d.model,
        confidence=d.confidence,
        iou=d.iou,
        imgsz=d.imgsz,
        device=d.device,
        classes=d.classes,
        runtime=getattr(d, "runtime", "auto"),
    )

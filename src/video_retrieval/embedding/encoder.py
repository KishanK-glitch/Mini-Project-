"""Image/text encoders (SigLIP, CLIP) producing L2-normalised embeddings in a shared space,
plus HSV colour measurement for person/vehicle crops."""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)).astype("float32")


def resolve_device(name: Optional[str]) -> str:
    """auto -> cuda (NVIDIA) > xpu (Intel Arc, needs the XPU build of PyTorch) > mps > cpu."""
    import torch

    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TorchEncoder:
    """Shared plumbing for HuggingFace dual encoders."""

    logit_scale: float = 100.0
    dim: int = 0

    def _to_device(self, model, device: str):
        try:
            model = model.to(device).eval()
            self.device = device
        except Exception as exc:  # e.g. XPU runtime missing
            log.warning("Could not use device %r (%s) - falling back to CPU", device, exc)
            model = model.to("cpu").eval()
            self.device = "cpu"
        return model

    @staticmethod
    def _tensor(out):
        # transformers 4.x returns a tensor; be tolerant of output objects.
        return out if hasattr(out, "shape") else out.pooler_output

    def _pil_batches(self, images_bgr: Sequence[np.ndarray]):
        from PIL import Image

        for i in range(0, len(images_bgr), self.batch_size):
            yield [
                Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                for im in images_bgr[i : i + self.batch_size]
            ]


class SiglipEncoder(_TorchEncoder):
    """Google SigLIP: sigmoid-loss CLIP successor, better at fine attributes and calibration."""

    def __init__(self, model_name: str, device: str, batch_size: int, query_template: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.batch_size = batch_size
        self.query_template = query_template
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        self.model = self._to_device(model, resolve_device(device))
        self.dim = int(model.config.vision_config.hidden_size)
        self.logit_scale = float(model.logit_scale.exp())
        log.info("SigLIP ready: %s on %s (dim=%d, scale=%.1f)",
                 model_name, self.device, self.dim, self.logit_scale)

    def encode_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        feats: List[np.ndarray] = []
        with self._torch.inference_mode():
            for batch in self._pil_batches(images_bgr):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self._tensor(self.model.get_image_features(**inputs))
                feats.append(out.float().cpu().numpy())
        return _l2(np.concatenate(feats)) if feats else np.empty((0, self.dim), "float32")

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        # SigLIP was trained on lower-case text padded to 64 tokens.
        prompts = [self.query_template.format(t).lower() for t in texts]
        with self._torch.inference_mode():
            inputs = self.processor(
                text=prompts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=64,
            ).to(self.device)
            out = self._tensor(self.model.get_text_features(**inputs))
        return _l2(out.float().cpu().numpy())


class ClipEncoder(_TorchEncoder):
    def __init__(self, model_name: str, device: str, batch_size: int, query_template: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.batch_size = batch_size
        self.query_template = query_template
        self.processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        self.model = self._to_device(model, resolve_device(device))
        self.dim = int(model.config.projection_dim)
        self.logit_scale = float(model.logit_scale.exp())
        log.info("CLIP ready: %s on %s (dim=%d)", model_name, self.device, self.dim)

    def encode_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        feats: List[np.ndarray] = []
        with self._torch.inference_mode():
            for batch in self._pil_batches(images_bgr):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self._tensor(self.model.get_image_features(**inputs))
                feats.append(out.float().cpu().numpy())
        return _l2(np.concatenate(feats)) if feats else np.empty((0, self.dim), "float32")

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        prompts = [self.query_template.format(t) for t in texts]
        with self._torch.inference_mode():
            inputs = self.processor(
                text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            out = self._tensor(self.model.get_text_features(**inputs))
        return _l2(out.float().cpu().numpy())


_COLORS = {
    "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "yellow": (1, 1, 0),
    "white": (1, 1, 1), "black": (0, 0, 0), "orange": (1, 0.5, 0), "gray": (.5, .5, .5),
    "grey": (.5, .5, .5),
}


class DummyEncoder:
    """Colour-only toy encoder. Lets the whole pipeline run offline (tests/demos).

    Images -> mean colour of a 2x2 grid; text -> average of any colour words found.
    NOT semantically meaningful: use SigLIP/CLIP for real work.
    """

    dim = 12

    def encode_images(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        vecs = []
        for im in images_bgr:
            small = cv2.resize(im, (2, 2), interpolation=cv2.INTER_AREA)[..., ::-1] / 255.0
            vecs.append((small - 0.5).reshape(-1))
        return _l2(np.array(vecs, dtype="float32")) if vecs else np.empty((0, self.dim), "float32")

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            found = [np.array(rgb) - 0.5 for w, rgb in _COLORS.items() if w in t.lower().split()]
            rgb = np.mean(found, axis=0) if found else np.zeros(3)
            out.append(np.tile(rgb, 4))
        return _l2(np.array(out, dtype="float32"))


def build_encoder(cfg: SimpleNamespace):
    e = cfg.embedding
    if e.backend == "siglip":
        return SiglipEncoder(e.model, e.device, e.batch_size, e.query_template)
    if e.backend == "clip":
        model = e.model if "clip" in e.model.lower() else "openai/clip-vit-base-patch32"
        return ClipEncoder(model, e.device, e.batch_size, e.query_template)
    if e.backend == "dummy":
        log.warning("Using DummyEncoder (colour-only) - results are not semantic")
        return DummyEncoder()
    raise ValueError(f"Unsupported embedding backend: {e.backend}")


# --------------------------------------------------------------------------
# Colour measurement (HSV rules; OpenCV hue range is 0-179)
# --------------------------------------------------------------------------
def _region_color(region: np.ndarray) -> Optional[str]:
    if region is None or region.size == 0:
        return None
    small = cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    black = v < 55
    white = (s < 45) & (v >= 190) & ~black
    gray = (s < 45) & ~white & ~black
    red_hue = (h < 8) | (h >= 172)
    conds = [
        black, white, gray,
        (h >= 8) & (h < 25) & (v < 150),            # brown (dark orange/yellow)
        (h >= 8) & (h < 20),                        # orange
        (h >= 20) & (h < 35),                       # yellow
        (h >= 35) & (h < 85),                       # green
        (h >= 85) & (h < 130),                      # blue (incl. cyan)
        (h >= 130) & (h < 150),                     # purple
        ((h >= 150) & (h < 172)) | (red_hue & (s < 110) & (v >= 150)),  # pink
        red_hue,                                    # red
    ]
    names = ["black", "white", "gray", "brown", "orange", "yellow", "green", "blue",
             "purple", "pink", "red"]
    idx = np.select(conds, list(range(len(names))), default=2)
    return names[int(np.bincount(idx, minlength=len(names)).argmax())]


def _regions(crop_bgr: np.ndarray, class_name: str):
    """(upper, lower) regions of a person crop; (body, None) for anything else."""
    h, w = crop_bgr.shape[:2]
    cx1, cx2 = int(w * 0.2), max(int(w * 0.8), int(w * 0.2) + 1)
    if class_name == "person":
        upper = crop_bgr[int(h * 0.18): max(int(h * 0.50), int(h * 0.18) + 1), cx1:cx2]
        lower = crop_bgr[int(h * 0.55): max(int(h * 0.92), int(h * 0.55) + 1), cx1:cx2]
        return upper, lower
    bx1, bx2 = int(w * 0.15), max(int(w * 0.85), int(w * 0.15) + 1)
    return crop_bgr[int(h * 0.30): max(int(h * 0.85), int(h * 0.30) + 1), bx1:bx2], None


def color_names(crop_bgr: np.ndarray, class_name: str) -> Tuple[Optional[str], Optional[str]]:
    """(upper, lower) colours for a person; (body, None) for anything else.

    For non-person classes the body colour is stored in the `upper_color` field.
    """
    upper, lower = _regions(crop_bgr, class_name)
    return _region_color(upper), (_region_color(lower) if lower is not None else None)


def _region_tone(region: Optional[np.ndarray]) -> Optional[str]:
    """dark / mid / light from median brightness. Works on colourless footage."""
    if region is None or region.size == 0:
        return None
    gray = cv2.cvtColor(cv2.resize(region, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m = float(np.median(gray))
    return "dark" if m < 80 else ("light" if m >= 160 else "mid")


def tone_names(crop_bgr: np.ndarray, class_name: str) -> Tuple[Optional[str], Optional[str]]:
    """(upper, lower) brightness tone; (body, None) for non-person classes."""
    upper, lower = _regions(crop_bgr, class_name)
    return _region_tone(upper), (_region_tone(lower) if lower is not None else None)

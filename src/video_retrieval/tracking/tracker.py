"""Multi-object tracking: ByteTrack (via `supervision`) + a simple IoU fallback."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from video_retrieval.detection.detector import Detection
from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrackedObject:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class ByteTrackTracker:
    """Wraps supervision's ByteTrack. `frame_rate` must be the *processed* fps."""

    def __init__(
        self,
        frame_rate: float,
        activation_threshold: float = 0.35,
        lost_track_buffer: int = 30,
        matching_threshold: float = 0.8,
        min_consecutive_frames: int = 1,
    ):
        import supervision as sv

        if not hasattr(sv, "ByteTrack"):  # removed in supervision >= 0.30
            raise ImportError("supervision.ByteTrack unavailable; install supervision<0.30")
        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=matching_threshold,
            frame_rate=max(1, int(round(frame_rate))),
            minimum_consecutive_frames=min_consecutive_frames,
        )
        self._names: Dict[int, str] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        sv = self._sv
        if not detections:
            sv_det = sv.Detections.empty()
        else:
            for d in detections:
                self._names[d.class_id] = d.class_name
            sv_det = sv.Detections(
                xyxy=np.array([d.xyxy for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.array([d.class_id for d in detections], dtype=int),
            )
        tracked = self._tracker.update_with_detections(sv_det)
        out: List[TrackedObject] = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None or tid < 0:
                continue
            cid = int(tracked.class_id[i])
            out.append(
                TrackedObject(
                    int(tid),
                    tuple(map(float, tracked.xyxy[i])),
                    float(tracked.confidence[i]),
                    cid,
                    self._names.get(cid, str(cid)),
                )
            )
        return out


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Greedy same-class IoU matching. Dependency-free fallback; weaker than ByteTrack."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold, self.max_age = iou_threshold, max_age
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        pairs = []
        for tid, t in self._tracks.items():
            for di, d in enumerate(detections):
                if d.class_id == t["class_id"]:
                    score = _iou(t["xyxy"], d.xyxy)
                    if score >= self.iou_threshold:
                        pairs.append((score, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d, matched = set(), set(), {}
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            matched[di] = tid

        for tid in list(self._tracks):
            if tid not in used_t:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        out: List[TrackedObject] = []
        for di, d in enumerate(detections):
            tid = matched.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = {"xyxy": d.xyxy, "class_id": d.class_id, "age": 0}
            out.append(TrackedObject(tid, d.xyxy, d.confidence, d.class_id, d.class_name))
        return out


def build_tracker(cfg: SimpleNamespace, fps: float, stride: int = 1):
    t = cfg.tracking
    effective_fps = fps / max(1, stride)
    if t.backend == "bytetrack":
        try:
            return ByteTrackTracker(
                frame_rate=effective_fps,
                activation_threshold=t.activation_threshold,
                lost_track_buffer=t.lost_track_buffer,
                matching_threshold=t.matching_threshold,
                min_consecutive_frames=t.min_consecutive_frames,
            )
        except ImportError:
            log.warning(
                "ByteTrack unavailable (pip install 'supervision>=0.22,<0.30') - "
                "using the simple IoU tracker instead"
            )
    return IoUTracker(max_age=t.lost_track_buffer)

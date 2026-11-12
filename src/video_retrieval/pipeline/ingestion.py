"""Ingestion: video/stream -> frames -> (motion gate) -> detection -> tracking
-> one averaged embedding per track -> stores."""
from __future__ import annotations

import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_retrieval.database.metadata_store import MetadataStore
from video_retrieval.database.vector_store import VectorStore
from video_retrieval.embedding.encoder import color_names
from video_retrieval.tracking.tracker import build_tracker
from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")


@dataclass
class IngestResult:
    video_id: int
    frames_processed: int
    observations: int
    tracks: int
    duration_sec: float
    frames_skipped_static: int = 0


@dataclass
class _Crop:
    quality: float
    img: np.ndarray
    ts: float
    frame_idx: int
    xyxy: Tuple[float, float, float, float]
    conf: float


@dataclass
class _TrackBuf:
    """Collects the sharpest crops of one track; becomes ONE observation when finished."""

    track_id: int
    t_start: float
    k: int
    t_end: float = 0.0
    last_seen: float = 0.0
    crops: List[_Crop] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)
    confs: List[float] = field(default_factory=list)

    def add(self, ts: float, class_name: str, conf: float, crop: Optional[_Crop]) -> None:
        self.t_end = self.last_seen = ts
        self.classes[class_name] += 1
        self.confs.append(conf)
        if crop is None:
            return
        if len(self.crops) < self.k:
            self.crops.append(crop)
        else:
            worst = min(range(len(self.crops)), key=lambda i: self.crops[i].quality)
            if crop.quality > self.crops[worst].quality:
                self.crops[worst] = crop


@dataclass
class _Pending:
    images: List[np.ndarray]  # crops to embed; the embedding is their mean
    row: Dict
    thumb: np.ndarray


def _open(source: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def _quality(crop: np.ndarray, conf: float) -> float:
    """Bigger, sharper, more confident crops score higher."""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(conf * np.sqrt(h * w) * np.log1p(sharp))


class IngestionPipeline:
    def __init__(
        self,
        cfg: SimpleNamespace,
        metadata: MetadataStore,
        vectors: VectorStore,
        detector,
        encoder,
    ):
        self.cfg = cfg
        self.metadata = metadata
        self.vectors = vectors
        self.detector = detector
        self.encoder = encoder
        self.thumb_dir = Path(cfg.paths.thumbnails)

    # ------------------------------------------------------------- helpers
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        max_side = self.cfg.ingestion.max_side
        if not max_side:
            return frame
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale >= 1:
            return frame
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _crop_view(self, frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
        ing = self.cfg.ingestion
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw, ph = (x2 - x1) * ing.crop_padding, (y2 - y1) * ing.crop_padding
        x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
        if min(x2 - x1, y2 - y1) < ing.min_crop_size:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _motion_signature(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (160, max(1, int(160 * frame.shape[0] / frame.shape[1]))))
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def _save_thumb(self, video_id: int, name: str, img: np.ndarray, max_side: int) -> str:
        rel = f"{video_id}/{name}.jpg"
        path = self.thumb_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return rel

    def _finalize(self, video_id: int, tb: _TrackBuf) -> Optional[_Pending]:
        if not tb.crops:
            return None
        best = max(tb.crops, key=lambda c: c.quality)
        cls = tb.classes.most_common(1)[0][0]
        # colour = most common reading across the kept crops (robust to one bad frame)
        readings = [color_names(c.img, cls) for c in tb.crops]
        upper = Counter(u for u, _ in readings if u).most_common(1)
        lower = Counter(l for _, l in readings if l).most_common(1)
        x1, y1, x2, y2 = best.xyxy
        row = dict(
            video_id=video_id, kind="object", track_id=tb.track_id, class_name=cls,
            confidence=float(np.mean(tb.confs)), frame_idx=best.frame_idx,
            timestamp_sec=float(best.ts), x1=x1, y1=y1, x2=x2, y2=y2,
            t_start=float(tb.t_start), t_end=float(max(tb.t_end, tb.t_start)),
            upper_color=upper[0][0] if upper else None,
            lower_color=lower[0][0] if lower else None,
        )
        return _Pending([c.img for c in tb.crops], row, best.img)

    def _flush(self, video_id: int, pending: List[_Pending]) -> int:
        if not pending:
            return 0
        all_imgs = [im for p in pending for im in p.images]
        raw = self.encoder.encode_images(all_imgs)
        vecs, off = [], 0
        for p in pending:  # mean of each item's crop embeddings, re-normalised
            n = len(p.images)
            m = raw[off: off + n].mean(axis=0)
            vecs.append(m / max(float(np.linalg.norm(m)), 1e-8))
            off += n
        rows = []
        for p in pending:
            row = dict(p.row)
            if self.cfg.ingestion.save_thumbnails:
                is_obj = row["kind"] == "object"
                row["thumbnail"] = self._save_thumb(
                    video_id,
                    f"{row['kind']}_{row['track_id'] or 0}_{row['frame_idx']}",
                    p.thumb,
                    max_side=160 if is_obj else 320,
                )
            rows.append(row)
        ids = self.metadata.add_observations(rows)
        self.vectors.add(ids, np.array(vecs, dtype="float32"))
        return len(ids)

    # ---------------------------------------------------------------- main
    def ingest(
        self,
        source: str,
        camera_id: str = "cam-01",
        start_time: Optional[datetime] = None,
        stop_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[int, int, int], None]] = None,
        max_frames: Optional[int] = None,
    ) -> IngestResult:
        ing = self.cfg.ingestion
        is_stream = source.lower().startswith(STREAM_PREFIXES) or source.isdigit()
        if not is_stream and not os.path.isfile(source):
            raise FileNotFoundError(f"Video not found: {source}")

        cap = _open(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
        stride = max(1, ing.frame_stride)

        video_id = self.metadata.add_video(
            path=source,
            name=Path(source).name if not is_stream else source,
            camera_id=camera_id,
            fps=float(fps),
            width=width,
            height=height,
            frame_count=max(total, 0),
            duration_sec=(total / fps) if total > 0 else 0.0,
            start_time=start_time or (datetime.now().replace(microsecond=0) if is_stream else None),
            status="processing",
        )
        log.info("Ingesting %s (video_id=%d, %.1f fps, stride=%d)", source, video_id, fps, stride)

        tracker = build_tracker(self.cfg, fps, stride)
        tracks: Dict[int, _TrackBuf] = {}
        pending: List[_Pending] = []
        n_tracks, n_obs, n_proc, n_skipped, idx, last_ts = 0, 0, 0, 0, -1, 0.0
        last_frame_emb = -1e9
        ref_sig: Optional[np.ndarray] = None
        last_detect_ts = -1e9
        t0 = time.monotonic()
        reconnects = 0
        status = "done"

        def finish(tb: _TrackBuf) -> None:
            item = self._finalize(video_id, tb)
            if item is not None:
                pending.append(item)

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    status = "stopped"
                    break
                if max_frames is not None and n_proc >= max_frames:
                    break

                if not cap.grab():
                    # Streams drop; files simply end.
                    if is_stream and reconnects < ing.stream_reconnect_attempts:
                        reconnects += 1
                        log.warning("Stream read failed, reconnecting (%d)", reconnects)
                        cap.release()
                        time.sleep(min(2 * reconnects, 10))
                        cap = _open(source)
                        continue
                    break
                idx += 1
                if idx % stride:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:  # corrupted frame -> skip (data cleaning)
                    continue

                ts = (time.monotonic() - t0) if is_stream else idx / fps
                last_ts = ts
                frame = self._resize(frame)
                n_proc += 1

                # ---- motion gate: skip YOLO when nothing changed since the last detection
                do_detect = True
                if ing.motion_skip:
                    sig = self._motion_signature(frame)
                    if ref_sig is not None and ts - last_detect_ts < ing.force_detect_every_sec:
                        changed = float((cv2.absdiff(sig, ref_sig) > 25).mean())
                        if changed < ing.motion_min_area:
                            do_detect = False
                    if do_detect:
                        ref_sig = sig
                if not do_detect:
                    n_skipped += 1
                else:
                    last_detect_ts = ts
                    for obj in tracker.update(self.detector.detect(frame)):
                        tb = tracks.get(obj.track_id)
                        if tb is not None and ts - tb.t_start > ing.max_track_sec:
                            finish(tb)  # split very long tracks
                            tb = None
                        if tb is None:
                            # The tracker recycles ids once a track dies, so give every
                            # track its own id (unique within this video).
                            n_tracks += 1
                            tb = tracks[obj.track_id] = _TrackBuf(n_tracks, ts, ing.crops_per_track)
                        view = self._crop_view(frame, obj.xyxy)
                        crop = None
                        if view is not None:
                            q = _quality(view, obj.confidence)
                            if len(tb.crops) < tb.k or q > min(c.quality for c in tb.crops):
                                crop = _Crop(q, view.copy(), ts, idx, obj.xyxy, obj.confidence)
                        tb.add(ts, obj.class_name, obj.confidence, crop)

                # ---- close tracks that haven't been seen for a while
                for tid in [t for t, b in tracks.items() if ts - b.last_seen > ing.track_timeout_sec]:
                    finish(tracks.pop(tid))

                # ---- scene-level embedding for context
                if ts - last_frame_emb >= ing.frame_embed_every_n_sec:
                    last_frame_emb = ts
                    pending.append(
                        _Pending(
                            [frame],
                            dict(
                                video_id=video_id, kind="frame", track_id=None,
                                class_name="scene", confidence=1.0, frame_idx=idx,
                                timestamp_sec=float(ts), t_start=float(ts), t_end=float(ts),
                            ),
                            frame,
                        )
                    )

                if len(pending) >= ing.batch_size:
                    n_obs += self._flush(video_id, pending)
                    pending = []
                    self.vectors.save()
                if progress_cb and n_proc % 25 == 0:
                    progress_cb(video_id, n_proc, n_obs + len(pending))

            for tb in list(tracks.values()):
                finish(tb)
            n_obs += self._flush(video_id, pending)
            self.vectors.save()
            self.metadata.update_video(
                video_id,
                status=status,
                frame_count=idx + 1,
                duration_sec=(idx + 1) / fps if not is_stream else last_ts,
            )
        except Exception:
            self.metadata.update_video(video_id, status="failed")
            raise
        finally:
            cap.release()

        if progress_cb:
            progress_cb(video_id, n_proc, n_obs)
        log.info(
            "Finished video %d: %d frames processed (%d static frames skipped), %d tracks, %d embeddings",
            video_id, n_proc, n_skipped, n_tracks, n_obs,
        )
        return IngestResult(video_id, n_proc, n_obs, n_tracks, last_ts, n_skipped)

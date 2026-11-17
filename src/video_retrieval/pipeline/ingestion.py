"""Ingestion: any video/stream -> (convert to H.264 if needed) -> frames -> night handling
-> motion gate -> detection -> tracking -> one averaged embedding per track -> stores."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
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
from video_retrieval.embedding.encoder import color_names, tone_names
from video_retrieval.tracking.tracker import build_tracker
from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")


# --------------------------------------------------------------------------
# ffmpeg helpers: support (almost) any video format by converting to H.264 MP4
# --------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    """System ffmpeg if installed, else the one bundled with `pip install imageio-ffmpeg`."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


def probe_fps(ffmpeg: str, src: str) -> float:
    """Frame rate as reported by ffmpeg (falls back to 25)."""
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", src], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Video:" in line:
            m = _FPS_RE.search(line)
            if m:
                return float(m.group(1))
    return 25.0


def is_readable(path: str) -> bool:
    """True if OpenCV can open the file and decode its first frame."""
    cap = cv2.VideoCapture(path)
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


def normalize_video(
    src: str, dst_dir: Path, max_side: int = 1280, stop_event: Optional[threading.Event] = None
) -> Path:
    """Convert any ffmpeg-readable video into a constant-frame-rate H.264 MP4 (no audio)."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "This video format needs converting, but ffmpeg was not found. "
            "Run: pip install imageio-ffmpeg"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{Path(src).resolve()}|{os.path.getsize(src)}|{os.path.getmtime(src)}|{max_side}".encode()
    ).hexdigest()[:10]
    dst = dst_dir / f"{Path(src).stem}_{key}.mp4"
    if dst.exists() and dst.stat().st_size > 0:  # converted before
        return dst

    fps = min(max(probe_fps(ffmpeg, src), 5.0), 30.0)
    if max_side:
        m = int(max_side)
        vf = f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))'"
    else:
        vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    tmp = dst.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", vf, "-r", f"{fps:g}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    log.info("Converting %s to H.264 MP4 (%.1f fps)...", Path(src).name, fps)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop_event is not None and stop_event.is_set():
                proc.kill()
                tmp.unlink(missing_ok=True)
                raise RuntimeError("Conversion stopped")
    err = (proc.stderr.read() or "").strip().splitlines()
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        detail = err[-1] if err else "unknown error"
        raise RuntimeError(f"Could not read this video (unsupported or corrupt file): {detail}")
    os.replace(tmp, dst)
    return dst


# --------------------------------------------------------------------------
# Night / infrared helpers
# --------------------------------------------------------------------------
def frame_stats(frame: np.ndarray) -> Tuple[float, float, float]:
    """(mean saturation 0-255, mean brightness 0-255, fraction of colourful pixels).

    Infrared / night-mode footage is colourless: mean saturation ~0 AND almost no colourful
    pixels. A grey but genuinely coloured scene (rain, fog) still has some colourful pixels.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (80, max(1, int(80 * h / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lit = hsv[..., 2] > 25  # ignore pure-black pixels
    sat = float(hsv[..., 1][lit].mean()) if lit.any() else 0.0
    colourful = float(((hsv[..., 1] > 45) & (hsv[..., 2] > 40)).mean())
    return sat, float(hsv[..., 2].mean()), colourful


_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_GAMMA_LUT = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)], dtype="uint8")


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Local contrast (CLAHE) + gamma on the luminance channel; helps YOLO/embeddings at night."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    if l.mean() < 90:
        l = cv2.LUT(l, _GAMMA_LUT)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
@dataclass
class IngestResult:
    video_id: int
    frames_processed: int
    observations: int
    tracks: int
    duration_sec: float
    frames_skipped_static: int = 0
    night_observations: int = 0


@dataclass
class _Crop:
    quality: float
    img: np.ndarray  # (enhanced) crop that gets embedded
    raw: np.ndarray  # untouched crop used to measure colour / tone
    ts: float
    frame_idx: int
    xyxy: Tuple[float, float, float, float]
    conf: float
    night: bool


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


def _mode(values) -> Optional[str]:
    c = Counter(v for v in values if v)
    return c.most_common(1)[0][0] if c else None


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

    def _prepare_source(
        self, source: str, stage_cb: Optional[Callable[[str], None]], stop_event
    ) -> Tuple[str, str]:
        """Return (path to open, name to display). Converts odd formats to H.264 MP4."""
        ing = self.cfg.ingestion
        name = Path(source).name
        mode = str(getattr(ing, "normalize", "auto"))
        if mode == "never" or (mode == "auto" and is_readable(source)):
            return source, name
        if stage_cb:
            stage_cb("converting")
        out = normalize_video(
            source, Path(self.cfg.paths.uploads) / "normalized",
            int(getattr(ing, "normalize_max_side", 1280)), stop_event,
        )
        return str(out), name

    def _finalize(self, video_id: int, tb: _TrackBuf) -> Optional[_Pending]:
        if not tb.crops:
            return None
        best = max(tb.crops, key=lambda c: c.quality)
        cls = tb.classes.most_common(1)[0][0]
        is_night = sum(c.night for c in tb.crops) * 2 > len(tb.crops)
        # measured on the untouched crops (enhancement would shift colour / brightness)
        tones = [tone_names(c.raw, cls) for c in tb.crops]
        upper_color = lower_color = None
        if not is_night:  # colourless footage has no usable colour
            cols = [color_names(c.raw, cls) for c in tb.crops]
            upper_color, lower_color = _mode(u for u, _ in cols), _mode(l for _, l in cols)
        x1, y1, x2, y2 = best.xyxy
        row = dict(
            video_id=video_id, kind="object", track_id=tb.track_id, class_name=cls,
            confidence=float(np.mean(tb.confs)), frame_idx=best.frame_idx,
            timestamp_sec=float(best.ts), x1=x1, y1=y1, x2=x2, y2=y2,
            t_start=float(tb.t_start), t_end=float(max(tb.t_end, tb.t_start)),
            upper_color=upper_color, lower_color=lower_color,
            upper_tone=_mode(u for u, _ in tones), lower_tone=_mode(l for _, l in tones),
            is_night=bool(is_night),
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
        stage_cb: Optional[Callable[[str], None]] = None,
    ) -> IngestResult:
        ing = self.cfg.ingestion
        is_stream = source.lower().startswith(STREAM_PREFIXES) or source.isdigit()
        if not is_stream and not os.path.isfile(source):
            raise FileNotFoundError(f"Video not found: {source}")

        display_name = source if is_stream else Path(source).name
        if not is_stream:
            source, display_name = self._prepare_source(source, stage_cb, stop_event)
        if stage_cb:
            stage_cb("running")

        cap = _open(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {display_name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else 0
        stride = max(1, ing.frame_stride)

        video_id = self.metadata.add_video(
            path=source,
            name=display_name,
            camera_id=camera_id,
            fps=float(fps),
            width=width,
            height=height,
            frame_count=max(total, 0),
            duration_sec=(total / fps) if total > 0 else 0.0,
            start_time=start_time or (datetime.now().replace(microsecond=0) if is_stream else None),
            status="processing",
        )
        log.info("Ingesting %s (video_id=%d, %.1f fps, stride=%d)", display_name, video_id, fps, stride)

        tracker = build_tracker(self.cfg, fps, stride)
        tracks: Dict[int, _TrackBuf] = {}
        pending: List[_Pending] = []
        n_tracks, n_obs, n_proc, n_skipped, idx, last_ts = 0, 0, 0, 0, -1, 0.0
        n_night = 0
        last_frame_emb = -1e9
        ref_sig: Optional[np.ndarray] = None
        last_detect_ts = -1e9
        sat_ema: Optional[float] = None
        frac_ema: Optional[float] = None
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

                # ---- is colour usable? (night / infrared footage is colourless)
                sat, lum, frac = frame_stats(frame)
                sat_ema = sat if sat_ema is None else 0.6 * sat_ema + 0.4 * sat
                frac_ema = frac if frac_ema is None else 0.6 * frac_ema + 0.4 * frac
                night = bool(
                    getattr(ing, "night_detect", True)
                    and sat_ema < ing.night_sat_threshold
                    and frac_ema < getattr(ing, "night_colour_fraction", 0.03)
                )
                enh_mode = str(getattr(ing, "enhance_low_light", "auto"))
                enhance = enh_mode == "always" or (enh_mode == "auto" and (night or lum < 60))

                # ---- motion gate: skip YOLO when nothing changed since the last detection
                do_detect = True
                if ing.motion_skip:
                    sig = self._motion_signature(frame)  # on the raw frame (CLAHE amplifies noise)
                    if ref_sig is not None and ts - last_detect_ts < ing.force_detect_every_sec:
                        changed = float((cv2.absdiff(sig, ref_sig) > 25).mean())
                        if changed < ing.motion_min_area:
                            do_detect = False
                    if do_detect:
                        ref_sig = sig

                frame_due = ts - last_frame_emb >= ing.frame_embed_every_n_sec
                work = enhance_low_light(frame) if (enhance and (do_detect or frame_due)) else frame

                if not do_detect:
                    n_skipped += 1
                else:
                    last_detect_ts = ts
                    dets = (self.detector.detect(work, conf=ing.night_confidence) if night
                            else self.detector.detect(work))
                    for obj in tracker.update(dets):
                        tb = tracks.get(obj.track_id)
                        if tb is not None and ts - tb.t_start > ing.max_track_sec:
                            finish(tb)  # split very long tracks
                            tb = None
                        if tb is None:
                            # The tracker recycles ids once a track dies, so give every
                            # track its own id (unique within this video).
                            n_tracks += 1
                            tb = tracks[obj.track_id] = _TrackBuf(n_tracks, ts, ing.crops_per_track)
                        view = self._crop_view(work, obj.xyxy)
                        crop = None
                        if view is not None:
                            q = _quality(view, obj.confidence)
                            if len(tb.crops) < tb.k or q > min(c.quality for c in tb.crops):
                                raw_view = self._crop_view(frame, obj.xyxy)
                                img = view.copy()
                                crop = _Crop(q, img, img if work is frame else raw_view.copy(),
                                             ts, idx, obj.xyxy, obj.confidence, night)
                        tb.add(ts, obj.class_name, obj.confidence, crop)

                # ---- close tracks that haven't been seen for a while
                for tid in [t for t, b in tracks.items() if ts - b.last_seen > ing.track_timeout_sec]:
                    finish(tracks.pop(tid))

                # ---- scene-level embedding for context
                if frame_due:
                    last_frame_emb = ts
                    pending.append(
                        _Pending(
                            [work],
                            dict(
                                video_id=video_id, kind="frame", track_id=None,
                                class_name="scene", confidence=1.0, frame_idx=idx,
                                timestamp_sec=float(ts), t_start=float(ts), t_end=float(ts),
                                is_night=night,
                            ),
                            work,
                        )
                    )

                if len(pending) >= ing.batch_size:
                    n_night += sum(1 for p in pending if p.row.get("is_night"))
                    n_obs += self._flush(video_id, pending)
                    pending = []
                    self.vectors.save()
                if progress_cb and n_proc % 25 == 0:
                    progress_cb(video_id, n_proc, n_obs + len(pending))

            for tb in list(tracks.values()):
                finish(tb)
            n_night += sum(1 for p in pending if p.row.get("is_night"))
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
            "Finished video %d: %d frames (%d static skipped), %d tracks, %d embeddings (%d night/IR)",
            video_id, n_proc, n_skipped, n_tracks, n_obs, n_night,
        )
        return IngestResult(video_id, n_proc, n_obs, n_tracks, last_ts, n_skipped, n_night)

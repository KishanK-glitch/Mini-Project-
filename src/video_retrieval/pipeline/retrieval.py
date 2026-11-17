"""Natural-language retrieval: text -> embedding -> vector search -> ranked time segments."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from video_retrieval.database.metadata_store import MetadataStore
from video_retrieval.database.vector_store import VectorStore
from video_retrieval.pipeline.ingestion import find_ffmpeg as _find_ffmpeg
from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)

# Query words -> detector classes, used for a small score boost (not a hard filter).
CLASS_KEYWORDS = {
    "person": {"person", "people", "man", "men", "woman", "women", "boy", "girl", "child",
              "pedestrian", "guy", "someone"},
    "bicycle": {"bicycle", "bike", "cyclist"},
    "car": {"car", "sedan", "suv", "taxi", "vehicle", "hatchback"},
    "motorcycle": {"motorcycle", "motorbike", "scooter", "bike"},
    "bus": {"bus"},
    "truck": {"truck", "lorry", "van"},
    "backpack": {"backpack", "rucksack", "bag"},
    "handbag": {"handbag", "purse", "bag"},
    "suitcase": {"suitcase", "luggage"},
}


COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown",
          "black", "white", "gray"]
_COLOR_ALIASES = {"grey": "gray"}


def colour_swaps(query: str) -> List[str]:
    """Copies of the query with each colour word replaced by every other colour.

    "person in a green hoodie" -> "person in a red hoodie", "... blue hoodie", ...
    Used as contrast prompts: if another colour fits a crop better, the crop is not a match.
    """
    words = query.lower().split()
    out: List[str] = []
    for i, w in enumerate(words):
        bare = w.strip(".,!?")
        colour = _COLOR_ALIASES.get(bare, bare)
        if colour in COLORS:
            for other in COLORS:
                if other != colour:
                    out.append(" ".join(words[:i] + [other] + words[i + 1:]))
    return out


_UPPER = {"shirt", "tshirt", "hoodie", "hoody", "jacket", "sweater", "sweatshirt", "top",
          "coat", "kurta", "blouse", "jersey", "tee", "shirts"}
_LOWER = {"pants", "pant", "trousers", "trouser", "jeans", "shorts", "skirt"}
_BODY = {"car", "truck", "bus", "van", "bike", "bicycle", "motorcycle", "scooter", "suv",
         "taxi", "vehicle", "lorry", "hatchback", "sedan"}


def parse_color_constraints(query: str) -> List[tuple]:
    """"green hoodie and blue jeans" -> [("upper","green"), ("lower","blue")].

    A colour followed (within 2 words) by a garment/vehicle word is tied to that region;
    a bare colour ("person wearing red") matches either upper or lower.
    """
    toks = query.lower().replace("t-shirt", "tshirt").replace(",", " ").split()
    toks = [t.strip(".,!?") for t in toks]
    out: List[tuple] = []
    for i, tok in enumerate(toks):
        colour = _COLOR_ALIASES.get(tok, tok)
        if colour not in COLORS:
            continue
        region = "any"
        for nxt in toks[i + 1: i + 3]:
            if nxt in _UPPER:
                region = "upper"; break
            if nxt in _LOWER:
                region = "lower"; break
            if nxt in _BODY:
                region = "body"; break
        out.append((region, colour))
    return out


def colour_ok(o: Dict[str, Any], constraints: List[tuple]) -> bool:
    up, lo = o.get("upper_color"), o.get("lower_color")
    for region, col in constraints:
        if region == "upper":
            ok = up == col
        elif region == "lower":
            ok = lo == col
        elif region == "body":  # vehicles keep their body colour in upper_color
            ok = o["class_name"] != "person" and up == col
        else:
            ok = col in (up, lo)
        if not ok:
            return False
    return True


_TONE_WORDS = {"dark": "dark", "light": "light", "bright": "light", "pale": "light"}


def _region_after(toks: List[str], i: int) -> str:
    for nxt in toks[i + 1: i + 3]:
        if nxt in _UPPER:
            return "upper"
        if nxt in _LOWER:
            return "lower"
        if nxt in _BODY:
            return "body"
    return "any"


def parse_tone_constraints(query: str) -> List[tuple]:
    """"dark jacket and light trousers" -> [("upper","dark"), ("lower","light")]."""
    toks = [t.strip(".,!?") for t in query.lower().replace("t-shirt", "tshirt").replace(",", " ").split()]
    out = []
    for i, tok in enumerate(toks):
        if tok in _TONE_WORDS:
            region = _region_after(toks, i)
            if region != "any":
                out.append((region, _TONE_WORDS[tok]))
    return out


_NIGHT_TONE = {"black": "dark", "white": "light"}


def derive_night_tones(colour_constraints: List[tuple]) -> List[tuple]:
    """On colourless footage only black/white survive as a brightness hint."""
    return [(region, _NIGHT_TONE[col]) for region, col in colour_constraints
            if col in _NIGHT_TONE and region in ("upper", "lower", "body")]


def tone_ok(o: Dict[str, Any], constraints: List[tuple]) -> bool:
    """Brightness match. Tolerant: 'mid' fits both dark and light (IR brightness is unreliable)."""
    for region, want in constraints:
        if region == "upper":
            have = o.get("upper_tone")
        elif region == "lower":
            have = o.get("lower_tone")
        else:  # body (vehicles)
            if o["class_name"] == "person":
                return False
            have = o.get("upper_tone")
        if have is None or (have != want and have != "mid"):
            return False
    return True


def strip_colours(query: str) -> str:
    """Remove colour and dark/light words: used for night/IR footage where colour is unusable."""
    drop = set(COLORS) | set(_COLOR_ALIASES) | set(_TONE_WORDS)
    kept = [w for w in query.split() if w.lower().strip(".,!?") not in drop]
    return " ".join(kept) if kept else query


def parse_query_classes(query: str) -> set:
    words = set(query.lower().replace(",", " ").split())
    return {cls for cls, kws in CLASS_KEYWORDS.items() if words & kws}


def fmt_timecode(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def extract_clip(video_path: str, start: float, end: float, out_path: Path) -> None:
    """Cut [start, end] seconds into an MP4. Uses ffmpeg (H.264) if present, else OpenCV.

    The OpenCV fallback writes MPEG-4 Part 2 ("mp4v"), which browsers cannot play,
    so install `imageio-ffmpeg` to get browser-playable H.264 clips.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", video_path,
               "-t", f"{end - start:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return
    log.warning("ffmpeg not found - clip will use mp4v and may not play in browsers "
                "(pip install imageio-ffmpeg)")
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(int((end - start) * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
    finally:
        cap.release()


class RetrievalPipeline:
    def __init__(self, cfg: SimpleNamespace, metadata: MetadataStore, vectors: VectorStore, encoder):
        self.cfg, self.metadata, self.vectors, self.encoder = cfg, metadata, vectors, encoder
        self._neg_cache: Dict[tuple, np.ndarray] = {}

    def _min_score(self) -> float:
        ms = self.cfg.retrieval.min_score
        if isinstance(ms, (int, float)):
            return float(ms)
        return float(getattr(ms, self.cfg.embedding.backend, 0.0))

    # --------------------------------------------------------------- search
    @staticmethod
    def _basic_ok(o, v, video_id, camera_id, start_time, end_time, wanted) -> bool:
        if video_id is not None and v["id"] != video_id:
            return False
        if camera_id is not None and v["camera_id"] != camera_id:
            return False
        if wanted and (o["kind"] != "object" or o["class_name"] not in wanted):
            return False
        if start_time or end_time:
            if v["start_time"] is None:
                return False
            abs_t = v["start_time"] + timedelta(seconds=o["timestamp_sec"])
            if (start_time and abs_t < start_time) or (end_time and abs_t > end_time):
                return False
        return True

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[int] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        r = self.cfg.retrieval
        top_k = top_k or r.top_k

        colour_on = getattr(r, "color_mode", "off") == "strict"
        colour_cons = parse_color_constraints(query) if colour_on else []
        tone_cons = parse_tone_constraints(query) if colour_on else []
        night_tone_cons = tone_cons + derive_night_tones(colour_cons)
        has_colour = bool(colour_cons or tone_cons)
        plain_q = strip_colours(query) if has_colour else query

        # Colourful footage is searched with the full query. Night/IR footage has no usable
        # colour, so it is searched with the colour words removed.
        qvec = self.encoder.encode_text([query])[0]
        sources = [(self.vectors.search(qvec, r.candidate_pool), False)]
        qvec_plain = qvec
        if has_colour:
            qvec_plain = self.encoder.encode_text([plain_q])[0]
            sources.append((self.vectors.search(qvec_plain, r.candidate_pool), True))

        ids = {i for hits, _ in sources for i, _ in hits}
        obs = self.metadata.get_observations(list(ids))
        videos = self.metadata.get_videos({o["video_id"] for o in obs.values()})

        wanted = {c.lower() for c in classes} if classes else None
        parsed = parse_query_classes(query)
        floor = self._min_score()
        groups: Dict[bool, list] = {False: [], True: []}
        for hits, night_group in sources:
            for oid, score in hits:
                o = obs.get(oid)
                if o is None or score < floor:
                    continue
                is_night = bool(o.get("is_night"))
                if has_colour and is_night != night_group:
                    continue
                v = videos[o["video_id"]]
                if not self._basic_ok(o, v, video_id, camera_id, start_time, end_time, wanted):
                    continue
                if has_colour:
                    if night_group:  # colourless: only brightness hints can be checked
                        if night_tone_cons and (o["kind"] != "object" or not tone_ok(o, night_tone_cons)):
                            continue
                    elif o["kind"] != "object" or not (
                        colour_ok(o, colour_cons) and tone_ok(o, tone_cons)
                    ):
                        continue  # colour measured from pixels must agree with the query
                if o["class_name"] in parsed:
                    score += r.class_boost
                groups[night_group].append((score, o, v, None))

        cands: list = []
        for night_group, items in groups.items():
            if not items:
                continue
            q_text, q_vec = (plain_q, qvec_plain) if night_group else (query, qvec)
            if getattr(r, "calibrate", False):
                items = self._calibrate(q_text, q_vec, items)
            cands += items

        segments = self._segments(cands, r.segment_gap_sec) if cands else []
        margin = getattr(r, "relative_margin", 0)

        def trim(segs: list) -> list:
            segs = sorted(segs, key=lambda seg: seg["score"], reverse=True)
            if segs and margin:
                cutoff = segs[0]["score"] * (1 - margin)
                segs = [x for x in segs if x["score"] >= cutoff]
            return segs

        if has_colour:
            # Scores from the full query (colour verified) and the colour-stripped query
            # (night/IR) are on different scales, so trim each group on its own and list the
            # colour-verified results first.
            day_segs = [x for x in segments if not x["best"].get("is_night")]
            night_segs = [x for x in segments if x["best"].get("is_night")]
            segments = trim(day_segs) + trim(night_segs)
        else:
            segments = trim(segments)
        results = [self._format(rank, seg) for rank, seg in enumerate(segments[:top_k], 1)]

        notes: List[str] = []
        n_night = sum(1 for x in results if x["is_night"])
        if has_colour and n_night:
            notes.append(
                f"Colour was ignored for {n_night} night/infrared result(s) - they were matched "
                "on shape, clothing type and brightness instead."
            )
        return {
            "query": query,
            "took_ms": round((time.perf_counter() - t0) * 1000, 2),
            "count": len(results),
            "results": results,
            "notes": notes,
        }

    def _contrast_embeddings(self, query: str) -> np.ndarray:
        r = self.cfg.retrieval
        texts = tuple(list(getattr(r, "negative_prompts", []) or []) + colour_swaps(query))
        if not texts:
            return np.empty((0, self.vectors.dim), "float32")
        if texts not in self._neg_cache:
            if len(self._neg_cache) > 256:
                self._neg_cache.clear()
            self._neg_cache[texts] = self.encoder.encode_text(list(texts))
        return self._neg_cache[texts]

    def _calibrate(self, query: str, qvec: np.ndarray, cands: list) -> list:
        """Turn raw similarity into P(query | crop) against contrast prompts; drop weak hits.

        CLIP always returns *something*; a raw score of 0.30 can be a perfect match for one
        query and noise for another. Comparing the query against swapped-colour / generic
        prompts for the *same crop* tells us whether the query is actually the best description.
        """
        r = self.cfg.retrieval
        neg = self._contrast_embeddings(query)
        if len(neg) == 0:
            return cands
        vecs = self.vectors.get_vectors([c[1]["id"] for c in cands])
        scale = float(getattr(self.encoder, "logit_scale", r.logit_scale))
        logits = np.concatenate([vecs @ qvec[:, None], vecs @ neg.T], axis=1) * scale
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = (probs[:, 0] / probs.sum(axis=1))
        kept = [(s, o, v, float(p)) for (s, o, v, _), p in zip(cands, probs)
                if p >= r.min_confidence]
        log.debug("calibration kept %d/%d hits for %r", len(kept), len(cands), query)
        return kept

    @staticmethod
    def _span(o: Dict[str, Any]) -> tuple:
        t0 = o["t_start"] if o.get("t_start") is not None else o["timestamp_sec"]
        t1 = o["t_end"] if o.get("t_end") is not None else o["timestamp_sec"]
        return t0, max(t0, t1)

    @classmethod
    def _segments(cls, cands: list, gap: float) -> List[Dict[str, Any]]:
        """Group hits per (video, track); merge spans of one track that are less than `gap` apart."""
        groups = defaultdict(list)
        for c in cands:
            o = c[1]
            key = (o["video_id"], o["track_id"] if o["kind"] == "object" else None)
            groups[key].append(c)

        segs = []
        for items in groups.values():
            items.sort(key=lambda c: cls._span(c[1])[0])
            runs, cur, cur_end = [], [items[0]], cls._span(items[0][1])[1]
            for it in items[1:]:
                s0, e0 = cls._span(it[1])
                if s0 - cur_end > gap:
                    runs.append(cur)
                    cur, cur_end = [it], e0
                else:
                    cur.append(it)
                    cur_end = max(cur_end, e0)
            runs.append(cur)
            for run in runs:
                best = max(run, key=lambda c: c[0])
                segs.append(
                    {
                        "score": best[0],
                        "best": best[1],
                        "video": best[2],
                        "start": min(cls._span(c[1])[0] for c in run),
                        "end": max(cls._span(c[1])[1] for c in run),
                        "n": len(run),
                        "prob": best[3],
                    }
                )
        return segs

    def _format(self, rank: int, seg: Dict[str, Any]) -> Dict[str, Any]:
        pad = self.cfg.retrieval.clip_padding_sec
        v, o = seg["video"], seg["best"]
        start = max(0.0, seg["start"] - pad)
        end = seg["end"] + pad
        if v["duration_sec"]:
            end = min(end, v["duration_sec"])
        abs_start = abs_end = None
        if v["start_time"] is not None:
            abs_start = v["start_time"] + timedelta(seconds=start)
            abs_end = v["start_time"] + timedelta(seconds=end)
        is_file = os.path.isfile(v["path"])
        return {
            "rank": rank,
            "score": round(float(seg["score"]), 4),
            "confidence": None if seg.get("prob") is None else round(seg["prob"], 4),
            "video_id": v["id"],
            "video_name": v["name"],
            "camera_id": v["camera_id"],
            "kind": o["kind"],
            "track_id": o["track_id"],
            "class_name": o["class_name"],
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "timecode": fmt_timecode(start),
            "start_time": abs_start,
            "end_time": abs_end,
            "num_matches": seg["n"],
            "upper_color": o.get("upper_color"),
            "lower_color": o.get("lower_color"),
            "upper_tone": o.get("upper_tone"),
            "lower_tone": o.get("lower_tone"),
            "is_night": bool(o.get("is_night")),
            "thumbnail_url": f"/thumbnails/{o['thumbnail']}" if o["thumbnail"] else None,
            "clip_url": f"/clips/{v['id']}?start={start:.2f}&end={end:.2f}" if is_file else None,
        }

    # ---------------------------------------------------------------- clips
    def get_clip(self, video_id: int, start: float, end: float) -> Path:
        v = self.metadata.get_video(video_id)
        if v is None or not os.path.isfile(v["path"]):
            raise FileNotFoundError("Source video is not available as a file")
        if end <= start:
            raise ValueError("end must be greater than start")
        out = Path(self.cfg.paths.clips) / f"v{video_id}_{int(start * 1000)}_{int(end * 1000)}.mp4"
        if not out.exists():
            extract_clip(v["path"], start, end, out)
        return out

    # ----------------------------------------------------------- evaluation
    def evaluate(self, ground_truth_file: str, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, Any]:
        """Top-k accuracy + latency against labelled queries.

        ground_truth_file: JSON list of
          {"query": "...", "video_id": 1 | "video": "name.mp4", "start": 12.0, "end": 20.0}
        (`start`/`end` in seconds; a hit overlaps that interval in the right video).
        """
        with open(ground_truth_file, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        max_k = max(ks)
        hits_at = {k: 0 for k in ks}
        latencies: List[float] = []
        for item in items:
            res = self.search(item["query"], top_k=max_k)
            latencies.append(res["took_ms"])
            for k in ks:
                for hit in res["results"][:k]:
                    same = (
                        hit["video_id"] == item["video_id"]
                        if "video_id" in item
                        else item.get("video", "") in hit["video_name"]
                    )
                    if same and hit["start_sec"] <= item["end"] and hit["end_sec"] >= item["start"]:
                        hits_at[k] += 1
                        break
        n = max(len(items), 1)
        lat = np.array(latencies) if latencies else np.zeros(1)
        return {
            "num_queries": len(items),
            "top_k_accuracy": {f"top{k}": round(hits_at[k] / n, 4) for k in ks},
            "latency_ms": {
                "mean": round(float(lat.mean()), 2),
                "p95": round(float(np.percentile(lat, 95)), 2),
            },
        }

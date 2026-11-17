"""Entry point: REST API + CLI for natural-language CCTV video retrieval.

    python main.py serve                         # start the API  (http://localhost:8000)
    python main.py ingest path/to/video.mp4      # index a video / RTSP stream
    python main.py search "person in a red shirt"
    python main.py evaluate ground_truth.json    # Top-k accuracy + latency
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from video_retrieval.api.schemas import (  # noqa: E402
    IngestRequest,
    JobStatus,
    SearchRequest,
    SearchResponse,
    VideoInfo,
)
from video_retrieval.database.metadata_store import MetadataStore  # noqa: E402
from video_retrieval.database.vector_store import VectorStore  # noqa: E402
from video_retrieval.detection.detector import build_detector  # noqa: E402
from video_retrieval.embedding.encoder import build_encoder  # noqa: E402
from video_retrieval.pipeline.ingestion import IngestionPipeline  # noqa: E402
from video_retrieval.pipeline.retrieval import RetrievalPipeline  # noqa: E402
from video_retrieval.utils.config import load_config  # noqa: E402
from video_retrieval.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("main")


# --------------------------------------------------------------------------
# Service container (heavy models are loaded lazily, once)
# --------------------------------------------------------------------------
class Services:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metadata = MetadataStore(cfg.database.metadata_url)
        self.vectors = VectorStore(cfg.database.vector_dir, cfg.database.vector_backend)
        self._encoder = None
        self._detector = None
        self._lock = threading.Lock()

    @property
    def encoder(self):
        with self._lock:
            if self._encoder is None:
                self._encoder = build_encoder(self.cfg)
            return self._encoder

    @property
    def detector(self):
        with self._lock:
            if self._detector is None:
                self._detector = build_detector(self.cfg)
            return self._detector

    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(
            self.cfg, self.metadata, self.vectors, self.detector, self.encoder
        )

    def retrieval(self) -> RetrievalPipeline:
        return RetrievalPipeline(self.cfg, self.metadata, self.vectors, self.encoder)


@dataclass
class Job:
    job_id: str
    source: str
    status: str = "queued"  # queued | converting | running | done | failed | stopped
    video_id: Optional[int] = None
    frames_processed: int = 0
    observations: int = 0
    error: Optional[str] = None
    stop: threading.Event = field(default_factory=threading.Event)

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id, status=self.status, source=self.source, video_id=self.video_id,
            frames_processed=self.frames_processed, observations=self.observations, error=self.error,
        )


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCTV Video Search</title>

    <style>
        * { box-sizing: border-box; }

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
            max-width: 950px;
            margin: 50px auto;
            padding: 20px;
        }

        h1 { font-size: 26px; margin-bottom: 8px; }
        h2 { font-size: 18px; margin: 0 0 14px 0; }

        .subtitle { color: #666; margin-bottom: 30px; }

        .card {
            background: white;
            padding: 22px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 22px;
        }

        label { display: block; font-weight: bold; margin-bottom: 8px; }

        select, input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 15px;
            margin-bottom: 18px;
            background: white;
        }

        .row { display: flex; gap: 16px; }
        .row > div { flex: 1; }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { background: #999; cursor: not-allowed; }

        .hint { color: #777; font-size: 13px; margin: -10px 0 16px 0; }

        .bar { height: 8px; background: #e5e7eb; border-radius: 4px; margin-top: 14px; overflow: hidden; display: none; }
        .bar > div { height: 100%; width: 0; background: #2563eb; transition: width .2s; }

        #upStatus { margin: 12px 0 0 0; color: #444; }
        #upStatus.error { color: #b91c1c; }

        #status { margin: 20px 0 6px 0; color: #555; }
        #notes { margin: 0 0 16px 0; color: #92400e; background: #fef3c7; border-radius: 6px; padding: 8px 12px; display: none; font-size: 14px; }

        .result {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 15px;
            display: flex;
            gap: 18px;
            align-items: center;
        }

        .result img {
            width: 180px;
            height: 115px;
            object-fit: cover;
            border-radius: 6px;
            cursor: zoom-in;
        }

        .details { line-height: 1.7; }
        .details small { color: #666; }

        .badge {
            display: inline-block;
            background: #1e293b;
            color: #fff;
            border-radius: 4px;
            padding: 1px 8px;
            font-size: 12px;
            margin-left: 8px;
            vertical-align: middle;
        }

        a { color: #2563eb; text-decoration: none; cursor: pointer; }

        /* Popup for enlarged image / clip player */
        #modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, .85);
            z-index: 10;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        #modal.open { display: flex; }
        #modalBody img, #modalBody video {
            max-width: 90vw;
            max-height: 85vh;
            border-radius: 8px;
            background: #000;
        }

        @media (max-width: 600px) {
            .row { flex-direction: column; gap: 0; }
            .result { flex-direction: column; align-items: flex-start; }
            .result img { width: 100%; height: auto; }
        }
    </style>
</head>

<body>

    <h1>CCTV Video Search</h1>
    <p class="subtitle">Upload surveillance footage, then search it using natural language.</p>

    <!-- 1. Upload -->
    <div class="card">
        <h2>1. Upload footage</h2>

        <label for="file">Video file</label>
        <input type="file" id="file">
        <p class="hint" id="fmtHint"></p>

        <div class="row">
            <div>
                <label for="upCamera">Camera name</label>
                <input type="text" id="upCamera" value="cam-01" list="cameraList">
                <datalist id="cameraList"></datalist>
            </div>
            <div>
                <label for="upStart">Recording start time (optional)</label>
                <input type="datetime-local" id="upStart">
            </div>
        </div>

        <button id="upBtn" onclick="startUpload()">Upload &amp; index</button>
        <div class="bar" id="bar"><div id="barFill"></div></div>
        <p id="upStatus"></p>
    </div>

    <!-- 2. Search -->
    <div class="card">
        <h2>2. Search</h2>

        <div class="row">
            <div>
                <label for="camera">CCTV camera</label>
                <select id="camera"><option value="">All CCTV cameras</option></select>
            </div>
            <div>
                <label for="videoSel">Video</label>
                <select id="videoSel"><option value="">All videos</option></select>
            </div>
        </div>

        <label for="query">Describe what you are looking for</label>
        <input type="text" id="query" placeholder="e.g. A person wearing a red shirt">

        <button id="searchBtn" onclick="searchVideos()">Search</button>
    </div>

    <p id="status"></p>
    <div id="notes"></div>
    <div id="results"></div>

    <div id="modal" onclick="closeModal()">
        <div id="modalBody" onclick="event.stopPropagation()"></div>
    </div>

    <script>
        const $ = id => document.getElementById(id);
        const sleep = ms => new Promise(r => setTimeout(r, ms));
        let uploadInfo = { extensions: [], max_mb: 0 };

        // ---------- popup: enlarged image or clip player ----------
        function openModal(kind, url) {
            $("modalBody").innerHTML = "";
            const el = document.createElement(kind === "video" ? "video" : "img");
            el.src = url;
            if (kind === "video") { el.controls = true; el.autoplay = true; el.muted = true; }
            el.onerror = () => {
                $("modalBody").innerHTML = '<p style="color:#fff">Could not load this file.</p>';
            };
            $("modalBody").appendChild(el);
            $("modal").classList.add("open");
        }
        function closeModal() { $("modal").classList.remove("open"); $("modalBody").innerHTML = ""; }
        document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

        // ---------- cameras + videos ----------
        async function loadFilters(selectVideoId) {
            try {
                const response = await fetch("/videos");
                if (!response.ok) throw new Error("Could not load cameras");
                const videos = await response.json();

                const prevCam = $("camera").value;
                const prevVid = $("videoSel").value;
                $("camera").innerHTML = '<option value="">All CCTV cameras</option>';
                $("videoSel").innerHTML = '<option value="">All videos</option>';
                $("cameraList").innerHTML = "";

                [...new Set(videos.map(v => v.camera_id).filter(Boolean))].forEach(c => {
                    const o = document.createElement("option");
                    o.value = c; o.textContent = c;
                    $("camera").appendChild(o);
                    const d = document.createElement("option");
                    d.value = c;
                    $("cameraList").appendChild(d);
                });
                videos.filter(v => v.status === "done" || v.status === "stopped").forEach(v => {
                    const o = document.createElement("option");
                    o.value = v.id;
                    o.textContent = v.name + " (" + v.camera_id + ")";
                    $("videoSel").appendChild(o);
                });

                $("camera").value = prevCam;
                $("videoSel").value = prevVid;
                if (selectVideoId) {
                    const v = videos.find(x => x.id === selectVideoId);
                    if (v) { $("camera").value = v.camera_id; $("videoSel").value = String(v.id); }
                }
            } catch (error) {
                $("status").textContent = error.message;
            }
        }

        async function loadUploadInfo() {
            try {
                const r = await fetch("/upload-info");
                uploadInfo = await r.json();
                $("file").accept = uploadInfo.extensions.join(",") + ",video/*";
                $("fmtHint").textContent = "Most video formats work (" +
                    uploadInfo.extensions.slice(0, 8).join(" ") + " ...). Max " +
                    uploadInfo.max_mb + " MB. Other formats are converted automatically.";
            } catch (e) { /* hint is optional */ }
        }

        // ---------- upload ----------
        function setUp(text, isError) {
            $("upStatus").textContent = text;
            $("upStatus").className = isError ? "error" : "";
        }
        function setBar(pct) {
            $("bar").style.display = "block";
            $("barFill").style.width = pct + "%";
        }
        function endUpload() { $("upBtn").disabled = false; }
        function failUpload(msg) { setUp(msg, true); $("bar").style.display = "none"; endUpload(); }

        function startUpload() {
            const file = $("file").files[0];
            if (!file) { setUp("Choose a video file first.", true); return; }
            const dot = file.name.lastIndexOf(".");
            const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
            if (uploadInfo.extensions.length && !uploadInfo.extensions.includes(ext)) {
                setUp("Unsupported file type " + (ext || "(none)") + ".", true); return;
            }
            if (uploadInfo.max_mb && file.size > uploadInfo.max_mb * 1024 * 1024) {
                setUp("File is larger than " + uploadInfo.max_mb + " MB.", true); return;
            }

            const fd = new FormData();
            fd.append("file", file);
            fd.append("camera_id", $("upCamera").value.trim() || "cam-01");
            if ($("upStart").value) fd.append("start_time", $("upStart").value);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/upload");
            xhr.upload.onprogress = e => {
                if (e.lengthComputable) {
                    const pct = Math.round(e.loaded / e.total * 100);
                    setBar(pct);
                    setUp("Uploading... " + pct + "%");
                }
            };
            xhr.onerror = () => failUpload("Network error while uploading.");
            xhr.onload = () => {
                let body = {};
                try { body = JSON.parse(xhr.responseText); } catch (e) { /* not json */ }
                if (xhr.status === 202) pollJob(body.job_id);
                else failUpload(body.detail || ("Upload failed (" + xhr.status + ")"));
            };
            $("upBtn").disabled = true;
            setBar(0);
            setUp("Uploading...");
            xhr.send(fd);
        }

        async function pollJob(jobId) {
            while (true) {
                let job;
                try { job = await (await fetch("/jobs/" + jobId)).json(); }
                catch (e) { failUpload("Lost connection to the server."); return; }

                if (job.status === "queued") setUp("Waiting in the queue...");
                else if (job.status === "converting") setUp("Converting the video to a standard format...");
                else if (job.status === "running") setUp("Indexing... " + job.frames_processed + " frames processed");
                else if (job.status === "done" || job.status === "stopped") {
                    setBar(100);
                    setUp("Done. " + job.observations + " searchable items indexed. You can search it now.");
                    await loadFilters(job.video_id);
                    endUpload();
                    return;
                } else { failUpload(job.error || ("Indexing " + job.status)); return; }
                await sleep(1500);
            }
        }

        // ---------- search ----------
        async function searchVideos() {
            const query = $("query").value.trim();
            if (!query) { $("status").textContent = "Please enter a search description."; return; }

            $("searchBtn").disabled = true;
            $("status").textContent = "Searching footage...";
            $("notes").style.display = "none";
            $("results").innerHTML = "";

            try {
                const response = await fetch("/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        query: query,
                        top_k: 12,
                        camera_id: $("camera").value || null,
                        video_id: $("videoSel").value ? parseInt($("videoSel").value) : null
                    })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.detail || "Search failed");

                $("status").textContent = data.count
                    ? `${data.count} results found in ${data.took_ms} ms`
                    : "No matching footage found.";
                if (data.notes && data.notes.length) {
                    $("notes").textContent = data.notes.join(" ");
                    $("notes").style.display = "block";
                }

                data.results.forEach(item => {
                    const card = document.createElement("div");
                    card.className = "result";

                    if (item.thumbnail_url) {
                        const image = document.createElement("img");
                        image.src = item.thumbnail_url;
                        image.alt = "Matching CCTV footage";
                        image.title = "Click to enlarge";
                        image.onclick = () => openModal("image", item.thumbnail_url);
                        card.appendChild(image);
                    }

                    const details = document.createElement("div");
                    details.className = "details";

                    const title = document.createElement("strong");
                    title.textContent = `#${item.rank} ${item.class_name}`;
                    if (item.is_night) {
                        const badge = document.createElement("span");
                        badge.className = "badge";
                        badge.textContent = "Night / IR";
                        title.appendChild(badge);
                    }

                    const camera = document.createElement("div");
                    camera.textContent = `Camera: ${item.camera_id || "Unknown"}`;
                    const video = document.createElement("div");
                    video.textContent = `Video: ${item.video_name}`;
                    const time = document.createElement("div");
                    time.textContent = `Time: ${item.timecode}`;

                    const look = document.createElement("div");
                    if (item.is_night) {
                        const tone = [item.upper_tone, item.lower_tone].filter(Boolean).join(" / ");
                        look.textContent = "Colour not available" + (tone ? ` (brightness: ${tone})` : "");
                    } else if (item.upper_color) {
                        look.textContent = "Colour: " + [item.upper_color, item.lower_color].filter(Boolean).join(" / ");
                    }

                    const score = document.createElement("small");
                    score.textContent = `Score: ${item.score}` +
                        (item.confidence != null ? ` · conf ${item.confidence}` : "");

                    details.append(title, camera, video, time);
                    if (look.textContent) details.appendChild(look);
                    details.appendChild(score);

                    if (item.clip_url) {
                        const link = document.createElement("a");
                        link.href = item.clip_url;
                        link.textContent = "Play clip";
                        link.onclick = e => { e.preventDefault(); openModal("video", item.clip_url); };
                        details.appendChild(document.createElement("br"));
                        details.appendChild(link);
                    }

                    card.appendChild(details);
                    $("results").appendChild(card);
                });
            } catch (error) {
                $("status").textContent = error.message;
            } finally {
                $("searchBtn").disabled = false;
            }
        }

        loadUploadInfo();
        loadFilters();
        $("query").addEventListener("keydown", e => { if (e.key === "Enter") searchVideos(); });
    </script>

</body>
</html>
"""


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
def create_app(config_path: Optional[str] = None) -> FastAPI:
    cfg = load_config(config_path)
    setup_logging(cfg.app.log_level)
    jobs: Dict[str, Job] = {}
    executor = ThreadPoolExecutor(max_workers=1)  # one ingestion at a time (GPU-friendly)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = Services(cfg)
        log.info("API ready (%d vectors indexed)", len(app.state.services.vectors))
        yield
        for job in jobs.values():
            job.stop.set()
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Video Retrieval API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(cfg.api.cors_origins),
        allow_methods=["*"], allow_headers=["*"],
    )
    app.mount("/thumbnails", StaticFiles(directory=cfg.paths.thumbnails), name="thumbnails")

    def svc() -> Services:
        return app.state.services

    def run_job(job: Job, req: IngestRequest) -> None:
        job.status = "running"
        try:
            def progress(video_id: int, frames: int, obs: int) -> None:
                job.video_id, job.frames_processed, job.observations = video_id, frames, obs

            def stage(name: str) -> None:  # queued -> converting -> running
                job.status = name

            res = svc().ingestion().ingest(
                req.source, req.camera_id, req.start_time, job.stop, progress, stage_cb=stage
            )
            job.video_id, job.frames_processed, job.observations = (
                res.video_id, res.frames_processed, res.observations,
            )
            job.status = "stopped" if job.stop.is_set() else "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("Ingestion failed")
            job.status, job.error = "failed", str(exc)

    def submit(req: IngestRequest) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], source=req.source)
        jobs[job.job_id] = job
        executor.submit(run_job, job, req)
        return job

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {"status": "ok", "vectors": len(svc().vectors), "videos": len(svc().metadata.list_videos())}

    # ---- ingestion
    @app.post("/ingest", response_model=JobStatus, status_code=202)
    def ingest(req: IngestRequest):
        return submit(req).to_status()

    upload_cfg = cfg.api.upload
    allowed_ext = {e.lower() for e in upload_cfg.allowed_extensions}
    max_bytes = int(upload_cfg.max_mb) * 1024 * 1024

    @app.get("/upload-info")
    def upload_info():
        return {"extensions": sorted(allowed_ext), "max_mb": int(upload_cfg.max_mb)}

    @app.post("/upload", response_model=JobStatus, status_code=202)
    def upload(
        file: UploadFile = File(...),
        camera_id: str = Form("cam-01"),
        start_time: Optional[datetime] = Form(None),
    ):
        name = Path(file.filename or "video").name
        ext = Path(name).suffix.lower()
        if ext not in allowed_ext:
            raise HTTPException(
                415, f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed_ext))}"
            )
        # one folder per upload, so the original file name is kept for display
        folder = Path(cfg.paths.uploads) / uuid.uuid4().hex[:8]
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, f"File is larger than {upload_cfg.max_mb} MB")
                    out.write(chunk)
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception:  # noqa: BLE001
            shutil.rmtree(folder, ignore_errors=True)
            log.exception("Saving upload failed")
            raise HTTPException(500, "Could not save the uploaded file")
        if size == 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, "The uploaded file is empty")
        req = IngestRequest(source=str(dest), camera_id=camera_id.strip() or "cam-01", start_time=start_time)
        return submit(req).to_status()

    @app.get("/jobs", response_model=List[JobStatus])
    def list_jobs():
        return [j.to_status() for j in jobs.values()]

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job")
        return jobs[job_id].to_status()

    @app.delete("/jobs/{job_id}", response_model=JobStatus)
    def stop_job(job_id: str):
        """Gracefully stop a running job (useful for endless RTSP streams)."""
        if job_id not in jobs:
            raise HTTPException(404, "Unknown job")
        jobs[job_id].stop.set()
        return jobs[job_id].to_status()

    # ---- videos
    @app.get("/videos", response_model=List[VideoInfo])
    def videos():
        return svc().metadata.list_videos()

    # ---- search
    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        return svc().retrieval().search(
            req.query, req.top_k, req.video_id, req.camera_id, req.start_time, req.end_time, req.classes
        )

    @app.get("/search", response_model=SearchResponse)
    def search_get(q: str = Query(..., min_length=2), top_k: int = Query(10, ge=1, le=100)):
        return svc().retrieval().search(q, top_k)

    @app.get("/clips/{video_id}")
    def clip(video_id: int, start: float = Query(..., ge=0), end: float = Query(...)):
        try:
            path = svc().retrieval().get_clip(video_id, start, end)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return FileResponse(path, media_type="video/mp4")

    return app


app = create_app()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Path to a YAML config file")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Run the REST API")
    s.add_argument("--host"); s.add_argument("--port", type=int); s.add_argument("--reload", action="store_true")

    i = sub.add_parser("ingest", help="Index a video file, RTSP URL or webcam index")
    i.add_argument("source"); i.add_argument("--camera-id", default="cam-01")
    i.add_argument("--start-time", type=datetime.fromisoformat, help="ISO time of first frame")
    i.add_argument("--max-frames", type=int)

    q = sub.add_parser("search", help="Search indexed footage")
    q.add_argument("query"); q.add_argument("--top-k", type=int, default=5)

    e = sub.add_parser("evaluate", help="Top-k accuracy + latency on labelled queries")
    e.add_argument("ground_truth")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.app.log_level)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("main:app", host=args.host or cfg.api.host, port=args.port or cfg.api.port, reload=args.reload)
        return

    services = Services(cfg)
    if args.cmd == "ingest":
        res = services.ingestion().ingest(
            args.source, args.camera_id, args.start_time, max_frames=args.max_frames
        )
        print(json.dumps(res.__dict__, indent=2))
    elif args.cmd == "search":
        out = services.retrieval().search(args.query, args.top_k)
        print(f"{out['count']} results in {out['took_ms']} ms")
        for h in out["results"]:
            print(f"#{h['rank']}  {h['score']:.3f}  {h['video_name']}  {h['timecode']}  "
                  f"[{h['start_sec']}s-{h['end_sec']}s]  {h['class_name']} track={h['track_id']}")
    elif args.cmd == "evaluate":
        print(json.dumps(services.retrieval().evaluate(args.ground_truth), indent=2))


if __name__ == "__main__":
    _cli()

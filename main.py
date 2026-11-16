"""Entry point: REST API + CLI for natural-language CCTV video retrieval.

    python main.py serve                         # start the API  (http://localhost:8000)
    python main.py ingest path/to/video.mp4      # index a video / RTSP stream
    python main.py search "person in a red shirt"
    python main.py evaluate ground_truth.json    # Top-k accuracy + latency
"""
from __future__ import annotations

import argparse
import json
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
    status: str = "queued"
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
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
            max-width: 950px;
            margin: 50px auto;
            padding: 20px;
        }

        h1 {
            font-size: 26px;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }

        .search-box {
            background: white;
            padding: 22px;
            border-radius: 10px;
            border: 1px solid #ddd;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
        }

        select, input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 15px;
            margin-bottom: 18px;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #999;
            cursor: not-allowed;
        }

        #status {
            margin: 20px 0;
            color: #555;
        }

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
        }

        .details {
            line-height: 1.7;
        }

        .details small {
            color: #666;
        }

        a {
            color: #2563eb;
            text-decoration: none;
        }

        @media (max-width: 600px) {
            .result {
                flex-direction: column;
                align-items: flex-start;
            }

            .result img {
                width: 100%;
                height: auto;
            }
        }
    </style>
</head>

<body>

    <h1>CCTV Video Search</h1>
    <p class="subtitle">
        Search surveillance footage using natural language.
    </p>

    <div class="search-box">

        <label for="camera">Select CCTV Camera</label>
        <select id="camera">
            <option value="">All CCTV Cameras</option>
        </select>

        <label for="query">Describe what you are looking for</label>
        <input
            type="text"
            id="query"
            placeholder="e.g. A person wearing a red shirt"
        >

        <button id="searchBtn" onclick="searchVideos()">
            Search
        </button>

    </div>

    <p id="status"></p>
    <div id="results"></div>

    <script>
        const cameraSelect = document.getElementById("camera");
        const queryInput = document.getElementById("query");
        const resultsDiv = document.getElementById("results");
        const statusText = document.getElementById("status");
        const searchBtn = document.getElementById("searchBtn");

        // Load available cameras from the backend
        async function loadCameras() {
            try {
                const response = await fetch("/videos");

                if (!response.ok) {
                    throw new Error("Could not load cameras");
                }

                const videos = await response.json();

                const cameras = [...new Set(
                    videos.map(video => video.camera_id).filter(Boolean)
                )];

                cameras.forEach(cameraId => {
                    const option = document.createElement("option");
                    option.value = cameraId;
                    option.textContent = cameraId;
                    cameraSelect.appendChild(option);
                });

            } catch (error) {
                statusText.textContent = error.message;
            }
        }

        // Search footage
        async function searchVideos() {
            const query = queryInput.value.trim();
            const cameraId = cameraSelect.value;

            if (!query) {
                statusText.textContent = "Please enter a search description.";
                return;
            }

            searchBtn.disabled = true;
            statusText.textContent = "Searching footage...";
            resultsDiv.innerHTML = "";

            try {
                const response = await fetch("/search", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        query: query,
                        top_k: 12,
                        camera_id: cameraId || null
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "Search failed");
                }

                statusText.textContent = data.count
                    ? `${data.count} results found in ${data.took_ms} ms`
                    : "No matching footage found.";

                data.results.forEach(item => {
                    const card = document.createElement("div");
                    card.className = "result";

                    if (item.thumbnail_url) {
                        const image = document.createElement("img");
                        image.src = item.thumbnail_url;
                        image.alt = "Matching CCTV footage";
                        card.appendChild(image);
                    }

                    const details = document.createElement("div");
                    details.className = "details";

                    const title = document.createElement("strong");
                    title.textContent = `#${item.rank} ${item.class_name}`;

                    const camera = document.createElement("div");
                    camera.textContent = `Camera: ${item.camera_id || "Unknown"}`;

                    const video = document.createElement("div");
                    video.textContent = `Video: ${item.video_name}`;

                    const time = document.createElement("div");
                    time.textContent = `Time: ${item.timecode}`;

                    const score = document.createElement("small");
                    score.textContent = `Score: ${item.score}`;

                    details.append(title, camera, video, time, score);

                    if (item.clip_url) {
                        const link = document.createElement("a");
                        link.href = item.clip_url;
                        link.target = "_blank";
                        link.textContent = "Play clip";
                        details.appendChild(document.createElement("br"));
                        details.appendChild(link);
                    }

                    card.appendChild(details);
                    resultsDiv.appendChild(card);
                });

            } catch (error) {
                statusText.textContent = error.message;
            } finally {
                searchBtn.disabled = false;
            }
        }

        loadCameras();

        queryInput.addEventListener("keydown", event => {
            if (event.key === "Enter") {
                searchVideos();
            }
        });
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

            res = svc().ingestion().ingest(
                req.source, req.camera_id, req.start_time, job.stop, progress
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

    @app.post("/upload", response_model=JobStatus, status_code=202)
    def upload(
        file: UploadFile = File(...),
        camera_id: str = Form("cam-01"),
        start_time: Optional[datetime] = Form(None),
    ):
        dest = Path(cfg.paths.uploads) / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'video').name}"
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                out.write(chunk)
        return submit(IngestRequest(source=str(dest), camera_id=camera_id, start_time=start_time)).to_status()

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

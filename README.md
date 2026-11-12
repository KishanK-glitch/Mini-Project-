# Natural Language-Based Video Retrieval for Surveillance using Vision-Language Models

Department of Artificial Intelligence and Machine Learning
**Team:** Kishan K (4CB24AI045) · Ananya (4CB24AI005) · Bhawshik (4CB24AI016)
**Guide:** Mr. Nagaraj Gadagin, Asst. Professor

Search large CCTV archives with plain English, e.g. *"a person wearing a red shirt near the main gate"*,
and get back the matching **timestamps and video clips** instead of reviewing footage by hand.

## How it works

```
 CCTV file / RTSP ──► frame extraction ──► YOLO detection ──► ByteTrack tracking
                                                                   │
                                  crops + full frames ◄────────────┘
                                          │
                         SigLIP embeddings (mean of the sharpest crops per track)
                                          │
                  ┌───────────────────────┴──────────────────────┐
                  ▼                                              ▼
       Metadata DB (SQLite / PostgreSQL)                  Vector index (FAISS)
       videos, tracks, timestamps, class                  one vector per observation
                  └───────────────────────┬──────────────────────┘
                                          │
  "red shirt near the gate" ─► SigLIP text embedding ─► similarity search
                                          │
                  group by track, merge nearby hits ─► ranked segments + clips
```

| Stage (slide 8) | Module |
|---|---|
| 1. Data collection & preprocessing | `pipeline/ingestion.py` (frame stride, resize, skips corrupt frames) |
| 2. Detection & tracking | `detection/detector.py` (YOLO11, OpenVINO), `tracking/tracker.py` (ByteTrack) |
| 3. Feature extraction & metadata | `embedding/encoder.py` (SigLIP + HSV colours), `database/metadata_store.py` |
| 4. Indexing & search | `database/vector_store.py` (FAISS) |
| 5. Natural-language retrieval | `pipeline/retrieval.py`, REST API in `main.py` |

Two kinds of embeddings are stored: **one per track** (the average of that person/vehicle's
sharpest crops, with the track's start/end time and measured upper/lower colour) and **whole frames**
(every 5 s, so queries about *where* something is have context). Frames with no motion skip YOLO.

## Project structure

```
Mini-Project-/
├── config/default.yaml          # all tunables
├── src/video_retrieval/
│   ├── api/                     # Pydantic request/response schemas
│   ├── database/                # metadata_store.py (SQL), vector_store.py (FAISS)
│   ├── detection/detector.py    # YOLO
│   ├── embedding/encoder.py     # SigLIP / CLIP, HSV colour measurement
│   ├── pipeline/                # ingestion.py, retrieval.py (+ evaluation)
│   ├── tracking/tracker.py      # ByteTrack (+ IoU fallback)
│   └── utils/                   # config.py, logging.py
├── main.py                      # FastAPI app + CLI
├── docker-compose.yml           # PostgreSQL (optional)
├── .env.example · .gitignore · pyproject.toml · requirements.txt
```

## Setup

Python 3.10+ (3.11 recommended).

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For a GPU, install the CUDA build of PyTorch first (see pytorch.org), then the rest.
YOLO weights and the SigLIP model download automatically on first run.
`ffmpeg` is optional: if installed, clips are cut as H.264 (plays in every browser); otherwise OpenCV is used.

## Usage

```bash
python main.py serve                                   # API + web UI at http://localhost:8000
python main.py ingest data/uploads/gate.mp4 --camera-id main-gate --start-time 2026-11-05T17:00:00
python main.py search "a person wearing a red shirt" --top-k 5
python main.py evaluate ground_truth.json
```

`--start-time` gives results real wall-clock times (like `17:02:13`). RTSP works the same way:
`python main.py ingest rtsp://user:pass@192.168.1.10/stream` (stop it via `DELETE /jobs/{id}` when using the API).

### REST API (docs at `/docs`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Index a file path / RTSP URL (background job) |
| POST | `/upload` | Upload a video file and index it |
| GET / DELETE | `/jobs`, `/jobs/{id}` | Job progress / stop a job |
| POST | `/search` | `{"query": "...", "top_k": 10, "classes": ["person"], "camera_id": "...", "start_time": "...", "end_time": "..."}` |
| GET | `/search?q=...` | Quick search |
| GET | `/clips/{video_id}?start=&end=` | Cut and stream a clip |
| GET | `/videos`, `/health` | Inventory / status |

## PostgreSQL (optional)

```bash
docker compose up -d postgres
# in .env:
DATABASE_URL=postgresql+psycopg2://vr_user:vr_password@localhost:5432/video_retrieval
```

## Evaluation (slide 8 plan)

`python main.py evaluate ground_truth.json` reports **Top-k accuracy** (k = 1, 5, 10) and **response time** (mean / p95).
Ground truth is a JSON list of labelled queries:

```json
[{"query": "person with a backpack", "video": "gate.mp4", "start": 12.0, "end": 20.0}]
```

A query counts as a hit when a returned segment overlaps the labelled interval. Detection/tracking
quality (mAP, MOTA/IDF1) is best measured with the standard toolkits on your annotated datasets
(e.g. `ultralytics val`, `TrackEval`); compare search time against manual review for the last metric.

## Accuracy features

- **Per-track embeddings:** each track is embedded once, from the average of its 5 sharpest crops (less blur/angle noise, smaller index).
- **Colour verification:** upper/lower (or vehicle body) colour is measured from pixels with HSV rules. In `color_mode: strict`, a colour word in the query ("green hoodie", "blue jeans", "white car") must match the measured colour.
- **Rejection of non-matches:** every hit is re-scored against contrast prompts (generic ones from `retrieval.negative_prompts` plus the query with its colour words swapped). Hits under `retrieval.min_confidence` are dropped, so a query with no match returns an empty list. Tune it by comparing the `conf` shown on the cards for a query that is present and one that is absent.
- **Motion skipping:** frames with no change since the last detection skip YOLO (still detects at least every 2 s).

## Intel CPU / Arc GPU

- **YOLO:** runs through OpenVINO automatically (`detection.runtime: auto`) on the Arc GPU if present, else the CPU. The first run exports `yolo11s` to OpenVINO once. If OpenVINO fails, it falls back to PyTorch.
- **SigLIP:** runs on the CPU by default. To use the Arc GPU, install the Intel-XPU build of PyTorch and keep `embedding.device: auto`:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu`
  (needs a recent Intel graphics driver).

## Tuning tips

- After changing the detector or embedding model, delete `data/` and re-ingest (vector sizes and tracks change).
- SigLIP similarity scores are lower than CLIP's; `retrieval.min_score` is set per backend.
- `retrieval.min_confidence` and `retrieval.relative_margin` control how strict results are.
- Any config key can be set via env var, e.g. `VR__EMBEDDING__DEVICE=xpu`.

## Known limitations

- SigLIP/CLIP match *appearance* (colours, clothing, object types) well but does not understand motion or
  temporal events ("running", "loitering"); this is the gap noted in the literature review. Future work:
  VideoCLIP-style clip embeddings or captioning (Cap4Video) and LLM query refinement.
- Spatial phrases ("near the main gate") are only matched through full-frame embeddings, not geometry.
- ByteTrack comes from `supervision<0.30`; with a newer version the app falls back to a simpler IoU tracker.

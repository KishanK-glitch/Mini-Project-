"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(..., description="Video file path, RTSP/HTTP URL, or webcam index")
    camera_id: str = "cam-01"
    start_time: Optional[datetime] = Field(
        None, description="Wall-clock time of the first frame (enables absolute timestamps)"
    )


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | failed | stopped
    source: str
    video_id: Optional[int] = None
    frames_processed: int = 0
    observations: int = 0
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["a person wearing a red shirt near the gate"])
    top_k: int = Field(10, ge=1, le=100)
    video_id: Optional[int] = None
    camera_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    classes: Optional[List[str]] = Field(None, description="Restrict to these object classes")


class SearchHit(BaseModel):
    rank: int
    score: float
    confidence: Optional[float] = None  # calibrated 0-1 probability that the query matches
    video_id: int
    video_name: str
    camera_id: str
    kind: str  # object | frame
    track_id: Optional[int] = None
    class_name: str
    start_sec: float
    end_sec: float
    timecode: str  # HH:MM:SS offset into the video
    start_time: Optional[datetime] = None  # absolute, if the video has a start time
    end_time: Optional[datetime] = None
    num_matches: int
    upper_color: Optional[str] = None
    lower_color: Optional[str] = None
    thumbnail_url: Optional[str] = None
    clip_url: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    count: int
    results: List[SearchHit]


class VideoInfo(BaseModel):
    id: int
    name: str
    path: str
    camera_id: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float
    start_time: Optional[datetime] = None
    status: str

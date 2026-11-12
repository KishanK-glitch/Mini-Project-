"""Relational metadata store (SQLite by default, PostgreSQL/MySQL via SQLAlchemy URL).

Two tables:
  videos        one row per ingested video / stream
  observations  one row per embedded item: a tracked-object crop (kind="object")
                or a whole frame (kind="frame"). observation.id == vector id in FAISS.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024))
    name: Mapped[str] = mapped_column(String(255))
    camera_id: Mapped[str] = mapped_column(String(64), default="cam-01", index=True)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="processing")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(8), default="object")  # object | frame
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    class_name: Mapped[str] = mapped_column(String(32), default="scene")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    frame_idx: Mapped[int] = mapped_column(Integer, default=0)
    timestamp_sec: Mapped[float] = mapped_column(Float, default=0.0)
    x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    x2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thumbnail: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # track extent (object rows) - the whole time span this embedding represents
    t_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # colours measured from pixels (HSV); upper = shirt/jacket, lower = trousers, body = vehicles
    upper_color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    lower_color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)


def _as_dict(obj: Base) -> Dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


class MetadataStore:
    def __init__(self, url: str):
        kwargs: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)
        self._check_schema()
        self._Session = sessionmaker(self.engine, expire_on_commit=False)

    def _check_schema(self) -> None:
        from sqlalchemy import inspect

        cols = {c["name"] for c in inspect(self.engine).get_columns("observations")}
        missing = {c.name for c in Observation.__table__.columns} - cols
        if missing:
            raise RuntimeError(
                f"Existing database is from an older version (missing columns: {sorted(missing)}). "
                "Delete the data folder and re-ingest your videos."
            )

    def session(self) -> Session:
        return self._Session()

    # ---------------------------------------------------------------- videos
    def add_video(self, **fields: Any) -> int:
        with self.session() as s:
            video = Video(**fields)
            s.add(video)
            s.commit()
            return video.id

    def update_video(self, video_id: int, **fields: Any) -> None:
        with self.session() as s:
            video = s.get(Video, video_id)
            if video is None:
                raise KeyError(f"video {video_id} not found")
            for key, value in fields.items():
                setattr(video, key, value)
            s.commit()

    def get_video(self, video_id: int) -> Optional[Dict[str, Any]]:
        with self.session() as s:
            video = s.get(Video, video_id)
            return _as_dict(video) if video else None

    def get_videos(self, ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
        ids = list(set(ids))
        if not ids:
            return {}
        with self.session() as s:
            rows = s.scalars(select(Video).where(Video.id.in_(ids))).all()
            return {v.id: _as_dict(v) for v in rows}

    def list_videos(self) -> List[Dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(select(Video).order_by(Video.id)).all()
            return [_as_dict(v) for v in rows]

    # ---------------------------------------------------------- observations
    def add_observations(self, rows: List[Dict[str, Any]]) -> List[int]:
        """Insert rows and return their new ids (same order as `rows`)."""
        if not rows:
            return []
        with self.session() as s:
            objs = [Observation(**r) for r in rows]
            s.add_all(objs)
            s.flush()
            ids = [o.id for o in objs]
            s.commit()
            return ids

    def get_observations(self, ids: Iterable[int], chunk: int = 500) -> Dict[int, Dict[str, Any]]:
        ids = list(ids)
        out: Dict[int, Dict[str, Any]] = {}
        with self.session() as s:
            for i in range(0, len(ids), chunk):
                part = ids[i : i + chunk]
                for o in s.scalars(select(Observation).where(Observation.id.in_(part))):
                    out[o.id] = _as_dict(o)
        return out

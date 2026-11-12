"""Vector index for similarity search (FAISS, with a pure-NumPy fallback).

Vectors are expected to be L2-normalised, so inner product == cosine similarity.
Vector ids are the `observations.id` values from the metadata store.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from video_retrieval.utils.logging import get_logger

log = get_logger(__name__)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class VectorStore:
    def __init__(self, directory: str, backend: str = "faiss"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        if backend == "faiss" and faiss is None:
            log.warning("faiss not installed - falling back to NumPy brute-force search")
            backend = "numpy"
        self.backend = backend
        self._path = self.dir / ("index.faiss" if backend == "faiss" else "vectors.npz")
        self._lock = threading.RLock()
        self.dim: Optional[int] = None
        self._index = None
        self._ids = np.empty(0, dtype="int64")
        self._vecs = np.empty((0, 0), dtype="float32")
        self._mtime = 0.0
        self._load()

    # ------------------------------------------------------------ internals
    def _init(self, dim: int) -> None:
        self.dim = dim
        if self.backend == "faiss":
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            self._vecs = np.empty((0, dim), dtype="float32")
            self._ids = np.empty(0, dtype="int64")

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._lock:
            if self.backend == "faiss":
                self._index = faiss.read_index(str(self._path))
                self.dim = self._index.d
            else:
                data = np.load(self._path)
                self._ids, self._vecs = data["ids"], data["vecs"]
                self.dim = int(self._vecs.shape[1])
            self._mtime = self._path.stat().st_mtime
            log.info("Loaded %d vectors from %s", len(self), self._path)

    def _maybe_reload(self) -> None:
        """Pick up index changes written by another process (e.g. the CLI)."""
        if self._path.exists() and self._path.stat().st_mtime > self._mtime:
            self._load()

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        if self.backend == "faiss":
            return int(self._index.ntotal) if self._index is not None else 0
        return int(self._ids.shape[0])

    def add(self, ids: List[int], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        ids_arr = np.asarray(ids, dtype="int64")
        if len(ids_arr) == 0:
            return
        if vectors.ndim != 2 or len(ids_arr) != len(vectors):
            raise ValueError("ids and vectors must have matching length")
        with self._lock:
            if self.dim is None:
                self._init(vectors.shape[1])
            elif vectors.shape[1] != self.dim:
                raise ValueError(
                    f"Embedding dim {vectors.shape[1]} != index dim {self.dim}. "
                    "Did you change the embedding model? Delete data/vectors and re-ingest."
                )
            if self.backend == "faiss":
                self._index.add_with_ids(vectors, ids_arr)
            else:
                self._vecs = np.vstack([self._vecs, vectors])
                self._ids = np.concatenate([self._ids, ids_arr])

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        self._maybe_reload()
        with self._lock:
            n = len(self)
            if n == 0 or self.dim is None:
                return []
            q = np.ascontiguousarray(query, dtype="float32").reshape(1, -1)
            k = min(k, n)
            if self.backend == "faiss":
                scores, ids = self._index.search(q, k)
                return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
            sims = self._vecs @ q[0]
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            return [(int(self._ids[i]), float(sims[i])) for i in top]

    def get_vectors(self, ids: List[int]) -> np.ndarray:
        """Return the stored vectors for `ids` (same order). Raises KeyError if missing."""
        self._maybe_reload()
        with self._lock:
            if self.dim is None:
                raise KeyError("index is empty")
            if self.backend == "faiss":
                return np.stack([self._index.reconstruct(int(i)) for i in ids]).astype("float32")
            pos = {int(v): k for k, v in enumerate(self._ids)}
            return self._vecs[[pos[int(i)] for i in ids]].astype("float32")

    def save(self) -> None:
        with self._lock:
            if self.dim is None:
                return
            tmp = self.dir / ("index.tmp" if self.backend == "faiss" else "vectors.tmp.npz")
            if self.backend == "faiss":
                faiss.write_index(self._index, str(tmp))
            else:
                np.savez(tmp, ids=self._ids, vecs=self._vecs)
            os.replace(tmp, self._path)
            self._mtime = self._path.stat().st_mtime

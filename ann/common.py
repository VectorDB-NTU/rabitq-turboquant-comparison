from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RECALL_KS = [1, 2, 4, 8, 16, 32, 64]
BITS = [2, 4]
OPENAI_DIMS = [1536, 3072]
DEFAULT_REPEATS = 10
DEFAULT_TURBO_SEED_START = 42
DEFAULT_NUM_THREADS = 32

ROOT_DIR = Path(__file__).resolve().parents[2]
ANN_DIR = Path(__file__).resolve().parent
DATA_DIR = ANN_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ANN_DIR / "results"
FIGURES_DIR = ANN_DIR / "figures"
BUILD_DIR = ANN_DIR / "build"
BIN_DIR = BUILD_DIR / "bin"

OPENAI_DATASETS = {
    1536: "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
    3072: "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
}

GLOVE_URL = "https://ann-benchmarks.com/glove-200-angular.hdf5"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    train_npy: Path
    test_npy: Path
    neighbors_npy: Path | None
    train_fvecs: Path
    test_fvecs: Path
    y_min: float
    y_max: float
    y_ticks: list[float]
    plot_left_padding: int


def ensure_work_dirs() -> None:
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, BUILD_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return rows / norms


def write_fvecs(path: Path, array: np.ndarray) -> None:
    data = np.asarray(array, dtype=np.float32, order="C")
    n, dim = data.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for i in range(n):
            np.array([dim], dtype=np.int32).tofile(handle)
            data[i].tofile(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def dataset_specs(processed_dir: Path = PROCESSED_DATA_DIR) -> dict[str, DatasetSpec]:
    processed_dir = processed_dir.resolve()
    return {
        "glove_200": DatasetSpec(
            key="glove_200",
            title="(a) GloVe - d=200",
            train_npy=processed_dir / "glove-200-angular-normalized-100k-train.npy",
            test_npy=processed_dir / "glove-200-angular-normalized-100k-test.npy",
            neighbors_npy=processed_dir / "glove-200-angular-normalized-100k-neighbors.npy",
            train_fvecs=processed_dir / "glove-200-angular-normalized-100k-train.fvecs",
            test_fvecs=processed_dir / "glove-200-angular-normalized-100k-test.fvecs",
            y_min=0.48,
            y_max=1.005,
            y_ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            plot_left_padding=88,
        ),
        "openai_1536": DatasetSpec(
            key="openai_1536",
            title="(b) OpenAI3 - d=1536",
            train_npy=processed_dir / "dbpedia-openai3-1536-train.npy",
            test_npy=processed_dir / "dbpedia-openai3-1536-test.npy",
            neighbors_npy=None,
            train_fvecs=processed_dir / "dbpedia-openai3-1536-train.fvecs",
            test_fvecs=processed_dir / "dbpedia-openai3-1536-test.fvecs",
            y_min=0.85,
            y_max=1.005,
            y_ticks=[0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0],
            plot_left_padding=108,
        ),
        "openai_3072": DatasetSpec(
            key="openai_3072",
            title="(c) OpenAI3 - d=3072",
            train_npy=processed_dir / "dbpedia-openai3-3072-train.npy",
            test_npy=processed_dir / "dbpedia-openai3-3072-test.npy",
            neighbors_npy=None,
            train_fvecs=processed_dir / "dbpedia-openai3-3072-train.fvecs",
            test_fvecs=processed_dir / "dbpedia-openai3-3072-test.fvecs",
            y_min=0.85,
            y_max=1.005,
            y_ticks=[0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0],
            plot_left_padding=108,
        ),
    }

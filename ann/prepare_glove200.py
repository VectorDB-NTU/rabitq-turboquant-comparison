from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from common import GLOVE_URL, PROCESSED_DATA_DIR, RAW_DATA_DIR, dump_json, ensure_work_dirs, l2_normalize, write_fvecs


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] {destination.name} already exists")
        return destination

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    print(f"Saved {destination}")
    return destination


def compute_angular_ground_truth(
    train_norm: np.ndarray,
    test_norm: np.ndarray,
    topk: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = test_norm.shape[0]
    train_t = np.asarray(train_norm.T, dtype=np.float32, order="C")
    neighbors = np.empty((n_queries, topk), dtype=np.int32)
    distances = np.empty((n_queries, topk), dtype=np.float32)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        scores = np.asarray(test_norm[start:end] @ train_t, dtype=np.float32)

        candidate_idx = np.argpartition(scores, kth=scores.shape[1] - topk, axis=1)[:, -topk:]
        candidate_scores = np.take_along_axis(scores, candidate_idx, axis=1)
        order = np.argsort(-candidate_scores, axis=1)

        top_idx = np.take_along_axis(candidate_idx, order, axis=1)
        top_scores = np.take_along_axis(candidate_scores, order, axis=1)

        neighbors[start:end] = top_idx.astype(np.int32, copy=False)
        distances[start:end] = (1.0 - top_scores).astype(np.float32, copy=False)
        print(f"Computed GloVe ground truth for queries [{start}, {end})")

    return neighbors, distances


def prepare_glove_dataset(
    input_hdf5: Path | None = None,
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    sample_train_size: int = 100000,
    seed: int = 42,
    gt_k: int = 100,
    gt_batch_size: int = 128,
    force: bool = False,
) -> dict:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_hdf5 = (
        input_hdf5.resolve()
        if input_hdf5 is not None
        else raw_dir / "glove-200-angular.hdf5"
    )
    if not input_hdf5.exists():
        download_file(GLOVE_URL, input_hdf5)

    prefix = "glove-200-angular-normalized-100k"
    train_path = output_dir / f"{prefix}-train.npy"
    test_path = output_dir / f"{prefix}-test.npy"
    neighbors_path = output_dir / f"{prefix}-neighbors.npy"
    distances_path = output_dir / f"{prefix}-distances.npy"
    train_fvecs = output_dir / f"{prefix}-train.fvecs"
    test_fvecs = output_dir / f"{prefix}-test.fvecs"
    indices_path = output_dir / "glove-200-angular-100k-train-indices.npy"
    metadata_path = output_dir / "glove-200-angular-100k-metadata.json"

    expected_paths = [
        train_path,
        test_path,
        neighbors_path,
        distances_path,
        train_fvecs,
        test_fvecs,
        indices_path,
        metadata_path,
    ]
    if not force and all(path.exists() for path in expected_paths):
        print(f"[skip] {prefix} already prepared in {output_dir}")
        return {
            "train_npy": str(train_path),
            "test_npy": str(test_path),
            "neighbors_npy": str(neighbors_path),
            "distances_npy": str(distances_path),
            "train_fvecs": str(train_fvecs),
            "test_fvecs": str(test_fvecs),
            "train_indices": str(indices_path),
            "metadata": str(metadata_path),
        }

    with h5py.File(input_hdf5, "r") as handle:
        train = np.asarray(handle["train"], dtype=np.float32)
        test = np.asarray(handle["test"], dtype=np.float32)

    rng = np.random.default_rng(seed)
    train_indices = np.sort(rng.choice(train.shape[0], size=sample_train_size, replace=False))
    sampled_train = np.asarray(train[train_indices], dtype=np.float32, order="C")
    sampled_test = np.asarray(test, dtype=np.float32, order="C")

    train_norm = np.asarray(l2_normalize(sampled_train), dtype=np.float32, order="C")
    test_norm = np.asarray(l2_normalize(sampled_test), dtype=np.float32, order="C")
    neighbors, distances = compute_angular_ground_truth(
        train_norm, test_norm, topk=gt_k, batch_size=gt_batch_size
    )

    np.save(train_path, train_norm)
    np.save(test_path, test_norm)
    np.save(neighbors_path, neighbors)
    np.save(distances_path, distances)
    np.save(indices_path, train_indices)
    write_fvecs(train_fvecs, train_norm)
    write_fvecs(test_fvecs, test_norm)

    metadata = {
        "source_hdf5": str(input_hdf5),
        "download_url": GLOVE_URL,
        "sample_train_size": sample_train_size,
        "seed": seed,
        "gt_k": gt_k,
        "gt_batch_size": gt_batch_size,
        "train_shape": list(train_norm.shape),
        "test_shape": list(test_norm.shape),
        "neighbors_shape": list(neighbors.shape),
        "files": {
            "train_npy": train_path.name,
            "test_npy": test_path.name,
            "neighbors_npy": neighbors_path.name,
            "distances_npy": distances_path.name,
            "train_fvecs": train_fvecs.name,
            "test_fvecs": test_fvecs.name,
            "train_indices": indices_path.name,
        },
        "notes": [
            "The reproduction figure uses the 100k sampled training/base set.",
            "Both train and query vectors are L2-normalized so inner product matches cosine similarity.",
            "Ground truth is recomputed exactly against the sampled normalized training set.",
        ],
    }
    dump_json(metadata_path, metadata)

    print(f"Saved {prefix}: train={train_norm.shape}, test={test_norm.shape}")
    return {
        "train_npy": str(train_path),
        "test_npy": str(test_path),
        "neighbors_npy": str(neighbors_path),
        "distances_npy": str(distances_path),
        "train_fvecs": str(train_fvecs),
        "test_fvecs": str(test_fvecs),
        "train_indices": str(indices_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the normalized GloVe-200 100k subset used in the comparison figure."
    )
    parser.add_argument(
        "--input-hdf5",
        type=Path,
        default=None,
        help="Optional path to a pre-downloaded glove-200-angular.hdf5 file",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--sample-train-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--gt-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_work_dirs()
    prepare_glove_dataset(
        input_hdf5=args.input_hdf5,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        sample_train_size=args.sample_train_size,
        seed=args.seed,
        gt_k=args.gt_k,
        gt_batch_size=args.gt_batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()

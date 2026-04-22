from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from common import OPENAI_DATASETS, PROCESSED_DATA_DIR, dump_json, ensure_work_dirs, write_fvecs


def prepare_openai_dataset(
    dim: int,
    output_dir: Path = PROCESSED_DATA_DIR,
    sample_size: int = 101000,
    test_size: int = 1000,
    seed: int = 42,
    force: bool = False,
) -> dict:
    if dim not in OPENAI_DATASETS:
        raise ValueError(f"Unsupported dim: {dim}")
    if sample_size <= test_size:
        raise ValueError("sample_size must be larger than test_size")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"dbpedia-openai3-{dim}"
    train_path = output_dir / f"{prefix}-train.npy"
    test_path = output_dir / f"{prefix}-test.npy"
    train_fvecs = output_dir / f"{prefix}-train.fvecs"
    test_fvecs = output_dir / f"{prefix}-test.fvecs"
    sampled_indices_path = output_dir / f"{prefix}-sampled-indices.npy"
    test_positions_path = output_dir / f"{prefix}-test-positions.npy"
    metadata_path = output_dir / f"{prefix}-metadata.json"

    expected_paths = [
        train_path,
        test_path,
        train_fvecs,
        test_fvecs,
        sampled_indices_path,
        test_positions_path,
        metadata_path,
    ]
    if not force and all(path.exists() for path in expected_paths):
        print(f"[skip] {prefix} already prepared in {output_dir}")
        return {
            "train_npy": str(train_path),
            "test_npy": str(test_path),
            "train_fvecs": str(train_fvecs),
            "test_fvecs": str(test_fvecs),
            "sampled_indices": str(sampled_indices_path),
            "test_positions": str(test_positions_path),
            "metadata": str(metadata_path),
        }

    dataset_name = OPENAI_DATASETS[dim]
    column_name = f"text-embedding-3-large-{dim}-embedding"

    print(f"Loading {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(len(dataset), size=sample_size, replace=False))
    sampled = dataset.select(sampled_indices.tolist())

    embeddings = np.asarray(sampled[column_name], dtype=np.float32)

    test_positions = np.sort(rng.choice(sample_size, size=test_size, replace=False))
    train_mask = np.ones(sample_size, dtype=bool)
    train_mask[test_positions] = False

    train = np.asarray(embeddings[train_mask], dtype=np.float32, order="C")
    test = np.asarray(embeddings[test_positions], dtype=np.float32, order="C")

    np.save(train_path, train)
    np.save(test_path, test)
    np.save(sampled_indices_path, sampled_indices)
    np.save(test_positions_path, test_positions)
    write_fvecs(train_fvecs, train)
    write_fvecs(test_fvecs, test)

    metadata = {
        "dataset_name": dataset_name,
        "dim": dim,
        "seed": seed,
        "sample_size": sample_size,
        "test_size": test_size,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "files": {
            "train_npy": train_path.name,
            "test_npy": test_path.name,
            "train_fvecs": train_fvecs.name,
            "test_fvecs": test_fvecs.name,
            "sampled_indices": sampled_indices_path.name,
            "test_positions": test_positions_path.name,
        },
    }
    dump_json(metadata_path, metadata)

    print(f"Saved {prefix}: train={train.shape}, test={test.shape}")
    return {
        "train_npy": str(train_path),
        "test_npy": str(test_path),
        "train_fvecs": str(train_fvecs),
        "test_fvecs": str(test_fvecs),
        "sampled_indices": str(sampled_indices_path),
        "test_positions": str(test_positions_path),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the DBpedia OpenAI3 dataset used in the comparison figure."
    )
    parser.add_argument("--dim", type=int, choices=sorted(OPENAI_DATASETS), required=True)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--sample-size", type=int, default=101000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_work_dirs()
    prepare_openai_dataset(
        dim=args.dim,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        test_size=args.test_size,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from common import OPENAI_DIMS, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_work_dirs
from prepare_glove200 import prepare_glove_dataset
from prepare_openai_dbpedia import prepare_openai_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare all three datasets used in the RaBitQ vs TurboQuant recall figure."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--openai-sample-size", type=int, default=101000)
    parser.add_argument("--openai-test-size", type=int, default=1000)
    parser.add_argument("--openai-seed", type=int, default=42)
    parser.add_argument("--glove-seed", type=int, default=42)
    parser.add_argument("--glove-sample-train-size", type=int, default=100000)
    parser.add_argument("--glove-gt-k", type=int, default=100)
    parser.add_argument("--glove-gt-batch-size", type=int, default=128)
    parser.add_argument("--glove-input-hdf5", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_work_dirs()

    for dim in OPENAI_DIMS:
        prepare_openai_dataset(
            dim=dim,
            output_dir=args.processed_dir,
            sample_size=args.openai_sample_size,
            test_size=args.openai_test_size,
            seed=args.openai_seed,
            force=args.force,
        )

    prepare_glove_dataset(
        input_hdf5=args.glove_input_hdf5,
        raw_dir=args.raw_dir,
        output_dir=args.processed_dir,
        sample_train_size=args.glove_sample_train_size,
        seed=args.glove_seed,
        gt_k=args.glove_gt_k,
        gt_batch_size=args.glove_gt_batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from common import (
    BITS,
    BIN_DIR,
    DEFAULT_NUM_THREADS,
    DEFAULT_REPEATS,
    DEFAULT_TURBO_SEED_START,
    PROCESSED_DATA_DIR,
    RECALL_KS,
    RESULTS_DIR,
    dataset_specs,
    dump_json,
    ensure_work_dirs,
)
from run_rabitq_recall import run_rabitq_recall
from run_turboquant_recall import run_turboquant_recall


def summarize_runs(runs: list[list[float]]) -> dict[str, list[float] | list[list[float]]]:
    mean = []
    std = []
    for idx in range(len(RECALL_KS)):
        values = [run[idx] for run in runs]
        mean.append(statistics.mean(values))
        std.append(statistics.pstdev(values))
    return {"runs": runs, "mean": mean, "std": std}


def collect_turbo_runs(
    train_npy: Path,
    test_npy: Path,
    bitwidth: int,
    variant: str,
    repeats: int,
    seed_start: int,
    device: str,
    batch_size: int,
    neighbors_npy: Path | None = None,
) -> dict[str, list[float] | list[list[float]]]:
    runs = []
    for offset in range(repeats):
        seed = seed_start + offset
        recalls = run_turboquant_recall(
            train_path=train_npy,
            test_path=test_npy,
            neighbors_path=neighbors_npy,
            bitwidth=bitwidth,
            variant=variant,
            seed=seed,
            device=device,
            batch_size=batch_size,
            k_values=RECALL_KS,
        )
        runs.append([recalls[k] for k in RECALL_KS])
        print(
            f"[turbo-{variant}] {train_npy.name} bit={bitwidth} "
            f"run={offset + 1}/{repeats} done"
        )
    return summarize_runs(runs)


def collect_rabitq_runs(
    binary: Path,
    train_fvecs: Path,
    test_fvecs: Path,
    bitwidth: int,
    repeats: int,
    num_threads: int,
    neighbors_npy: Path | None = None,
) -> dict[str, list[float] | list[list[float]]]:
    runs = []
    for offset in range(repeats):
        recalls, _ = run_rabitq_recall(
            binary=binary,
            base_fvecs=train_fvecs,
            query_fvecs=test_fvecs,
            neighbors_path=neighbors_npy,
            bitwidth=bitwidth,
            k_values=RECALL_KS,
            metric="ip",
            faster_quant=False,
            rotator="matrix",
            num_threads=num_threads,
        )
        runs.append([recalls[k] for k in RECALL_KS])
        print(f"[rabitq] {train_fvecs.name} bit={bitwidth} run={offset + 1}/{repeats} done")
    return summarize_runs(runs)


def collect_results(
    processed_dir: Path,
    rabitq_binary: Path,
    turbo_device: str,
    turbo_batch_size: int,
    turbo_repeats: int,
    rabitq_repeats: int,
    turbo_seed_start: int,
    num_threads: int,
) -> dict:
    specs = dataset_specs(processed_dir)
    results: dict[str, dict] = {}

    for dataset_key, spec in specs.items():
        panel = {}
        for bitwidth in BITS:
            panel[f"turbo_mse_{bitwidth}bit"] = collect_turbo_runs(
                train_npy=spec.train_npy,
                test_npy=spec.test_npy,
                neighbors_npy=spec.neighbors_npy,
                bitwidth=bitwidth,
                variant="mse",
                repeats=turbo_repeats,
                seed_start=turbo_seed_start,
                device=turbo_device,
                batch_size=turbo_batch_size,
            )
            panel[f"turbo_prod_{bitwidth}bit"] = collect_turbo_runs(
                train_npy=spec.train_npy,
                test_npy=spec.test_npy,
                neighbors_npy=spec.neighbors_npy,
                bitwidth=bitwidth,
                variant="prod",
                repeats=turbo_repeats,
                seed_start=turbo_seed_start,
                device=turbo_device,
                batch_size=turbo_batch_size,
            )
            panel[f"rabitq_{bitwidth}bit"] = collect_rabitq_runs(
                binary=rabitq_binary,
                train_fvecs=spec.train_fvecs,
                test_fvecs=spec.test_fvecs,
                neighbors_npy=spec.neighbors_npy,
                bitwidth=bitwidth,
                repeats=rabitq_repeats,
                num_threads=num_threads,
            )
        results[dataset_key] = panel

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all TurboQuant and RaBitQ experiments needed for the combined three-panel figure."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--rabitq-binary", type=Path, default=BIN_DIR / "rabitq")
    parser.add_argument("--turbo-device", type=str, default="cuda")
    parser.add_argument("--turbo-batch-size", type=int, default=256)
    parser.add_argument("--turbo-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--rabitq-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--turbo-seed-start", type=int, default=DEFAULT_TURBO_SEED_START)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESULTS_DIR / "recall_at1_three_panel.json",
    )
    args = parser.parse_args()

    ensure_work_dirs()

    results = collect_results(
        processed_dir=args.processed_dir.resolve(),
        rabitq_binary=args.rabitq_binary.resolve(),
        turbo_device=args.turbo_device,
        turbo_batch_size=args.turbo_batch_size,
        turbo_repeats=args.turbo_repeats,
        rabitq_repeats=args.rabitq_repeats,
        turbo_seed_start=args.turbo_seed_start,
        num_threads=args.num_threads,
    )

    payload = {
        "config": {
            "recall_k": RECALL_KS,
            "bits": BITS,
            "turbo_variant_labels": {
                "mse": "TurboQuant_mse",
                "prod": "TurboQuant_prod",
            },
            "rabitq_label": "RaBitQ",
            "turbo_device": args.turbo_device,
            "turbo_batch_size": args.turbo_batch_size,
            "turbo_repeats": args.turbo_repeats,
            "rabitq_repeats": args.rabitq_repeats,
            "turbo_seed_start": args.turbo_seed_start,
            "num_threads": args.num_threads,
            "rabitq_binary": str(args.rabitq_binary.resolve()),
            "rabitq_rotator": "matrix",
            "rabitq_faster_quant": False,
        },
        **results,
    }
    dump_json(args.output_json.resolve(), payload)
    print(f"Saved experiment results to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

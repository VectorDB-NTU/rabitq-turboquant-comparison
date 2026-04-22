from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import BIN_DIR, DEFAULT_NUM_THREADS, RECALL_KS


def parse_k_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("k-values must not be empty")
    return values


def parse_rabitq_output(stdout: str, k_values: list[int]) -> dict[int, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("t\\K"):
            header_idx = idx
            break
    if header_idx is None or header_idx + 1 >= len(lines):
        raise ValueError(f"Could not find recall table in rabitq output:\n{stdout}")

    header_tokens = lines[header_idx].split()
    row_tokens = lines[header_idx + 1].split()
    if row_tokens[0] != "1":
        raise ValueError(f"Expected t=1 row, got: {lines[header_idx + 1]}")

    values = {int(k): float(v) for k, v in zip(header_tokens[1:], row_tokens[1:])}
    missing = [k for k in k_values if k not in values]
    if missing:
        raise ValueError(f"Missing recall values for K={missing}\n{stdout}")
    return {k: values[k] for k in k_values}


def run_rabitq_recall(
    binary: Path,
    base_fvecs: Path,
    query_fvecs: Path,
    bitwidth: int,
    k_values: list[int] | None = None,
    neighbors_path: Path | None = None,
    metric: str = "ip",
    faster_quant: bool = False,
    rotator: str = "matrix",
    num_threads: int = DEFAULT_NUM_THREADS,
) -> tuple[dict[int, float], str]:
    k_values = k_values or RECALL_KS
    if not binary.exists():
        raise FileNotFoundError(
            f"RaBitQ binary not found: {binary}. Build it first with CMake in ann/."
        )

    cmd = [
        str(binary),
        "--base",
        str(base_fvecs),
        "--query",
        str(query_fvecs),
        "--bits",
        str(bitwidth),
        "--metric",
        metric,
        "--faster-quant",
        "true" if faster_quant else "false",
        "--ranks",
        ",".join(str(k) for k in k_values),
        "--rotator",
        rotator,
        "--num-threads",
        str(num_threads),
    ]
    if neighbors_path is not None:
        cmd.extend(["--neighbors", str(neighbors_path)])

    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_rabitq_output(proc.stdout, k_values), proc.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ann/rabitq.cpp evaluator once and extract Recall@1@K."
    )
    parser.add_argument("--binary", type=Path, default=BIN_DIR / "rabitq")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--neighbors", type=Path, default=None)
    parser.add_argument("--bitwidth", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], required=True)
    parser.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    parser.add_argument("--faster-quant", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--rotator", type=str, default="matrix", choices=["auto", "identity", "matrix", "fht"])
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    recalls, raw_stdout = run_rabitq_recall(
        binary=args.binary,
        base_fvecs=args.base,
        query_fvecs=args.query,
        neighbors_path=args.neighbors,
        bitwidth=args.bitwidth,
        k_values=k_values,
        metric=args.metric,
        faster_quant=args.faster_quant == "true",
        rotator=args.rotator,
        num_threads=args.num_threads,
    )

    payload = {
        "binary": str(args.binary.resolve()),
        "base": str(args.base.resolve()),
        "query": str(args.query.resolve()),
        "neighbors": str(args.neighbors.resolve()) if args.neighbors else None,
        "bitwidth": args.bitwidth,
        "metric": args.metric,
        "faster_quant": args.faster_quant == "true",
        "rotator": args.rotator,
        "num_threads": args.num_threads,
        "k_values": k_values,
        "recall": {str(k): recalls[k] for k in k_values},
        "raw_stdout": raw_stdout,
    }

    print(
        f"Running RaBitQ Recall@1@K for bitwidth={args.bitwidth} on {args.base.name} "
        f"(rotator={args.rotator}, faster_quant={args.faster_quant})"
    )
    for k in k_values:
        print(f"Recall@{k}: {recalls[k]:.4f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

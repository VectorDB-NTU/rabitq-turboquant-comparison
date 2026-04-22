from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from common import RECALL_KS, ROOT_DIR


def load_turbo_quant_module():
    turbo_path = ROOT_DIR / "vector_search" / "turbo_quant.py"
    spec = importlib.util.spec_from_file_location("ann_turbo_quant_impl", turbo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load TurboQuant module from {turbo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_k_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("k-values must not be empty")
    return values


def recall_from_ground_truth(
    queries: torch.Tensor,
    quantized_base: torch.Tensor,
    ground_truth_top1: torch.Tensor,
    k_values: list[int],
    batch_size: int,
) -> dict[int, float]:
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    num_queries = queries.shape[0]

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        query_batch = queries[start:end]
        gt_batch = ground_truth_top1[start:end]

        approx_scores = torch.matmul(query_batch, quantized_base.T)
        approx_topk = torch.topk(approx_scores, k=max_k, dim=1).indices

        for k in k_values:
            hit_mask = (approx_topk[:, :k] == gt_batch.unsqueeze(1)).any(dim=1)
            hits[k] += int(hit_mask.sum().item())

    return {k: hits[k] / num_queries for k in k_values}


def recall_from_exact_ip(
    queries: torch.Tensor,
    base: torch.Tensor,
    quantized_base: torch.Tensor,
    k_values: list[int],
    batch_size: int,
) -> dict[int, float]:
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    num_queries = queries.shape[0]

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        query_batch = queries[start:end]

        exact_scores = torch.matmul(query_batch, base.T)
        exact_top1 = torch.argmax(exact_scores, dim=1)

        approx_scores = torch.matmul(query_batch, quantized_base.T)
        approx_topk = torch.topk(approx_scores, k=max_k, dim=1).indices

        for k in k_values:
            hit_mask = (approx_topk[:, :k] == exact_top1.unsqueeze(1)).any(dim=1)
            hits[k] += int(hit_mask.sum().item())

    return {k: hits[k] / num_queries for k in k_values}


def run_turboquant_recall(
    train_path: Path,
    test_path: Path,
    bitwidth: int,
    variant: str,
    seed: int,
    device: str = "cuda",
    batch_size: int = 256,
    k_values: list[int] | None = None,
    neighbors_path: Path | None = None,
) -> dict[int, float]:
    if device != "cuda":
        raise ValueError(
            "The reference TurboQuant implementation in vector_search/turbo_quant.py uses CUDA."
        )

    turbo_quant = load_turbo_quant_module()
    k_values = k_values or RECALL_KS

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train = np.load(train_path).astype(np.float32, copy=False)
    test = np.load(test_path).astype(np.float32, copy=False)

    base = torch.tensor(train, device=device, dtype=torch.float32)
    queries = torch.tensor(test, device=device, dtype=torch.float32)

    dim = base.shape[1]
    rot = turbo_quant.generate_random_rotation(dim)
    if variant == "mse":
        quantized = turbo_quant.quantize_vectors(base, bitwidth, rot)
    elif variant == "prod":
        quantized = turbo_quant.quantize_vectors_unbiased(base, bitwidth, rot)
    else:
        raise ValueError(f"Unsupported TurboQuant variant: {variant}")

    if neighbors_path is not None:
        neighbors = np.load(neighbors_path)
        ground_truth_top1 = torch.tensor(
            neighbors[:, 0].astype(np.int64, copy=False),
            device=device,
            dtype=torch.long,
        )
        return recall_from_ground_truth(
            queries, quantized, ground_truth_top1, k_values=k_values, batch_size=batch_size
        )

    return recall_from_exact_ip(
        queries,
        base,
        quantized,
        k_values=k_values,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Recall@1@K for TurboQuant mse/prod variants on arbitrary .npy datasets."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--neighbors", type=Path, default=None)
    parser.add_argument("--bitwidth", type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument("--variant", type=str, choices=["mse", "prod"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    recalls = run_turboquant_recall(
        train_path=args.train,
        test_path=args.test,
        neighbors_path=args.neighbors,
        bitwidth=args.bitwidth,
        variant=args.variant,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        k_values=k_values,
    )

    payload = {
        "train": str(args.train.resolve()),
        "test": str(args.test.resolve()),
        "neighbors": str(args.neighbors.resolve()) if args.neighbors else None,
        "bitwidth": args.bitwidth,
        "variant": args.variant,
        "seed": args.seed,
        "device": args.device,
        "k_values": k_values,
        "recall": {str(k): recalls[k] for k in k_values},
    }

    print(
        f"Running TurboQuant-{args.variant} Recall@1@K for bitwidth={args.bitwidth} "
        f"on {args.train.name} (seed={args.seed}, device={args.device})"
    )
    for k in k_values:
        print(f"Recall@{k}: {recalls[k]:.4f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()

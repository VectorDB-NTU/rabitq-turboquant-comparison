# modified based on https://openreview.net/forum?id=tO3ASKZlok

import argparse
import numpy as np
import torch
import math
import csv

CENTROIDS = [
    torch.tensor([-0.797885, 0.797885]),
    torch.tensor([-1.510017, -0.4526475, 0.4526475, 1.510017]),
    torch.tensor([-2.1509, -1.34335, -0.75567, -0.244893, 0.244961, 0.75567, 1.34335, 2.1509]),
    torch.tensor([-2.7235756, -2.0604305, -1.6096783, -1.2484536, -0.9357067, -0.6516434,
                  -0.3848085, -0.12730813, 0.12730813, 0.3848085, 0.6516434, 0.9357067,
                  1.2484536, 1.6096783, 2.0604305, 2.7235756]),
    torch.tensor([-3.0996535, -2.5120323, -2.1263952, -1.829787, -1.5848435, -1.374458, -1.1892526,
                  -1.0232878, -0.872524, -0.7339456, -0.60524833, -0.48459405, -0.37025818,
                  -0.26076207, -0.1548669, -0.05133244, 0.05133244, 0.1548669, 0.26076207,
                  0.37025818, 0.48459405, 0.60524833, 0.7339456, 0.872524, 1.0232878, 1.1892526,
                  1.374458, 1.5848435, 1.829787, 2.1263952, 2.5120323, 3.0996535])
]
NORM_REDUCTION = [1.0, 0.60281, 0.3426687, 0.185758, 0.0974343]


def load_datasets(dim: int):
    train = np.load(f'dbpedia-openai3-{dim}-train.npy')
    test = np.load(f'dbpedia-openai3-{dim}-test.npy')
    return torch.tensor(train, device='cuda', dtype=torch.float32), torch.tensor(test, device='cuda', dtype=torch.float32)


def generate_random_rotation(dim: int) -> torch.Tensor:
    gaussian_matrix = torch.randn(dim, dim)
    q, _ = torch.linalg.qr(gaussian_matrix)
    return q.to('cuda')


def round_to_centroid(data: torch.Tensor, bitwidth: int, dim: int) -> torch.Tensor:
    centroids = CENTROIDS[bitwidth - 1].to('cuda') / math.sqrt(dim)
    dists = torch.abs(data[..., None] - centroids)
    return centroids[torch.argmin(dists, dim=-1)]


def quantize_vectors(vectors: torch.Tensor, bitwidth: int, rot: torch.Tensor) -> torch.Tensor:
    rotated = rot @ vectors.T
    quantized = round_to_centroid(rotated.T, bitwidth, vectors.shape[-1])
    return (rot.T @ quantized.T).T


def quantize_vectors_unbiased(vectors: torch.Tensor, bitwidth: int, rot: torch.Tensor) -> torch.Tensor:
    if bitwidth == 1:
        quantized = torch.zeros_like(vectors)
    else:
        quantized = quantize_vectors(vectors, bitwidth - 1, rot)

    residual = vectors - quantized
    rand_rot = generate_random_rotation(vectors.shape[-1])
    q_residual = (torch.sign(residual @ rand_rot.T) @ rand_rot) * math.sqrt(np.pi / 2.0) / math.sqrt(vectors.shape[-1])
    q_residual *= NORM_REDUCTION[bitwidth - 1]
    return quantized + q_residual


def recall_at_k(test, orig, quant, k) -> float:
    exact = torch.matmul(test, orig.T)
    correct = torch.argmax(exact, dim=1)
    quant_ = torch.matmul(test, quant.T)
    topk = torch.topk(quant_, k=k, dim=1).indices
    return (topk == correct.unsqueeze(1)).any(dim=1).float().mean().item()


def compute_ip_error(test, orig, quant) -> np.ndarray:
    orig_ip = torch.matmul(test, orig.T)
    quant_ip = torch.matmul(test, quant.T)
    return (orig_ip - quant_ip).cpu().numpy().flatten()


def compute_l2_error(orig, quant) -> np.ndarray:
    return torch.norm(orig - quant, p=2, dim=-1).cpu().numpy().flatten()


def run(args):
    dataset, test_dataset = load_datasets(args.dim)
    rot = generate_random_rotation(dataset.shape[-1])

    if args.metric == "recall":
        print(f"Running Recall@K for bitwidth={args.bitwidth}")
        quant = quantize_vectors(dataset, args.bitwidth, rot)
        for k in [1, 2, 4, 8, 16, 32, 64, 128]:
            r = recall_at_k(test_dataset, dataset, quant, k)
            print(f"Recall@{k}: {r:.4f}")

    elif args.metric == "ip-error":
        quant = quantize_vectors(dataset, args.bitwidth, rot)
        error = compute_ip_error(test_dataset, dataset, quant)
        fname = f"ip_error_dim{args.dim}_bw{args.bitwidth}_mse.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'ip_error'])
            for i, e in enumerate(error):
                writer.writerow([i, e])
        print(f"Per-vector errors saved to {fname}")

        quant = quantize_vectors_unbiased(dataset, args.bitwidth, rot)
        error = compute_ip_error(test_dataset, dataset, quant)
        fname = f"ip_error_dim{args.dim}_bw{args.bitwidth}_prod.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'ip_error'])
            for i, e in enumerate(error):
                writer.writerow([i, e])
        print(f"Per-vector errors saved to {fname}")

    elif args.metric == "l2-error":
        print(f"Computing L2 Error (bitwidth={args.bitwidth})")
        quant = quantize_vectors(dataset, args.bitwidth, rot)
        error = compute_l2_error(dataset, quant)
        print(f"Mean L2 Error: {np.mean(error):.4f}, Std: {np.std(error):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate quantization on DBpedia OpenAI3 embeddings.")
    parser.add_argument('--dim', type=int, choices=[3072, 1536], required=True, help='Embedding dimension')
    parser.add_argument('--bitwidth', type=int, choices=[1, 2, 3, 4, 5], required=True, help='Quantization bitwidth')
    parser.add_argument('--metric', type=str, choices=['recall', 'ip-error', 'l2-error'], required=True,
                        help='Metric to compute')

    args = parser.parse_args()
    run(args)

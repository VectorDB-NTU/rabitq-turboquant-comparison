# https://openreview.net/forum?id=tO3ASKZlok

import argparse
import numpy as np
from datasets import load_dataset

def main(dim: int, n: int, test_size: int):
    if dim not in [3072, 1536]:
        raise ValueError("Dim must be either 3072 or 1536")

    dataset_name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    print(f"Sampling {n} random entries from dataset")
    total_size = len(dataset)
    sampled_indices = np.random.choice(total_size, size=n, replace=False)
    dataset_sampled = dataset.select(sampled_indices)

    print("Extracting embeddings")
    x = np.array(dataset_sampled[f'text-embedding-3-large-{dim}-embedding'])

    print("Splitting into train/test")
    test_indices = np.random.choice(n, size=test_size, replace=False)
    train_indices = np.setdiff1d(np.arange(n), test_indices)

    x_train = x[train_indices]
    x_test = x[test_indices]

    np.save(f'dbpedia-openai3-{dim}-train.npy', x_train)
    np.save(f'dbpedia-openai3-{dim}-test.npy', x_test)
    print(f"Saved: dbpedia-openai3-{dim}-train.npy and dbpedia-openai3-{dim}-test.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process DBpedia OpenAI3 embeddings.")
    parser.add_argument('--dim', type=int, choices=[3072, 1536], required=True, help="Embedding dimensionality (3072 or 1536)")
    parser.add_argument('--n', type=int, default=101000, help="Number of total samples to extract")
    parser.add_argument('--test_size', type=int, default=1000, help="Size of the test split")

    args = parser.parse_args()
    main(args.dim, args.n, args.test_size)

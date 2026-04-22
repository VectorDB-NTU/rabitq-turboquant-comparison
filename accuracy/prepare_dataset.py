import numpy as np
import argparse


def npy_to_fvecs(npy_path, fvecs_path):
    data = np.load(npy_path).astype(np.float32)
    n, d = data.shape
    dim = np.array([d], dtype=np.int32)
    with open(fvecs_path, 'wb') as f:
        for row in data:
            f.write(dim.tobytes())
            f.write(row.tobytes())
    print(f"Converted {npy_path} -> {fvecs_path} ({n} vectors, dim={d})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=1536)
    args = parser.parse_args()

    npy_to_fvecs(f'dbpedia-openai3-{args.dim}-train.npy', f'dbpedia-openai3-{args.dim}-train.fvecs')
    npy_to_fvecs(f'dbpedia-openai3-{args.dim}-test.npy', f'dbpedia-openai3-{args.dim}-test.fvecs')

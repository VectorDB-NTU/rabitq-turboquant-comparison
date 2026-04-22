import numpy as np


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

    npy_to_fvecs(f'../dbpedia-openai3-1536-train.npy', f'dbpedia-openai3-1536-train.fvecs')
    npy_to_fvecs(f'../dbpedia-openai3-3072-train.npy', f'dbpedia-openai3-3072-train.fvecs')
    npy_to_fvecs(f'../glove200_train.npy', f'glove200_train.fvecs')
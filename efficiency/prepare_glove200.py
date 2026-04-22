# sample 100,000 vectors from the glove dataset, then stored as .npy format

import numpy as np

src = "wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"

vectors = []
with open(src, "r") as f:
    for line in f:
        parts = line.strip().split()
        # last 200 elements are floats; everything before is the word token
        if len(parts) < 201:
            continue
        vectors.append([float(x) for x in parts[-200:]])

vectors = np.array(vectors, dtype=np.float32)
print(f"Loaded {vectors.shape[0]} vectors of dim {vectors.shape[1]}")

rng = np.random.default_rng(42)
indices = rng.choice(vectors.shape[0], size=101_000, replace=False)
train = vectors[indices[:100_000]]
test = vectors[indices[100_000:]]

np.save("glove200_train.npy", train)
np.save("glove200_test.npy", test)
print(f"Saved glove200_train.npy {train.shape}, glove200_test.npy {test.shape}")

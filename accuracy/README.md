# Accuracy

Compares inner-product estimation errors between **TurboQuant** and **RaBitQ** on OpenAI embeddings (1536-dim).

## 1. Data Preparation

Download the DBpedia OpenAI embedding dataset:

```bash
python get_dataset.py --dim 1536
```

## 2. TurboQuant

Compute IP errors for bitwidths 1–4:

```bash
for bw in 1 2 3 4; do
  python turbo_quant.py --dim 1536 --bitwidth $bw --metric ip-error
done
```

Plot the results:

```bash
python draw_turbo.py
```

## 3. RaBitQ

Compile (requires the `RaBitQ-Library` submodule):

```bash
g++ -std=c++17 -Ofast -march=native -fopenmp -I../RaBitQ-Library/include rabitq.cpp -o rabitq
```

Convert `.npy` data to `.fvecs` format:

```bash
python prepare_dataset.py
```

Compute IP errors for bitwidths 1–4.

```bash
for bw in 1 2 3 4; do
  for type in mse prod; do
    ./rabitq $bw $type
  done
done
```

Plot the results:

```bash
python draw_rabitq.py
```
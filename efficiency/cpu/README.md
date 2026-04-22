# CPU Efficiency

Benchmarks RaBitQ quantization time on CPU across three datasets (GloVe-200d, OpenAI-1536d, OpenAI-3072d).
> **Prerequisite:** Download the datasets first — see the parent [README.md](../README.md).

## 1. Compile

Requires the `RaBitQ-Library` submodule:

```bash
g++ -std=c++17 -Ofast -march=native -fopenmp -I../../RaBitQ-Library/include rabitq.cpp -o rabitq
```

## 2. Prepare Data

Convert `.npy` files to `.fvecs` format:

```bash
python prepare_dataset.py
```

## 3. Run Benchmark

Measure quantization time (4-bit, with and without fast-quantization&FHT):

```bash
for data in glove200_train.fvecs dbpedia-openai3-1536-train.fvecs dbpedia-openai3-3072-train.fvecs; do
    for fast_quant in true false; do
        for bit in 4; do
            ./rabitq $data $bit $fast_quant
        done
    done
done
```
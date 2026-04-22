# ANN Figure Reproduction

This directory contains a self-contained reproduction pipeline for the three-panel ANN figure

`vector_search/figures/recall_at1_three_panel_standard_vs_unbiased_combined.png`

The figure compares:

- `TurboQuant_mse` at 2 bits and 4 bits
- `TurboQuant_prod` at 2 bits and 4 bits
- `RaBitQ` at 2 bits and 4 bits

on the following three datasets:

- GloVe-200
- OpenAI3-1536
- OpenAI3-3072

The final output produced by the scripts in this directory is:

- `figures/recall_at1_three_panel.png`

The figure shows `Recall@1@K` for `K in [1, 2, 4, 8, 16, 32, 64]`. Each curve is the mean over 10 runs, and the shaded band is mean ± standard deviation.

## Directory Layout

- `common.py`
  Shared paths, dataset configuration, and helper utilities.
- `prepare_openai_dbpedia.py`
  Downloads and prepares one OpenAI DBpedia dataset (`1536` or `3072`) into `.npy` and `.fvecs`.
- `prepare_glove200.py`
  Downloads and prepares the normalized GloVe-200 100k subset used by the figure.
- `prepare_all_datasets.py`
  Convenience entry point to prepare all three datasets.
- `run_turboquant_recall.py`
  Runs one TurboQuant variant (`mse` or `prod`) on one dataset.
- `run_rabitq_recall.py`
  Runs the copied `rabitq.cpp` evaluator once and parses `Recall@1@K`.
- `run_all_experiments.py`
  Runs the full 3-dataset, 6-curve experiment and saves aggregated results.
- `plot_recall_figure.py`
  Plots the final three-panel figure from the saved result JSON.
- `rabitq.cpp`
  The RaBitQ evaluator used in this reproduction.
- `CMakeLists.txt`
  Minimal CMake build file for `rabitq.cpp`.
- `requirements.txt`
  Python dependencies for dataset preparation, TurboQuant evaluation, and plotting.

## What Exactly Is Reproduced

This pipeline reproduces the same experiment protocol used for the figure:

- `TurboQuant_mse` uses `quantize_vectors(...)`
- `TurboQuant_prod` uses `quantize_vectors_unbiased(...)`
- both TurboQuant variants reuse the implementation in `vector_search/turbo_quant.py`
- `RaBitQ` uses `rabitq.cpp` in this directory
- `RaBitQ` is run with:
  - `metric=ip`
  - `faster_quant=false`
  - `rotator=matrix`
  - `num_threads=32`
- GloVe is evaluated on the normalized 100k subset
- OpenAI datasets use 100,000 base vectors and 1,000 queries
- all curves are aggregated over 10 runs


## Prerequisites

### Python

Use a Python environment with:

- `numpy`
- `torch`
- `datasets`
- `h5py`
- `Pillow`

From the `ann/` directory, you can install them with:

```bash
pip install -r requirements.txt
```

### C++ / Build Tools

To build `rabitq.cpp`, you need:

- `cmake >= 3.10`
- a C++17 compiler
- OpenMP

The build uses the headers in the sibling library checkout:

- `../../RaBitQ-Library/include`

## Step 1: Prepare the Datasets

All processed data are written under:

- `data/processed/`

Raw downloaded files are written under:

- `data/raw/`

To prepare all three datasets in one command:

```bash
python prepare_all_datasets.py
```

This does the following:

1. Downloads `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`
2. Samples `101000` vectors with seed `42`
3. Splits them into:
   - `100000` train/base vectors
   - `1000` test/query vectors
4. Saves:
   - `dbpedia-openai3-1536-train.npy`
   - `dbpedia-openai3-1536-test.npy`
   - `dbpedia-openai3-1536-train.fvecs`
   - `dbpedia-openai3-1536-test.fvecs`
5. Repeats the same process for `3072` dimensions
6. Downloads `glove-200-angular.hdf5`
7. Samples `100000` training vectors with seed `42`
8. Keeps the full `10000`-query test set
9. L2-normalizes both train and test
10. Recomputes exact angular ground truth against the sampled 100k subset
11. Saves:
   - `glove-200-angular-normalized-100k-train.npy`
   - `glove-200-angular-normalized-100k-test.npy`
   - `glove-200-angular-normalized-100k-neighbors.npy`
   - `glove-200-angular-normalized-100k-train.fvecs`
   - `glove-200-angular-normalized-100k-test.fvecs`

If you want to run the preparation scripts individually:

```bash
python prepare_openai_dbpedia.py --dim 1536
python prepare_openai_dbpedia.py --dim 3072
python prepare_glove200.py
```

## Step 2: Build the RaBitQ Evaluator

From the `ann/` directory:

```bash
cmake -S . -B build
cmake --build build -j
```

This produces:

- `build/bin/rabitq`

## Step 3: Optional Single-Method Sanity Checks

### TurboQuant

Run one TurboQuant variant on one dataset:

```bash
python run_turboquant_recall.py \
  --train data/processed/dbpedia-openai3-1536-train.npy \
  --test data/processed/dbpedia-openai3-1536-test.npy \
  --bitwidth 2 \
  --variant mse \
  --seed 42
```

For the unbiased variant:

```bash
python run_turboquant_recall.py \
  --train data/processed/dbpedia-openai3-1536-train.npy \
  --test data/processed/dbpedia-openai3-1536-test.npy \
  --bitwidth 2 \
  --variant prod \
  --seed 42
```

For GloVe, add the ground-truth neighbors:

```bash
python run_turboquant_recall.py \
  --train data/processed/glove-200-angular-normalized-100k-train.npy \
  --test data/processed/glove-200-angular-normalized-100k-test.npy \
  --neighbors data/processed/glove-200-angular-normalized-100k-neighbors.npy \
  --bitwidth 2 \
  --variant mse \
  --seed 42
```

### RaBitQ

Run the RaBitQ evaluator once:

```bash
python run_rabitq_recall.py \
  --binary build/bin/rabitq \
  --base data/processed/dbpedia-openai3-1536-train.fvecs \
  --query data/processed/dbpedia-openai3-1536-test.fvecs \
  --bitwidth 2
```

For GloVe:

```bash
python run_rabitq_recall.py \
  --binary build/bin/rabitq \
  --base data/processed/glove-200-angular-normalized-100k-train.fvecs \
  --query data/processed/glove-200-angular-normalized-100k-test.fvecs \
  --neighbors data/processed/glove-200-angular-normalized-100k-neighbors.npy \
  --bitwidth 2
```

## Step 4: Run the Full Experiment

This is the main reproduction command:

```bash
python run_all_experiments.py \
  --rabitq-binary build/bin/rabitq \
  --turbo-device cuda \
  --turbo-repeats 10 \
  --rabitq-repeats 10 \
  --turbo-seed-start 42 \
  --num-threads 32
```

This writes:

- `results/recall_at1_three_panel.json`

That JSON contains:

- all 10 raw runs for every curve
- the mean at each `K`
- the standard deviation at each `K`

## Step 5: Plot the Figure

Use the saved results to draw the final figure:

```bash
python plot_recall_figure.py
```

This writes:

- `figures/recall_at1_three_panel.png`
- `figures/recall_at1_three_panel.json`
- `figures/recall_at1_three_panel.csv`

The PNG is the final figure. The JSON and CSV are the plot-ready mean/std tables.

## Output File Summary

### Processed datasets

- `data/processed/dbpedia-openai3-1536-train.npy`
- `data/processed/dbpedia-openai3-1536-test.npy`
- `data/processed/dbpedia-openai3-1536-train.fvecs`
- `data/processed/dbpedia-openai3-1536-test.fvecs`
- `data/processed/dbpedia-openai3-3072-train.npy`
- `data/processed/dbpedia-openai3-3072-test.npy`
- `data/processed/dbpedia-openai3-3072-train.fvecs`
- `data/processed/dbpedia-openai3-3072-test.fvecs`
- `data/processed/glove-200-angular-normalized-100k-train.npy`
- `data/processed/glove-200-angular-normalized-100k-test.npy`
- `data/processed/glove-200-angular-normalized-100k-neighbors.npy`
- `data/processed/glove-200-angular-normalized-100k-train.fvecs`
- `data/processed/glove-200-angular-normalized-100k-test.fvecs`

### Experiment results

- `results/recall_at1_three_panel.json`

### Figure outputs

- `figures/recall_at1_three_panel.png`
- `figures/recall_at1_three_panel.json`
- `figures/recall_at1_three_panel.csv`


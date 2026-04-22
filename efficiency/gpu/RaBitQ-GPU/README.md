# RaBitQ-GPU

Standalone GPU implementation of RaBitQ scalar quantization.

The codes provides:

1. Two random-rotation implementations on GPU:
   - `FhtKac` — Fast Hadamard Transform + Kac's walk, `O(N·D·logD)`.
   - `Matrix` — random orthogonal matrix via cuBLAS SGEMM, `O(N·D²)`.
2. A quantizer that maps rotated residuals to RaBitQ
   scalar codes with configurable code width (1–9 bits).
3. Two per-vector factor modes:
   - **scalar**: `(delta, vl)` — enables direct reconstruction
     `x̂[d] = code[d] · delta + vl`.
   - **full**: `(f_add, f_rescale, f_error)` — the distance-estimation
     triplet used for approximate nearest-neighbor search.
4. A benchmark executable (`test_quantizer_standalone`) that reports timing
   (with / without rotation) and reconstruction error (per-vector L2 / IP,
   optionally full-pairwise IP against a test set).


## Environment

Requires CUDA Driver 13.x.

Create a dedicated mamba environment:

```bash
mamba create -n rabitq_gpu -c conda-forge -c nvidia -y \
    cuda-version=13.1 \
    cuda-nvcc cuda-cudart-dev cuda-cccl cuda-profiler-api \
    libcublas-dev libcurand-dev \
    cmake \
    gcc_linux-64=14.* gxx_linux-64=14.* sysroot_linux-64==2.28

mamba activate rabitq_gpu
```

## Build

Make sure the mamba environment is activated — `mamba activate rabitq_gpu`.
The env's activation scripts set `CC` and `CXX` to the bundled gcc 14
automatically; nothing else needs to be exported.

```bash
cd RaBitQ-GPU
mkdir -p build && cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

The executable lands at `RaBitQ-GPU/bin/test_quantizer_standalone`.

By default CMake targets SM 80, 86, 89, and 120 (A100 / A5000 / 4090 /
RTX PRO 6000 Blackwell). Override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`
if you only want one, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` for A100.


## Run

```
./bin/test_quantizer_standalone <base.fvecs> <total_bits> [fast] [num_vectors] \
                                 [test.fvecs] [delta_mode] [rotator] [factor_mode]
```

Positional args — later ones may be omitted:

| Arg           | Meaning                                                                 |
| ------------- | ----------------------------------------------------------------------- |
| `base.fvecs`  | Input vectors in `.fvecs` format.                                       |
| `total_bits`  | Scalar code width, 1..9. `ex_bits = total_bits - 1`.                    |
| `fast`        | `true` (default) uses the precomputed const scaling factor; `false` runs the per-vector search path. |
| `num_vectors` | 0 or omit = use all rows; otherwise a prefix subset.                    |
| `test.fvecs`  | Optional. If provided, reports full-pairwise IP error vs the test set. Pass `-` to skip. |
| `delta_mode`  | `0` = RECONSTRUCTION (default), `1` = UNBIASED, `2` = PLAIN. Scalar mode only. |
| `rotator`     | `fhtkac` (default) or `matrix`.                                          |
| `factor_mode` | `scalar` (default, emits `delta`, `vl`) or `full` (emits `f_add`, `f_rescale`, `f_error`). |

### Example

```bash
./bin/test_quantizer_standalone \
    ../data/openai1536_100k/base.fvecs \
    4 true 0 \
    ../data/openai1536_100k/test.fvecs \
    0 matrix
```

## Data format

`.fvecs` layout per vector: `[int32 dim][float32 × dim]`, repeated.

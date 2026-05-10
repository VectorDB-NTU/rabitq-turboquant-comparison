# Quantization Efficiency Results

Corresponds to Section 4.2 of the paper.

## Setup

Following the TurboQuant paper, we use three datasets including GloVe-200 (200-dimensional) and two DBpedia Entities datasets (1536-dimensional and 3072-dimensional). Specifically, we sample 100,000 vectors from each dataset and quantize the sampled vectors with 4 bits per dimension.

For RaBitQ, we test four implementations:

1. **RaBitQ**: the default implementation with the `faster_quant` flag disabled and a random orthogonal matrix for vector rotation (consistent with TurboQuant)
2. **RaBitQ_fastOn-FWHT**: a faster implementation of RaBitQ with the `faster_quant` flag enabled and Fast Walsh-Hadamard Transform (FWHT) from the FFHT Library and ideas in Kac's Walk for faster vector rotation
3. **RaBitQ (GPU)**: a standalone GPU-based implementation of RaBitQ
4. **RaBitQ_fastOn-FWHT (GPU)**: the GPU version of RaBitQ with the `faster_quant` flag enabled and using FWHT and ideas in Kac's Walk for faster vector rotation

For RaBitQ, we use it for reproducibility consideration since the TurboQuant paper implements a Python version of this implementation and uses it for evaluating the performance of RaBitQ. For RaBitQ (GPU), we use it for a more direct comparison with TurboQuant since the TurboQuant implementation runs on GPU.

We collect the running time of the methods, with the time for rotating the vectors included. For GPU-based implementations, the data transfer time from main memory (host memory) to GPU memory (device memory) is excluded.

### Hardware

- Nvidia A100 GPU (80 GiB VRAM) with 16 VCPUs (cloud instance, following the original setup of the papers)
- A dual-socket server equipped with two Intel Xeon Gold 6418H processors (48 cores / 96 threads in total)

## Table 2: Quantization time (in seconds) for different approaches across various dimensions using 4-bit quantization

| Approach | d=200 | d=1536 | d=3072 |
|----------|-------|--------|--------|
| RaBitQ (CPU) | 0.125 | 1.003 | 4.176 |
| RaBitQ_fastOn-FWHT (CPU) | 0.085 | 0.143 | 0.218 |
| RaBitQ (GPU) | 0.009 | 0.065 | 0.152 |
| RaBitQ_fastOn-FWHT (GPU) | 0.003 | 0.008 | 0.013 |
| TurboQuant (GPU) | 0.011 | 0.114 | 0.276 |

## Observations

### RaBitQ is faster than TurboQuant on the same hardware

When compared on the same hardware (i.e., GPU), RaBitQ is substantially faster than TurboQuant. The GPU implementation of RaBitQ outperforms TurboQuant across all three datasets by a large margin: it is approximately 1.2×, 1.8×, and 1.8× faster at *d* = 200, *d* = 1,536, and *d* = 3,072, respectively. Moreover, with `faster_quant` flag enabled and FWHT, the advances of RaBitQ are more dominant.

### RaBitQ on CPU is competitive with TurboQuant on GPU

Even the CPU implementation of RaBitQ (RaBitQ_fastOn-FWHT), which runs on a standard multi-core server without GPU, achieves quantization times within the same order of magnitude as TurboQuant running on an A100 GPU, despite the significant hardware gap.

### Discrepancy with reported RaBitQ results in the TurboQuant paper

The quantization times we observe for RaBitQ differ substantially from those reported in the TurboQuant paper. This discrepancy is explained by the asymmetric experimental conditions used in the TurboQuant paper. According to our private correspondence with the TurboQuant authors, their experiments evaluated RaBitQ on a single-core CPU with multi-threading disabled, while evaluating TurboQuant on an A100 GPU. The TurboQuant paper also implements a Python version of RaBitQ for the evaluation. These asymmetric setups were not disclosed in the TurboQuant paper.

> The second author of TurboQuant, Majid Daliri, stated in email correspondence that "we were using a single-core CPU instance, and multiprocessing was indeed disabled […] we weren't fully utilizing parallelism, which explains why it was significantly slower".

### Discrepancy with reported TurboQuant results in the TurboQuant paper

The quantization times we observe for TurboQuant also differ substantially from those reported in the TurboQuant paper, and the nature of this discrepancy is different. Even when we evaluate TurboQuant using the officially released implementation on the same A100 GPU hardware reported in the paper, we observe quantization times up to approximately two orders of magnitude slower than those reported in the TurboQuant paper. This suggests that the quantization times reported in the TurboQuant paper are not reproducible from the released implementation under the stated hardware configuration.

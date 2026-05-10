# KV Cache Quantization Results

Corresponds to Section 4.4 of the paper.

## Setup

### Choice of TurboQuant variant

We compare RaBitQ with TurboQuant for KV cache quantization in long-context generation. We note that the TurboQuant paper does not specify which of its two variants was used for KV cache quantization. Moreover, the open-source community has observed that the QJL-based variant (TurboQuant_prod) can hurt attention quality by amplifying variance through the softmax operation (see [vLLM documentation](https://docs.vllm.ai/en/v0.20.0/api/vllm/model_executor/layers/quantization/turboquant/)). We therefore use TurboQuant_mse for this comparison.

### Executability of the released TurboQuant code

The released code of TurboQuant contains the core quantization routines (including `TurboSketch`, centroid tables, and outlier separation), an attention layer that hardcodes TurboQuant as the only backend, and a LongBench evaluation script. However, the code cannot be executed as released: it depends on an unpublished CUDA kernel package and contains multiple bugs in the quantization pipeline. For example, the value-cache quantizer is never constructed, and the decode-phase quantization logic is unreachable due to an early return.

We fix these bugs and provide a unified KV cache framework in which RaBitQ and TurboQuant share identical cache logic, buffer management, and outlier handling, while retaining the original MSE-based centroids. Full details are available in our released code. In our evaluation, the quantization method is the only varying factor in the KV cache framework, while all other configurations are kept identical.

### Outlier-aware bit allocation

For a direct comparison, both methods use the same outlier-aware key-cache bit allocation. For each attention head with *d_h* = 128, the 32 key channels with the largest L2 norm are quantized at a higher bitwidth, while the remaining 96 channels use a lower bitwidth. Each quantized key vector stores two additional float16 scaling values, namely one for the sub-vector consisting of outlier channels and the other for the sub-vector consisting of remaining channels.

We evaluate two key-cache configurations:

- **2.5-bit**: 3-bit outlier channels and 2-bit non-outlier channels, corresponding to effective bitwidth (32 x 3 + 96 x 2 + 2 x 16) / 128 = 2.5
- **3.5-bit**: 4-bit outlier channels and 3-bit non-outlier channels, corresponding to effective bitwidth (32 x 4 + 96 x 3 + 2 x 16) / 128 = 3.5

Values are quantized uniformly at 2 bits, so the 2.5-bit and 3.5-bit labels refer to the key-cache configuration.

For LongBench-E, we use the official task metrics with the same output post-processing convention as the released TurboQuant evaluation code.

## Needle-In-A-Haystack

### Setup

We evaluate retrieval behavior on `meta-llama/Meta-Llama-3.1-8B-Instruct` using Needle-In-A-Haystack across 15 context lengths (4k-104k tokens) and 10 needle depths, yielding 150 test points per method.

The released TurboQuant code does not include an NIAH evaluation script. We build our evaluation on the official LLMTest_NeedleInAHaystack framework. However, its default GPT-3.5-turbo judge produces inconsistent scores for the same model output across repeated evaluations, making results difficult to reproduce. We therefore replace it with the keyword-coverage scorer used by Token-Sparse-Attention, which measures the fraction of expected-answer words that appear in the model output, yielding a deterministic metric in [0, 1].

One additional issue is that the NIAH framework constructs haystacks by concatenating Paul Graham essays loaded via `glob.glob`, whose iteration order is filesystem-dependent and therefore non-deterministic. We record the glob ordering observed on our machine and provide it in the released code for reproducibility.

### Figure 4: Evaluation of `meta-llama/Meta-Llama-3.1-8B-Instruct` on the "Needle-In-A-Haystack" test

![Needle-in-a-Haystack](figures/needle_heatmap.png)

Five panels left to right: Full-Precision (0.987), RaBitQ 2.5-bit (0.951), RaBitQ 3.5-bit (0.977), TurboQuant 2.5-bit (0.709), TurboQuant 3.5-bit (0.962).

### Results

The full-precision baseline scores 0.987. RaBitQ remains close to this level at both 2.5-bit and 3.5-bit, scoring 0.951 and 0.977, respectively. TurboQuant_mse also performs well at 3.5-bit (0.962), but drops to 0.709 at 2.5-bit: 86 out of 150 test points score below 0.8.

The failures are widespread across nearly all needle depths (only depth = 100% is fully correct) and concentrate at longer contexts, where the mean score falls from 0.898 (<=32k) to 0.615 (>32k). This suggests that the MSE-based centroid placement, while adequate at higher bitwidths, introduces sufficient approximation error at 2.5-bit to distort attention scores over long sequences, causing the model to fail to attend to the relevant passage.

## LongBench-E

### Setup

We evaluate on all 13 datasets of LongBench-E using `meta-llama/Meta-Llama-3.1-8B-Instruct` and `mistralai/Ministral-8B-Instruct-2410`, grouped into 6 categories.

> We note that the TurboQuant paper reports results for "Ministral-7B-Instruct", which does not correspond to any model available on public model hubs. It might mean Mistral-7B-Instruct, but the model exists in three versions and the paper does not specify which was used. Therefore, we adopt the unambiguous Ministral-8B-Instruct-2410 instead.

Category scores are computed as the mean of the dataset-level scores within each category. Following the TurboQuant reporting convention, the overall average is computed over all 13 dataset-level scores rather than over the 6 category scores. We follow the TurboQuant paper and explore the bitwidths of 2.5 bits and 3.5 bits on Llama and 2.5 bits on Ministral.

### Table 3: LongBench-E results for `meta-llama/Meta-Llama-3.1-8B-Instruct` and `mistralai/Ministral-8B-Instruct-2410`

#### `meta-llama/Meta-Llama-3.1-8B-Instruct`

| Method | SingleQA | MultiQA | Summ | Few shot | Synthetic | Code | Avg |
|--------|----------|---------|------|----------|-----------|------|-----|
| Full Cache (16-bit) | 45.39 | 45.76 | 26.38 | 68.60 | 59.12 | 48.00 | 50.39 |
| **RaBitQ 2.5-bit** | **43.74** | **45.49** | **22.56** | **67.38** | **58.96** | **44.34** | **48.64** |
| TurboQuant_mse 2.5-bit | 42.20 | 44.36 | 21.89 | 67.37 | 58.94 | 42.14 | 47.78 |
| RaBitQ 3.5-bit | **44.85** | 45.56 | 24.70 | 67.90 | **59.53** | **45.58** | 49.55 |
| TurboQuant_mse 3.5-bit | 44.11 | **45.75** | **25.17** | **68.11** | 59.49 | 45.53 | **49.57** |

#### `mistralai/Ministral-8B-Instruct-2410`

| Method | SingleQA | MultiQA | Summ | Few shot | Synthetic | Code | Avg |
|--------|----------|---------|------|----------|-----------|------|-----|
| Full Cache (16-bit) | 51.29 | 57.28 | 25.92 | 69.37 | 58.16 | 56.10 | 54.28 |
| **RaBitQ 2.5-bit** | **49.39** | **56.39** | **22.90** | **68.71** | **58.00** | **52.16** | **52.60** |
| TurboQuant_mse 2.5-bit | 47.86 | 55.58 | 21.30 | 68.64 | **58.00** | 51.00 | 51.80 |

### Analysis

Table 3 shows the same trend at 2.5-bit: RaBitQ achieves higher average scores than TurboQuant_mse on both models, with 48.64 vs. 47.78 on `meta-llama/Meta-Llama-3.1-8B-Instruct` and 52.60 vs. 51.80 on `mistralai/Ministral-8B-Instruct-2410`. The largest category-level gains appear on Code (+2.20 on Llama and +1.16 on Ministral), where generation depends strongly on long-range contextual consistency. At 3.5-bit on `meta-llama/Meta-Llama-3.1-8B-Instruct`, the two methods are comparable, both close to the full-cache baseline of 50.39. Overall, RaBitQ shows clearer gains at 2.5-bit, while the two methods become comparable as the bitwidth increases.

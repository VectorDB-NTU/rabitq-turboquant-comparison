# Nearest Neighbor Search Results

Corresponds to Section 4.3 of the paper.

## Setup

Following the TurboQuant paper, we use three datasets, namely GloVe-200, OpenAI3-1536, and OpenAI3-3072. For each dataset, we construct a base set and a query set.

- **OpenAI3-1536** and **OpenAI3-3072**: the base set has 100,000 vectors and the query set contains 1,000 vectors
- **GloVe-200**: we sample a subset of 100,000 vectors from the original corpus as the base set and use the provided query set of 10,000 vectors

Following the TurboQuant paper, for all three datasets, we use the inner product of normalized vectors as the metric for nearest neighbor search.

### Methods compared

We compare RaBitQ, namely RaBitQ_prod, with the two TurboQuant variants, namely TurboQuant_prod and TurboQuant_mse. Note that we exclude RaBitQ_mse from this comparison as it is not designed for inner-product estimation, which is the objective underlying nearest neighbor search. On the other hand, we include both variants of TurboQuant in the comparison for transparency as the TurboQuant paper did not specify the version they used in the experiment.

For each method, we first quantize the vectors in the base set and then find for each query vector in the query set, the *k* vectors whose quantized vectors have the largest estimated inner products with the query vector. We vary the bit-width *B* in {2, 4}.

We report Recall@1@*k* for *k* ∈ {1, 2, 4, 8, 16, 32, 64}. Let *g*(**q**) denote the exact top-1 nearest neighbor of query **q** (i.e., the one with the largest inner product), and let *A_k*(**q**) denote the approximate top-*k* result set returned by a method. Then

```
Recall@1@k = (1/|Q|) · Σ_{q∈Q} 𝟙[ g(q) ∈ A_k(q) ].
```

### Note on TurboQuant evaluation scripts

In the open-sourced code of TurboQuant, the evaluation script for the two OpenAI datasets is available, but that for the GloVe-200 data is not. Therefore, for the OpenAI datasets, we use the provided evaluation script directly; and for GloVe-200, we use a thin wrapper that calls the same TurboQuant core routines for random rotation generation and quantization; thus, the underlying TurboQuant quantizer itself is unchanged.

### Handling run-to-run variation

In addition, we note that both RaBitQ and TurboQuant involve randomness through their sampled rotation matrices. As a result, recall curves from a single run may exhibit mild run-to-run variation. To obtain a more stable comparison, we repeat each configuration, defined by method, bit-width, and dataset, 10 times using the full query set. We plot the mean recall over these runs as the main curve, and use the shaded band to represent one standard deviation around the mean.

## Figure 3: Recall comparison on different datasets

![Recall Comparison](figures/recall_at1_three_panel.png)

Three subplots correspond to GloVe-200, OpenAI3-1536, and OpenAI3-3072 respectively.

## Observations

### Overall comparison

Across all three datasets and both bit widths, RaBitQ consistently achieves higher recall than both TurboQuant variants. The advantage is most pronounced at small *k* and at the lower bit width of 2 bits, where the methods are most differentiated. As *k* increases, all methods converge toward perfect recall and the differences diminish accordingly.

### TurboQuant_mse outperforms TurboQuant_prod on recall

We observe that TurboQuant_mse consistently achieves higher recall than TurboQuant_prod across all settings. This is a notable finding because TurboQuant_prod is the variant specifically designed for inner-product estimation, which is the objective directly relevant to nearest neighbor search. The fact that the reconstruction-oriented variant yields better recall performance raises questions about which variant should be used in practice for this task, and about the theoretical guarantees that support TurboQuant_prod in this setting. We note that TurboQuant_mse does not guarantee unbiased inner-product estimation. The TurboQuant paper does not clearly specify which variant is used in its reported recall results.

### Discrepancy with results reported in the TurboQuant paper

We note that the recall values we obtain for RaBitQ differ from those reported in the TurboQuant paper. Specifically, the RaBitQ results reported therein fall below the one-standard-deviation band we measure across 10 repeated runs, each using the full query set, with different random seeds. The TurboQuant paper does not describe how run-to-run variation due to random rotation is handled in their reported RaBitQ results, making it difficult to assess the source of this discrepancy. Our results, by contrast, are averaged over 10 independent runs with standard deviations reported, and are fully reproducible from the code provided in our repository. These reproduced results do not support the TurboQuant paper's conclusion that TurboQuant consistently outperforms RaBitQ in nearest neighbor search.

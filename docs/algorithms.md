# Comparison of Methodology

This document corresponds to Section 2 of the paper, comparing the two methods under a unified framework.

## Table 1: Comparison of RaBitQ and TurboQuant

| | **RaBitQ** | **TurboQuant** |
|---|---|---|
| Preprocessing | Random rotation/JL Transformation. | Random rotation/JL Transformation. |
| Codebook | Uniform codebook obtained by shifting unsigned integers. | Non-uniform codebook constructed by *k*-means. |
| Quantization Algorithm | Select a rescaling factor *t*, rescale the rotated and normalized vector, and quantize each coordinate to its nearest codebook entry. A scalar factor is additionally stored for vector reconstruction or inner-product estimation. | Quantize each coordinate of the rotated and normalized vector to its nearest codebook entry, and store the norm of the input vector. For unbiased inner-product estimation, TurboQuant additionally applies QJL to the residual. |
| Quantization Code | One scalar and *D* unsigned integers of *B* bits each. | One scalar and *D* unsigned integers of *B* bits each; for unbiased inner-product estimation, one additional scalar is required. |
| Inner Product Estimation | Based on native arithmetic of unsigned integers without decoding. | Requires codebook lookup for decoding. |
| Theoretical Guarantee | **Optimal**: provides a sub-Gaussian tail bound on the error: the required bit-width scales as log log(1/δ). | **Suboptimal**: provides a bound on the mean squared error: the required bit-width scales as log(1/δ). |

## Problem Setting

Both RaBitQ and TurboQuant are generic vector quantization methods for high-dimensional vectors in Euclidean space, aiming to preserve geometric quantities from compressed representations. In applications such as approximate nearest neighbor search and LLM systems, a central goal is to preserve inner products between vectors. (In approximate nearest neighbor search, preserving inner products can also help preserve Euclidean distances, since computing Euclidean distances can be reduced to computing inner products.)

A quantization algorithm operates in two stages. In the quantization stage, each input data vector is mapped to a compact representation called a *quantization code*. In the estimation stage, the quantization code is used to estimate the inner product between the data vector and an arbitrary query vector.

The performance of a quantization algorithm is usually evaluated along five dimensions: (1) quantization time, (2) space consumption, (3) accuracy of inner-product estimation, (4) inner-product estimation time, and (5) theoretical guarantees. Throughout the paper, ‖·‖ denotes the ℓ₂ norm unless otherwise specified.

## Preprocessing of Both RaBitQ and TurboQuant

Both RaBitQ and TurboQuant encode the norm and direction of a vector separately. In practice, once the norm is stored, the core quantization procedure of each method focuses on quantizing the normalized vector.

Both RaBitQ and TurboQuant apply a random rotation, which is a form of Johnson–Lindenstrauss Transformation, as the first step for all vectors. Both methods use the distributional information of vectors after random rotation to design their quantization algorithms. Specifically, both methods sample and store a random rotation matrix, and apply the same random rotation to all vectors. In the following sections, without further specification, we assume that all vectors have been rotated by this random matrix.

## Quantization of RaBitQ

RaBitQ constructs its codebook from shifted grids of unsigned integers. Let *B* denote the bit-width per dimension. For an input vector **x**, RaBitQ first rescales the vector by a factor *t*, then rounds each coordinate of the rescaled vector *t*·**x** to the nearest point in a scalar codebook:

```
{ i - (2^B - 1)/2  |  i = 0, 1, …, 2^B - 1 },
```

and stores the corresponding unsigned integer for each coordinate.

Across the RaBitQ line and its implementations, three strategies are used to decide the rescaling factor *t*, where the first two strategies decide *t* on a per-vector basis and the last strategy uses the same *t* for all vectors:

- enumerating all critical rescaling factors that yield distinct quantization codes and selecting the one that maximizes the cosine similarity between the original vector and its quantized counterpart (Gao et al., 2025b);
- enumerating candidate rescaling factors from a prescribed set and selecting the one that maximizes the same cosine similarity (Shi et al., 2026);
- sampling random vectors uniformly from the unit sphere, precomputing the optimal rescaling factor for each, and using the expected value of these optimal factors for fast quantization (Gao et al., 2025a).

Let **x_u** ∈ {0, 1, …, 2^B - 1}^D denote the vector of *B*-bit unsigned integers produced by the procedure above, and define its shifted-grid representation as

```
x̂ := x_u - (2^B - 1)/2 · 1_D,
```

where **1_D** is the all-ones vector in ℝ^D. The vector **x̂** determines the quantized direction; an additional scalar factor is stored to incorporate the norm of the original vector and to support different objectives.

Let cos(**a**, **b**) := ⟨**a**/‖**a**‖, **b**/‖**b**‖⟩ denote the cosine similarity of two vectors. For unbiased inner-product estimation, RaBitQ stores the scalar

```
(‖x‖ / ‖x̂‖) · (1 / cos(x, x̂)).
```

We note that while RaBitQ was originally designed for unbiased inner-product estimation, it has also been adapted for vector reconstruction in the RaBitQ library. Specifically, to instead minimize the reconstruction error, it suffices to replace the scaling factor with

```
(‖x‖ / ‖x̂‖) · cos(x, x̂).
```

## Estimation of RaBitQ

Given a query vector, RaBitQ estimates the inner products between the data vectors and the query vector using the quantized representations. Since all data vectors have been rotated, RaBitQ rotates the query vector by the same matrix to preserve inner products; let **y** denote this rotated query vector.

RaBitQ estimates the inner product between a data vector **x** and the vector **y** as follows:

```
⟨x, y⟩ ≈ (‖x‖ / ‖x̂‖) · (1 / cos(x, x̂)) · ⟨x̂, y⟩.
```

Based on the distribution of vectors after Johnson-Lindenstrauss Transformation, as proved in Gao & Long (2024) and Gao et al. (2025b), the above estimator is unbiased and has a rigorous error bound. The scalar factor (‖**x**‖ / ‖**x̂**‖) · (1 / cos(**x**, **x̂**)) in the estimator is precomputed and stored during the quantization stage. The remaining term ⟨**x̂**, **y**⟩ is computed as

```
⟨x̂, y⟩ = ⟨x_u, y⟩ - (2^B - 1)/2 · Σ_{i=1}^D y[i],
```

where **x_u** is the stored *B*-bit code of the data vector. The term Σ y[i] depends only on the rotated query and can be computed once and reused across all data vectors. As a result, RaBitQ computes inner-product estimates directly from the compressed representation, i.e., **x_u**, without any decoding step.

Furthermore, the structure of RaBitQ's quantization code naturally supports incremental estimation. It can decompose a quantization code into two parts, e.g., the most significant bit and the remaining bits. During estimation, RaBitQ can first produce a coarse estimate of the inner product by accessing only the most significant bit. When higher accuracy is needed, it can access the remaining bits to refine the estimate, which helps significantly speed up the estimation in practice.

When using RaBitQ for vector reconstruction, based on the precomputed scalar factor (‖**x**‖ / ‖**x̂**‖) · cos(**x**, **x̂**), RaBitQ can reconstruct a vector **x** as follows:

```
x ≈ (‖x‖ / ‖x̂‖) · cos(x, x̂) · x̂.
```

## Quantization of TurboQuant

The TurboQuant method includes two variants: one optimized for vector reconstruction and the other for unbiased inner-product estimation.

For vector reconstruction, TurboQuant constructs a scalar codebook according to the Lloyd–Max condition. Specifically, after normalization and random rotation, the coordinates of a rotated vector follow the distribution induced by the uniform spherical measure, as characterized in Khokhlov (2006). For a target bit-width of *B*, TurboQuant constructs a scalar codebook with 2^B centroids by solving the corresponding one-dimensional continuous *k*-means problem under this distribution. Each coordinate is then quantized to the index of its nearest centroid, and the compressed representation stores these centroid indices for all coordinates. Note that it can compute the reconstructed vector, denoted by **x̄**, by looking-up the codebook based on the stored indices.

For inner-product estimation, TurboQuant introduces a residual-correction stage. Given a total budget of *B* bits per coordinate, it first applies (*B*-1) bits to obtain a reconstruction, denoted by **x̄**, based on the quantization algorithm for vector-reconstruction, and then computes the residual

```
r = x / ‖x‖ - x̄.
```

TurboQuant then applies Quantized Johnson–Lindenstrauss (QJL) (Zandieh et al., 2025b) transform to this residual:

```
q = sign(S r),
```

where **S** is a *D* × *D* random Gaussian matrix and sign(·) is the sign function where sign(*x*) = +1 if *x* ≥ 0 and sign(*x*) = -1 if *x* < 0. In addition to the first-stage quantization codes and the sign vector **q**, the quantized representation stores the vector's norm ‖**x**‖ and the residual norm ‖**r**‖.

## Estimation of TurboQuant

Given a query vector, similarly, TurboQuant estimates the inner products between the data vectors and the query vector using the quantized representations. Since all data vectors have been rotated, TurboQuant also rotates the query vector by the same matrix to preserve inner products; let **y** denote this rotated query vector.

To estimate the inner products between the data vectors and a query vector, TurboQuant combines the first-stage quantization code (using (*B*-1) bits per dimension) with a QJL-based estimator of the residual (using 1 bit per dimension) (Zandieh et al., 2025b) as follows:

```
⟨x, y⟩ ≈ ‖x‖ · ⟨ x̄ + sqrt(π/2) · (‖r‖/D) · S^T q ,  y ⟩
       = ‖x‖ · ⟨x̄, y⟩ + sqrt(π/2) · (‖x‖ · ‖r‖ / D) · ⟨q, S y⟩
```

where **x̄** corresponds to the reconstructed vector based on the quantization codes with (*B*-1) bits per dimension. This estimator is unbiased as proved in Zandieh et al. (2025a). The first component of the estimator still requires decoding the quantization code through the scalar codebook, while the second component uses the stored sign vector and residual norm to correct the bias.

When using TurboQuant for vector reconstruction, we can reconstruct a vector **x** as follows:

```
x ≈ ‖x‖ · x̄.
```

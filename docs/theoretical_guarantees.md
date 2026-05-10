# Comparison of Theoretical Guarantees

This document corresponds to Section 3 of the paper, comparing in detail the theoretical guarantees of RaBitQ and TurboQuant on inner-product estimation.

## Scope

We focus on the inner-product-oriented variant of each method, since the reconstruction-oriented variant is optimized for reconstruction error and does not provide unbiased inner-product estimation. Under this scope, both RaBitQ and TurboQuant provide unbiased estimators of the inner product between unit vectors.

## Probabilistic Guarantee Form

We first note that both RaBitQ and TurboQuant are randomized algorithms whose estimation error is a random variable. Rather than providing a deterministic guarantee, both methods can provide a probabilistic guarantee:

> the additive error of inner product between unit vectors is bounded by ε with probability at least 1 - δ, where ε, δ ∈ (0, 1).

The key quantity of interest is therefore the trade-off among the error bound ε, the failure probability δ, and the bit-width *B*.

## Optimal Trade-off (Alon & Klartag, 2017)

In 2017, Alon and Klartag established the optimal trade-off for approximate inner-product sketches under additive-error guarantees, providing matching upper and lower bounds on the bit-width *B* required to ensure that the additive error of inner-product estimation between unit vectors is bounded by ε with probability at least 1 - δ. Specifically, as adapted from the proof of Theorem 4.1 in Alon & Klartag (2017), when (1/ε²) · log(1/δ) ≥ *D* ≥ log(1/δ), the optimal bit-width satisfies

```
B = Θ( log( (1/D) · log(1/δ) / ε² ) ).
```

RaBitQ is proved to match this optimal trade-off; see Theorem 3.2 of Gao et al. (2025b). It is worth emphasizing that in the optimal case, the bit-width *B* grows with 1/δ at the rate of **log log(1/δ)**.

## TurboQuant: Variance Guarantee Only

In contrast, TurboQuant provides only a guarantee on the variance of the inner-product estimation error; see Theorem 2 of Zandieh & Mirrokni (2026). A variance guarantee can be converted into a tail bound via Chebyshev's inequality, which we restate as follows.

> **Lemma (Chebyshev's inequality, Durrett 2010).** Let *X* be a random variable with mean 0 and variance σ². Then for any *t* > 0,
>
> ```
> P( |X| ≥ t ) ≤ σ² / t².
> ```

However, TurboQuant's theoretical guarantee implies only a suboptimal trade-off between the bit-width *B* and the failure probability δ. More precisely, TurboQuant bounds only the variance of the estimator and such a guarantee does not directly yield a sub-Gaussian tail bound. If one applies Chebyshev's inequality to this variance bound, the resulting dependence requires *B* to scale as **log(1/δ)**. This is exponentially worse than the log log(1/δ) dependence attained by RaBitQ, which Alon and Klartag (2017) showed to be optimal.

## Summary

| | Growth of bit-width *B* with 1/δ | Optimal? |
|---|---|---|
| RaBitQ | log log(1/δ) | **Yes** (matches Alon-Klartag lower bound) |
| TurboQuant (via Chebyshev) | log(1/δ) | **No** (exponentially worse) |

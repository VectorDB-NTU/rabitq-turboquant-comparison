# Bugs in the Original TurboQuant LLM Code

This document details four critical bugs in the KV-cache quantization code from the TurboQuant OpenReview supplementary material. These bugs prevent the original code from running correctly; we fixed them in our unified evaluation framework.

Original code source: [first commit](https://github.com/VectorDB-NTU/RabitQ_TurboQuant_KV_Comparison/commit/99dc551470923cfaaaba284cec432071cee93b19) of the OpenReview supplementary material.

## Bug 1: Missing Value Quantizer

**Problem.** The value cache quantizer (`TurboSketch` for values) is never constructed. Only key-side sketches (`qjl_outlier`, `qjl_residual`) are created — the value side has no corresponding sketch instance, so value quantization cannot run at all during prefill.

**Impact.** The value cache stays unquantized, halving the memory savings of quantization.

**Fix.** Construct an independent `TurboSketch` instance for the value side in `kvcache_quant/kv_quantizer.py`.

## Bug 2: Decode-phase Sketch Never Updates

**Problem.** Both `TurboKeyQuantizer.update_sketch()` and `TurboValueQuantizer.update_sketch()` return unconditionally before reaching the quantization logic. During decode, new tokens are appended to the unquantized buffer but are never flushed into the quantized sketch.

**Impact.** The buffer grows indefinitely, defeating the purpose of quantization. Memory usage keeps growing as generation length increases.

**Fix.** Remove the early return so that `update_sketch()` correctly executes the quantization logic and merges the buffered tokens into the sketch.

## Bug 3: Value Reconstruction Uses Wrong Operation

**Problem.** `TurboValueQuantizer.attention_score()` reuses `TurboSketch.calc_score()`, which computes `query @ quantized_keys^T` (an inner-product score). For value reconstruction the correct operation is `attention_weights @ quantized_values` — the two have different semantics and dimensions.

**Impact.** Value reconstruction is completely incorrect, severely degrading model output quality.

**Fix.** Implement the correct `attention_weights @ quantized_values` operation for value reconstruction in `kvcache_quant/kv_quantizer.py`.

## Bug 4: Outlier Separation Not Applied During Decode Updates

**Problem.** The key update path during decode does not split new keys into outlier/residual channels before quantizing, unlike the prefill path which does. This means the outlier-aware quantization strategy is only applied once at prefill and lost for all subsequent tokens.

**Impact.** The outlier-aware strategy is only effective during prefill; all decoded tokens lose outlier protection, degrading quantization accuracy.

**Fix.** Add the same outlier/residual separation logic in the decode update path that the prefill path uses.

## Unified Framework After the Fixes

The fixed code lives in `kv_cache/kvcache_quant/` and uses a pluggable architecture:

- `kv_quantizer.py` — algorithm-agnostic KV cache management (buffers, outlier separation, prefill/decode flow)
- `rabitq_sketch.py` — RaBitQ sketch implementation
- `turbo_sketch.py` — TurboQuant sketch implementation

Both methods share identical cache logic, buffer management, and outlier handling — the only varying factor is the underlying quantization sketch.

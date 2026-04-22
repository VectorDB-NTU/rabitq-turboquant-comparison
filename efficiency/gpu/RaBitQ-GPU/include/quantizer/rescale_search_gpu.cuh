//
// Device-side helpers for the RaBitQ rescale-factor search, plus a host entry
// point that draws the constant scaling factor used in the fast-quantize path.
//
// Extracted from quantizer_gpu_fast.cu — only the pieces needed by the
// standalone quantizer.
//

#ifndef RABITQ_GPU_RESCALE_SEARCH_CUH
#define RABITQ_GPU_RESCALE_SEARCH_CUH

#include <cstddef>

/// Warp-cooperative rescale-factor search. Declared here so the standalone
/// quantizer kernel can call it via `extern __device__` linkage.
///
/// s_xp_norm  : shared-memory array of |x|/||x|| values, length D.
/// D          : working dimension (padded).
/// EX_BITS    : number of extended bits (total_bits - 1).
/// reuse_space: scratch array in shared memory, at least BlockSize + 2*nWarps
///              floats (BlockSize/32 == nWarps).
/// BlockSize  : blockDim.x of the calling kernel.
///
/// Returns the selected rescale factor `t` (broadcast via the return value
/// from thread 0's path; all threads read it from shared memory through the
/// kernel's own __syncthreads pattern).
__device__ float compute_best_rescale_parallel(
        float* s_xp_norm,
        int D,
        int EX_BITS,
        float* reuse_space,
        int BlockSize);

/// Host: estimate the constant scaling factor used by the fast-quantize path.
/// Averages `kConstNum` rescale factors computed on random Gaussian vectors.
float rabitq_get_const_scaling_factor_gpu(size_t dim, size_t ex_bits);

#endif // RABITQ_GPU_RESCALE_SEARCH_CUH

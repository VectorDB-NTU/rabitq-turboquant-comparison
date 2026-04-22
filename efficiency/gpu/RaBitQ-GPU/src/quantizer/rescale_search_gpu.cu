//
// Implementation of the rescale-factor search helpers extracted from
// IVF-RaBitQ-GPU-main/inc/gpu_index/quantizer_gpu_fast.cu.
//

#include "quantizer/rescale_search_gpu.cuh"
#include "quantizer/tight_start_constants.cuh"
#include "utils/utils_cuda.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <ctime>

// ---------------------------------------------------------------------------
// Block / warp reductions
// ---------------------------------------------------------------------------

static __inline__ __device__ float warpReduceSumdup(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static __inline__ __device__ float blockReduceSumdup(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warpReduceSumdup(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    float out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.f;
    if (wid == 0) out = warpReduceSumdup(out);
    return out;
}

// ---------------------------------------------------------------------------
// Warp-cooperative single-sample evaluator for the rescale search.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float evaluate_rescale_sample_warp(
        const float* __restrict__ s_xp_norm, int D, int EX_BITS, float t, int lane_id)
{
    constexpr float kEps = 1e-5f;
    int max_code = (1 << EX_BITS) - 1;
    float numerator = 0.0f;
    float sqr_denom = (lane_id == 0) ? static_cast<float>(D) * 0.25f : 0.0f;

    for (int j = lane_id; j < D; j += 32) {
        float val = fabsf(s_xp_norm[j]);
        int quantized = min(__float2int_rd(t * val + kEps), max_code);
        numerator += (quantized + 0.5f) * val;
        sqr_denom += quantized * quantized + quantized;
    }

    numerator = warpReduceSumdup(numerator);
    sqr_denom = warpReduceSumdup(sqr_denom);
    return numerator / sqrtf(sqr_denom);
}

// ---------------------------------------------------------------------------
// Warp-cooperative rescale search. See header for contract.
// ---------------------------------------------------------------------------
__device__ float compute_best_rescale_parallel(
        float* s_xp_norm,
        int D,
        int EX_BITS,
        float* reuse_space,
        int BlockSize)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int nWarps = BlockSize / 32;

    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;
    constexpr int COARSE_SAMPLES = 64;
    constexpr int FINE_SAMPLES = 64;

    // block-wide max of |s_xp_norm|
    float local_max = 0.0f;
    for (int i = tid; i < D; i += BlockSize) {
        local_max = fmaxf(local_max, fabsf(s_xp_norm[i]));
    }

    float* s_reduce = reuse_space;
    s_reduce[tid] = local_max;
    __syncthreads();
    for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + stride]);
        __syncthreads();
    }
    __shared__ float max_o_shared;
    if (tid == 0) max_o_shared = s_reduce[0];
    __syncthreads();
    float max_o = max_o_shared;

    if (max_o < kEps) return 1.0f;

    float t_end = static_cast<float>((1 << EX_BITS) - 1 + kNEnum) / max_o;
    float t_start = t_end * d_kTightStart_opt[EX_BITS];

    float* s_warp_ip = reuse_space + BlockSize;
    float* s_warp_t  = s_warp_ip + nWarps;

    // coarse grid search
    float best_coarse_ip = 0.0f;
    float best_coarse_t  = t_start;
    for (int base = 0; base < COARSE_SAMPLES; base += nWarps) {
        int si = base + warp_id;
        float tc = (si < COARSE_SAMPLES)
            ? t_start + (t_end - t_start) * si / (COARSE_SAMPLES - 1)
            : t_start;
        float ip = (si < COARSE_SAMPLES)
            ? evaluate_rescale_sample_warp(s_xp_norm, D, EX_BITS, tc, lane_id)
            : 0.0f;
        if (lane_id == 0 && ip > best_coarse_ip) {
            best_coarse_ip = ip;
            best_coarse_t  = tc;
        }
    }

    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_coarse_ip;
        s_warp_t[warp_id]  = best_coarse_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ip = (lane_id < nWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tc = (lane_id < nWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            float oi = __shfl_down_sync(0xffffffff, ip, s);
            float ot = __shfl_down_sync(0xffffffff, tc, s);
            if (oi > ip) { ip = oi; tc = ot; }
        }
        if (lane_id == 0) {
            s_warp_ip[0] = ip;
            s_warp_t[0]  = tc;
        }
    }
    __syncthreads();

    float center_t = s_warp_t[0];
    float range = (t_end - t_start) / COARSE_SAMPLES;
    float fine_start = fmaxf(t_start, center_t - range);
    float fine_end   = fminf(t_end,   center_t + range);

    // fine grid search
    float best_fine_ip = 0.0f;
    float best_fine_t  = center_t;
    for (int base = 0; base < FINE_SAMPLES; base += nWarps) {
        int si = base + warp_id;
        float tf = (si < FINE_SAMPLES)
            ? fine_start + (fine_end - fine_start) * si / (FINE_SAMPLES - 1)
            : center_t;
        float ip = (si < FINE_SAMPLES)
            ? evaluate_rescale_sample_warp(s_xp_norm, D, EX_BITS, tf, lane_id)
            : 0.0f;
        if (lane_id == 0 && ip > best_fine_ip) {
            best_fine_ip = ip;
            best_fine_t  = tf;
        }
    }

    if (lane_id == 0) {
        s_warp_ip[warp_id] = best_fine_ip;
        s_warp_t[warp_id]  = best_fine_t;
    }
    __syncthreads();

    if (warp_id == 0) {
        float ip = (lane_id < nWarps) ? s_warp_ip[lane_id] : -1.0f;
        float tf = (lane_id < nWarps) ? s_warp_t[lane_id]  : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            float oi = __shfl_down_sync(0xffffffff, ip, s);
            float ot = __shfl_down_sync(0xffffffff, tf, s);
            if (oi > ip) { ip = oi; tf = ot; }
        }
        if (lane_id == 0) s_warp_t[0] = tf;
    }
    __syncthreads();

    return s_warp_t[0];
}

// ---------------------------------------------------------------------------
// Fully-fused kernel: generate a random Gaussian row, normalize, search for
// the optimal rescale factor. One block = one sample row.
// ---------------------------------------------------------------------------
__global__ void rabitq_rescale_sample_kernel(
        float* __restrict__ output_factors,
        int rows,
        int cols,
        int ex_bits,
        unsigned long long seed)
{
    const int row_id = blockIdx.x;
    if (row_id >= rows) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float shared_mem[];
    float* row_data    = shared_mem;
    float* reuse_space = &row_data[cols];

    curandState rng_state;
    curand_init(seed, row_id * block_size + tid, 0, &rng_state);

    for (int i = tid; i < cols; i += block_size) {
        row_data[i] = curand_normal(&rng_state);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = row_data[i];
        local_sum += val * val;
    }

    float norm_squared = blockReduceSumdup(local_sum);

    __shared__ float inv_norm;
    if (tid == 0) inv_norm = rsqrtf(norm_squared);
    __syncthreads();

    for (int i = tid; i < cols; i += block_size) {
        row_data[i] = fabsf(row_data[i] * inv_norm);
    }
    __syncthreads();

    float rescale_factor = compute_best_rescale_parallel(
        row_data, cols, ex_bits, reuse_space, block_size);

    if (tid == 0) output_factors[row_id] = rescale_factor;
}

// ---------------------------------------------------------------------------
// Host entry point — average the per-row factors on device.
// ---------------------------------------------------------------------------
float rabitq_get_const_scaling_factor_gpu(size_t dim, size_t ex_bits) {
    constexpr long kConstNum = 100;

    float* d_factors;
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_factors, kConstNum * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    int block_size = 256;
    if (dim <= 512) block_size = 128;
    if (dim >= 1536) block_size = 512;

    size_t shared_mem_size = (dim + 3 * block_size) * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        block_size = 128;
        shared_mem_size = (dim + 3 * block_size) * sizeof(float);
    }

    unsigned long long seed = static_cast<unsigned long long>(time(nullptr));

    rabitq_rescale_sample_kernel<<<kConstNum, block_size, shared_mem_size>>>(
        d_factors, kConstNum, static_cast<int>(dim), static_cast<int>(ex_bits), seed);
    CUDA_CHECK(cudaGetLastError());

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_factors, d_sum, kConstNum);

    void* d_temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_factors, d_sum, kConstNum);
    CUDA_CHECK(cudaGetLastError());

    float sum;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_factors));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_temp_storage));

    return sum / kConstNum;
}

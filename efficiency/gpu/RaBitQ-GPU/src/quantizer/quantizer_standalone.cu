//
// Fused RaBitQ scalar quantization kernels operating on pre-computed residuals.
//
// Extracted from IVF-RaBitQ-GPU-main/inc/gpu_index/quantizer_standalone.cu —
// only the fused warp-cooperative path (sa_quantize_fused_kernel +
// sa_compute_delta_vl_kernel) is kept, plus the free-function entry points
// referenced by the benchmark.
//

#include "quantizer/quantizer_standalone.cuh"
#include "quantizer/rescale_search_gpu.cuh"
#include "quantizer/tight_start_constants.cuh"
#include "utils/utils_cuda.cuh"

#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Warp-level reductions local to this TU.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __device__ __forceinline__ float evaluate_rescale_sample(
        const float* __restrict__ s_abs_norm, int D, int ex_bits, float t, int lane_id)
{
    constexpr float kEps = 1e-5f;
    int max_code = (1 << ex_bits) - 1;
    float numerator = 0.0f;
    float sqr_denom = (lane_id == 0) ? static_cast<float>(D) * 0.25f : 0.0f;

    for (int j = lane_id; j < D; j += 32) {
        float val = s_abs_norm[j];
        int quantized = min(__float2int_rd(t * val + kEps), max_code);
        numerator += (quantized + 0.5f) * val;
        sqr_denom += quantized * quantized + quantized;
    }

    numerator = warp_reduce_sum(numerator);
    sqr_denom = warp_reduce_sum(sqr_denom);

    return numerator / sqrtf(sqr_denom);
}

// ---------------------------------------------------------------------------
// Fused rescale search + quantize kernel (used in both fast / non-fast paths).
// ---------------------------------------------------------------------------
template<typename CodeT, int kBlockSize = 256>
__global__ void sa_quantize_fused_kernel(
    const float* __restrict__ d_residual,
    CodeT* __restrict__ d_total_code,
    int N, int padded_dim, int ex_bits,
    float const_scaling_factor,
    bool use_fast)
{
    constexpr int kNWarps = kBlockSize / 32;
    constexpr float kEps = 1e-5f;
    constexpr int kNEnum = 10;
    constexpr int COARSE_SAMPLES = 64;
    constexpr int FINE_SAMPLES = 64;

    extern __shared__ char smem[];
    float* s_reduce   = reinterpret_cast<float*>(smem);
    float* s_abs_norm = s_reduce + kBlockSize;
    float* s_warp_ip  = s_abs_norm + padded_dim;
    float* s_warp_t   = s_warp_ip + kNWarps;

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    // norm (block reduction)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize)
        local_sum += res[i] * res[i];
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(s_reduce[0] + 1e-30f);

    float t;

    if (use_fast || ex_bits == 0) {
        t = use_fast ? const_scaling_factor : 1.0f;
    } else {
        // cache abs(res)*inv_norm in smem + find block max
        float local_max = 0.0f;
        for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
            float val = fabsf(res[i]) * inv_norm;
            s_abs_norm[i] = val;
            local_max = fmaxf(local_max, val);
        }
        s_reduce[threadIdx.x] = local_max;
        __syncthreads();
        for (int s = kBlockSize / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
            __syncthreads();
        }
        float max_o = s_reduce[0];

        if (max_o < kEps) { t = 1.0f; }
        else {
            float t_end = static_cast<float>((1 << ex_bits) - 1 + kNEnum) / max_o;
            float t_start = t_end * d_kTightStart_opt[ex_bits];

            // coarse grid
            float best_coarse_ip = 0.0f, best_coarse_t = t_start;
            for (int base = 0; base < COARSE_SAMPLES; base += kNWarps) {
                int si = base + warp_id;
                float tc = (si < COARSE_SAMPLES)
                    ? t_start + (t_end - t_start) * si / (COARSE_SAMPLES - 1) : t_start;
                float ip = (si < COARSE_SAMPLES)
                    ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tc, lane_id) : 0.0f;
                if (lane_id == 0 && ip > best_coarse_ip) { best_coarse_ip = ip; best_coarse_t = tc; }
            }

            if (lane_id == 0) { s_warp_ip[warp_id] = best_coarse_ip; s_warp_t[warp_id] = best_coarse_t; }
            __syncthreads();
            if (warp_id == 0) {
                float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
                float tc = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
                for (int s = kNWarps / 2; s > 0; s >>= 1) {
                    float oi = __shfl_down_sync(0xffffffff, ip, s);
                    float ot = __shfl_down_sync(0xffffffff, tc, s);
                    if (oi > ip) { ip = oi; tc = ot; }
                }
                if (lane_id == 0) { s_warp_ip[0] = ip; s_warp_t[0] = tc; }
            }
            __syncthreads();

            float center_t = s_warp_t[0];
            float range = (t_end - t_start) / COARSE_SAMPLES;
            float fine_start = fmaxf(t_start, center_t - range);
            float fine_end   = fminf(t_end,   center_t + range);

            // fine grid
            float best_fine_ip = 0.0f, best_fine_t = center_t;
            for (int base = 0; base < FINE_SAMPLES; base += kNWarps) {
                int si = base + warp_id;
                float tf = (si < FINE_SAMPLES)
                    ? fine_start + (fine_end - fine_start) * si / (FINE_SAMPLES - 1) : center_t;
                float ip = (si < FINE_SAMPLES)
                    ? evaluate_rescale_sample(s_abs_norm, padded_dim, ex_bits, tf, lane_id) : 0.0f;
                if (lane_id == 0 && ip > best_fine_ip) { best_fine_ip = ip; best_fine_t = tf; }
            }

            if (lane_id == 0) { s_warp_ip[warp_id] = best_fine_ip; s_warp_t[warp_id] = best_fine_t; }
            __syncthreads();
            if (warp_id == 0) {
                float ip = (lane_id < kNWarps) ? s_warp_ip[lane_id] : -1.0f;
                float tf = (lane_id < kNWarps) ? s_warp_t[lane_id]  : 0.0f;
                for (int s = kNWarps / 2; s > 0; s >>= 1) {
                    float oi = __shfl_down_sync(0xffffffff, ip, s);
                    float ot = __shfl_down_sync(0xffffffff, tf, s);
                    if (oi > ip) { ip = oi; tf = ot; }
                }
                if (lane_id == 0) s_warp_t[0] = tf;
            }
            __syncthreads();
            t = s_warp_t[0];
        }
    }

    // quantize (abs + branch)
    int mask = (1 << ex_bits) - 1;
    int offset = 1 << ex_bits;
    for (int i = threadIdx.x; i < padded_dim; i += kBlockSize) {
        float r = res[i];
        float abs_val = fabsf(r) * inv_norm;
        int k = __float2int_rd(t * abs_val + kEps);
        if (k > mask) k = mask;
        int total = (r >= 0.0f) ? (k + offset) : (mask - k);
        code[i] = static_cast<CodeT>(total);
    }
}

// ---------------------------------------------------------------------------
// delta / vl kernel.
// ---------------------------------------------------------------------------
template<typename CodeT>
__global__ void sa_compute_delta_vl_kernel(
    const float* __restrict__ d_residual,
    const CodeT* __restrict__ d_total_code,
    float* __restrict__ d_delta,
    float* __restrict__ d_vl,
    int N, int padded_dim, int ex_bits,
    int delta_mode)
{
    extern __shared__ char smem[];
    float* s_buf = reinterpret_cast<float*>(smem);

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res = d_residual + (size_t)vec_id * padded_dim;
    const CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float cb = -((float)(1 << ex_bits) - 0.5f);

    float local_res_sq = 0.0f, local_ucb_sq = 0.0f, local_dot = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float u = (float)code[i] + cb;
        local_res_sq += r * r;
        local_ucb_sq += u * u;
        local_dot    += r * u;
    }

    float* s_res = s_buf;
    float* s_ucb = s_buf + blockDim.x;
    float* s_dot = s_buf + 2 * blockDim.x;
    s_res[threadIdx.x] = local_res_sq;
    s_ucb[threadIdx.x] = local_ucb_sq;
    s_dot[threadIdx.x] = local_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            s_res[threadIdx.x] += s_res[threadIdx.x + s];
            s_ucb[threadIdx.x] += s_ucb[threadIdx.x + s];
            s_dot[threadIdx.x] += s_dot[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float norm_res = sqrtf(s_res[0]);
        float norm_ucb = sqrtf(s_ucb[0]);
        float cos_sim  = s_dot[0] / (norm_res * norm_ucb + 1e-30f);

        float ratio = norm_res / (norm_ucb + 1e-30f);
        float delta;
        if (delta_mode == 1)
            delta = ratio / (cos_sim + 1e-30f);
        else if (delta_mode == 2)
            delta = ratio;
        else
            delta = ratio * cos_sim;
        d_delta[vec_id] = delta;
        d_vl[vec_id]    = delta * cb;
    }
}

// ---------------------------------------------------------------------------
// Launcher + free-function entry points.
// ---------------------------------------------------------------------------
template<typename CodeT>
static void launch_quantize_fused(
    const float* d_residual, size_t N, uint32_t padded_dim, size_t ex_bits,
    float const_scaling_factor, bool use_fast, int delta_mode,
    CodeT* d_total_code, float* d_delta, float* d_vl)
{
    constexpr int block = 256;
    int iN = static_cast<int>(N);
    int iD = static_cast<int>(padded_dim);
    int iB = static_cast<int>(ex_bits);
    constexpr int nwarps = block / 32;

    size_t q_smem = (block + padded_dim + 2 * nwarps) * sizeof(float);
    sa_quantize_fused_kernel<CodeT, block><<<iN, block, q_smem>>>(
        d_residual, d_total_code, iN, iD, iB, const_scaling_factor, use_fast);
    CUDA_CHECK(cudaGetLastError());

    size_t f_smem = 3 * block * sizeof(float);
    sa_compute_delta_vl_kernel<CodeT><<<iN, block, f_smem>>>(
        d_residual, d_total_code, d_delta, d_vl, iN, iD, iB, delta_mode);
    CUDA_CHECK(cudaGetLastError());
}

void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_delta, float* d_vl, int delta_mode)
{
    launch_quantize_fused(d_residuals, N, static_cast<uint32_t>(padded_dim), ex_bits,
                          const_scaling_factor, use_fast, delta_mode,
                          d_total_code, d_delta, d_vl);
}

void standalone_quantize_fused_on_residuals(
    const float* d_residuals, size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_delta, float* d_vl, int delta_mode)
{
    launch_quantize_fused(d_residuals, N, static_cast<uint32_t>(padded_dim), ex_bits,
                          const_scaling_factor, use_fast, delta_mode,
                          d_total_code, d_delta, d_vl);
}

// ---------------------------------------------------------------------------
// Full-factor kernel — computes (f_add, f_rescale, f_error) per vector from
// codes + residual + (rotated) centroid.
//
// Mirrors IVF-RaBitQ-GPU-main/inc/gpu_index/quantizer_standalone.cu's
// sa_compute_full_factors_kernel.
// ---------------------------------------------------------------------------
template<typename CodeT>
__global__ void sa_compute_full_factors_kernel(
    const float* __restrict__ d_residual,
    const float* __restrict__ d_centroid,
    const CodeT* __restrict__ d_total_code,
    float* __restrict__ d_factors,
    int N, int padded_dim, int ex_bits)
{
    extern __shared__ char smem[];
    float* s_buf = reinterpret_cast<float*>(smem);  // 4 × blockDim.x

    int vec_id = blockIdx.x;
    if (vec_id >= N) return;

    const float* res  = d_residual   + (size_t)vec_id * padded_dim;
    const CodeT* code = d_total_code + (size_t)vec_id * padded_dim;

    float cb = -((float)(1 << ex_bits) - 0.5f);
    constexpr float kEpsilon = 1.9f;

    float local_l2 = 0.0f, local_ip_res = 0.0f, local_ip_cent = 0.0f, local_xu_sq = 0.0f;
    for (int i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        float r = res[i];
        float c = d_centroid[i];
        float xu_cb = (float)code[i] + cb;

        local_l2      += r * r;
        local_ip_res  += r * xu_cb;
        local_ip_cent += c * xu_cb;
        local_xu_sq   += xu_cb * xu_cb;
    }

    float* s0 = s_buf;
    float* s1 = s_buf +     blockDim.x;
    float* s2 = s_buf + 2 * blockDim.x;
    float* s3 = s_buf + 3 * blockDim.x;
    s0[threadIdx.x] = local_l2;
    s1[threadIdx.x] = local_ip_res;
    s2[threadIdx.x] = local_ip_cent;
    s3[threadIdx.x] = local_xu_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            s0[threadIdx.x] += s0[threadIdx.x + s];
            s1[threadIdx.x] += s1[threadIdx.x + s];
            s2[threadIdx.x] += s2[threadIdx.x + s];
            s3[threadIdx.x] += s3[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float l2_sq        = s0[0];
        float ip_resi_xucb = s1[0];
        float ip_cent_xucb = s2[0];
        float xu_sq        = s3[0];

        float l2_norm = sqrtf(l2_sq);
        float denom   = ip_resi_xucb + 1e-30f;

        float f_add     = l2_sq + 2.0f * l2_sq * (ip_cent_xucb / denom);
        float f_rescale = -2.0f * l2_sq / denom;

        float ratio = (l2_sq * xu_sq) / (denom * denom);
        float inner = fmaxf(0.0f, (ratio - 1.0f) / ((float)padded_dim - 1.0f));
        float f_error = 2.0f * l2_norm * kEpsilon * sqrtf(inner);

        d_factors[vec_id * 3 + 0] = f_add;
        d_factors[vec_id * 3 + 1] = f_rescale;
        d_factors[vec_id * 3 + 2] = f_error;
    }
}

template<typename CodeT>
static void launch_quantize_full(
    const float* d_residual, const float* d_centroid,
    size_t N, uint32_t padded_dim, size_t ex_bits,
    float const_scaling_factor, bool use_fast,
    CodeT* d_total_code, float* d_factors)
{
    constexpr int block = 256;
    int iN = static_cast<int>(N);
    int iD = static_cast<int>(padded_dim);
    int iB = static_cast<int>(ex_bits);
    constexpr int nwarps = block / 32;

    size_t q_smem = (block + padded_dim + 2 * nwarps) * sizeof(float);
    sa_quantize_fused_kernel<CodeT, block><<<iN, block, q_smem>>>(
        d_residual, d_total_code, iN, iD, iB, const_scaling_factor, use_fast);
    CUDA_CHECK(cudaGetLastError());

    size_t f_smem = 4 * block * sizeof(float);
    sa_compute_full_factors_kernel<CodeT><<<iN, block, f_smem>>>(
        d_residual, d_centroid, d_total_code, d_factors, iN, iD, iB);
    CUDA_CHECK(cudaGetLastError());
}

void standalone_quantize_full_on_residuals(
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint16_t* d_total_code, float* d_factors)
{
    launch_quantize_full(d_residuals, d_centroid, N,
                         static_cast<uint32_t>(padded_dim), ex_bits,
                         const_scaling_factor, use_fast,
                         d_total_code, d_factors);
}

void standalone_quantize_full_on_residuals(
    const float* d_residuals, const float* d_centroid,
    size_t N, size_t padded_dim,
    size_t ex_bits, float const_scaling_factor, bool use_fast,
    uint8_t* d_total_code, float* d_factors)
{
    launch_quantize_full(d_residuals, d_centroid, N,
                         static_cast<uint32_t>(padded_dim), ex_bits,
                         const_scaling_factor, use_fast,
                         d_total_code, d_factors);
}

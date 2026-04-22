//
// test_quantizer_standalone — benchmark standalone GPU RaBitQ quantization on
// real data. Reports timing (with / without rotation) and reconstruction error
// (L2 + diagonal IP, optional full pairwise IP against a test set).
//
// Usage:
//   ./test_quantizer_standalone <base.fvecs> <total_bits> [fast] [num_vectors]
//                               [test.fvecs] [delta_mode]
//
//   total_bits : 1..9          (ex_bits = total_bits - 1)
//   fast       : true|false    use precomputed const-scaling factor (default: true)
//   num_vectors: optional subset size (0 or omit = use all rows)
//   test.fvecs : optional — enables full-pairwise IP error vs test set
//   delta_mode : 0=RECONSTRUCTION (default), 1=UNBIASED, 2=PLAIN
//

#include "quantizer/quantizer_standalone.cuh"
#include "quantizer/rescale_search_gpu.cuh"
#include "quantizer/rotator_gpu.cuh"
#include "utils/IO.hpp"
#include "utils/utils_cuda.cuh"
#include "defines.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static inline uint32_t rd_up(uint32_t x, uint32_t n) {
    return ((x + n - 1u) / n) * n;
}

// ---------------------------------------------------------------------------
// Error computation — CPU streaming (no N×N matrix stored)
// ---------------------------------------------------------------------------
struct ErrorStats {
    double l2_mean = 0, l2_max = 0, l2_std = 0;
    double ip_mean = 0, ip_max = 0, ip_std = 0;
};

static void print_stats(const char* name, const ErrorStats& s) {
    std::cout << std::left << std::setw(18) << name
              << "L2: mean=" << std::scientific << std::setprecision(4) << s.l2_mean
              << " max=" << s.l2_max << " std=" << s.l2_std
              << "  IP: mean=" << s.ip_mean << " max=" << s.ip_max
              << " std=" << s.ip_std << std::endl;
}

static ErrorStats compute_errors_cpu(
    const float* orig, const uint16_t* codes,
    const float* delta, const float* vl,
    size_t N, size_t D)
{
    std::vector<double> l2_errs(N);
    std::vector<double> ip_errs(N);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        double sum_diff_sq = 0, dot_orig_orig = 0, dot_orig_recon = 0;
        for (size_t d = 0; d < D; ++d) {
            float o = orig[i * D + d];
            float r = static_cast<float>(codes[i * D + d]) * delta[i] + vl[i];
            double diff = o - r;
            sum_diff_sq += diff * diff;
            dot_orig_orig += static_cast<double>(o) * o;
            dot_orig_recon += static_cast<double>(o) * r;
        }
        l2_errs[i] = std::sqrt(sum_diff_sq);
        ip_errs[i] = dot_orig_orig - dot_orig_recon;
    }

    ErrorStats s{};

    double l2s = 0, l2sq = 0;
    size_t l2_valid = 0;
    for (size_t i = 0; i < N; ++i) {
        double e = l2_errs[i];
        if (!std::isfinite(e)) continue;
        l2s += e; l2sq += e * e; s.l2_max = std::max(s.l2_max, e); ++l2_valid;
    }
    s.l2_mean = l2_valid > 0 ? l2s / static_cast<double>(l2_valid) : 0;
    s.l2_std  = l2_valid > 0 ? std::sqrt(std::max(0.0, l2sq / l2_valid - s.l2_mean * s.l2_mean)) : 0;

    double ips = 0, ipsq = 0, ip_max_abs = 0;
    size_t ip_valid = 0;
    for (size_t i = 0; i < N; ++i) {
        double e = ip_errs[i];
        if (!std::isfinite(e)) continue;
        ips += e; ipsq += e * e;
        double a = std::abs(e);
        if (a > ip_max_abs) ip_max_abs = a;
        ++ip_valid;
    }
    s.ip_mean = ip_valid > 0 ? ips / static_cast<double>(ip_valid) : 0;
    s.ip_std  = ip_valid > 0 ? std::sqrt(std::max(0.0, ipsq / ip_valid - s.ip_mean * s.ip_mean)) : 0;
    s.ip_max  = ip_max_abs;
    return s;
}

static void compute_ip_with_test(
    const float* test_data, size_t N_test,
    const float* orig, const uint16_t* codes,
    const float* delta, const float* vl,
    size_t N, size_t D,
    size_t block_size,
    double& ip_mean, double& ip_std, double& ip_max)
{
    std::vector<float> recon(N * D);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            recon[i * D + d] = static_cast<float>(codes[i * D + d]) * delta[i] + vl[i];

    double total = 0, total_sq = 0, max_abs = 0;
    size_t count = 0;

    for (size_t start = 0; start < N_test; start += block_size) {
        size_t end = std::min(start + block_size, N_test);
        size_t bs = end - start;

        #pragma omp parallel for schedule(static) reduction(+:total,total_sq,count)
        for (size_t j = 0; j < bs; ++j) {
            const float* t = test_data + (start + j) * D;
            double local_max = 0;
            for (size_t i = 0; i < N; ++i) {
                double orig_ip = 0, quant_ip = 0;
                for (size_t d = 0; d < D; ++d) {
                    double td = t[d];
                    orig_ip += td * orig[i * D + d];
                    quant_ip += td * recon[i * D + d];
                }
                double err = orig_ip - quant_ip;
                if (std::isfinite(err)) {
                    total += err;
                    total_sq += err * err;
                    double a = std::abs(err);
                    if (a > local_max) local_max = a;
                    ++count;
                }
            }
            #pragma omp critical
            { if (local_max > max_abs) max_abs = local_max; }
        }
    }

    ip_mean = count > 0 ? total / static_cast<double>(count) : 0;
    ip_std  = count > 0 ? std::sqrt(std::max(0.0, total_sq / count - ip_mean * ip_mean)) : 0;
    ip_max  = max_abs;
}

// ---------------------------------------------------------------------------
// Timing helper — wall-clock ms over `iters` calls.
// ---------------------------------------------------------------------------
template<typename Fn>
static float benchmark_gpu(Fn fn, int iters = 1) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return static_cast<float>(ms / iters);
}

// ---------------------------------------------------------------------------
// Pad kernel + pad-then-rotate helper.
// ---------------------------------------------------------------------------
__global__ void test_sa_pad_kernel(const float* __restrict__ d_src,
                                    float* __restrict__ d_dst,
                                    int N, int DIM, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    int row = idx / D;
    int col = idx % D;
    d_dst[idx] = (col < DIM) ? d_src[row * DIM + col] : 0.0f;
}

static void pad_and_rotate(const float* d_data, size_t N, size_t DIM, size_t D,
                           const RotatorGPU& rotator, float* d_residuals) {
    if (DIM == D) {
        rotator.rotate(d_data, d_residuals, N);
    } else {
        float* d_padded = nullptr;
        CUDA_CHECK(cudaMalloc(&d_padded, N * D * sizeof(float)));

        int total = static_cast<int>(N * D);
        int block = 256;
        int grid = (total + block - 1) / block;
        test_sa_pad_kernel<<<grid, block>>>(d_data, d_padded,
                                            static_cast<int>(N), static_cast<int>(DIM), static_cast<int>(D));
        CUDA_CHECK(cudaGetLastError());

        rotator.rotate(d_padded, d_residuals, N);
        CUDA_CHECK(cudaFree(d_padded));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <base.fvecs> <total_bits> [fast] [num_vectors] [test.fvecs] [delta_mode] [rotator] [factor_mode]" << std::endl;
        std::cerr << "  total_bits  : 1..9 (ex_bits = total_bits - 1)" << std::endl;
        std::cerr << "  fast        : true|false (default: true)" << std::endl;
        std::cerr << "  num_vectors : 0 or omit to use all rows" << std::endl;
        std::cerr << "  test.fvecs  : optional — enables full-pairwise IP error" << std::endl;
        std::cerr << "  delta_mode  : 0=RECONSTRUCTION (default), 1=UNBIASED, 2=PLAIN (scalar mode only)" << std::endl;
        std::cerr << "  rotator     : fhtkac (default) | matrix" << std::endl;
        std::cerr << "  factor_mode : scalar (default) | full (emits f_add, f_rescale, f_error)" << std::endl;
        return 1;
    }

    const char* DATA_PATH = argv[1];
    int total_bits = std::atoi(argv[2]);
    if (total_bits < 1 || total_bits > 9) {
        std::cerr << "Error: total_bits must be 1..9" << std::endl;
        return 1;
    }

    bool fast = true;
    if (argc > 3 && (std::string(argv[3]) == "false" || std::string(argv[3]) == "0")) fast = false;

    size_t subset = 0;
    if (argc > 4) subset = static_cast<size_t>(std::atol(argv[4]));

    const char* TEST_PATH = nullptr;
    if (argc > 5 && std::string(argv[5]) != "-" && std::string(argv[5]) != "") TEST_PATH = argv[5];

    int delta_mode = 0;
    if (argc > 6) delta_mode = std::atoi(argv[6]);

    RotatorType rota_type = RotatorType::FhtKac;
    if (argc > 7 && std::string(argv[7]) == "matrix") rota_type = RotatorType::Matrix;

    bool full_factors = false;
    if (argc > 8 && std::string(argv[8]) == "full") full_factors = true;

    // Load data
    FloatRowMat data;
    load_vecs<float, FloatRowMat>(DATA_PATH, data);

    size_t N_total = data.rows();
    size_t DIM = data.cols();
    size_t N = (subset > 0 && subset < N_total) ? subset : N_total;
    uint32_t D = rd_up(static_cast<uint32_t>(DIM), 64);
    size_t ex_bits = total_bits - 1;

    std::cout << "Dataset:     " << DATA_PATH << std::endl;
    std::cout << "N:           " << N << " (total: " << N_total << ")" << std::endl;
    std::cout << "DIM:         " << DIM << ", padded: " << D << std::endl;
    std::cout << "Bits:        " << total_bits << " (ex_bits: " << ex_bits << ")" << std::endl;
    std::cout << "Fast:        " << (fast ? "Yes" : "No") << std::endl;
    std::cout << "Rotator:     " << (rota_type == RotatorType::Matrix ? "Matrix" : "FhtKac") << std::endl;
    std::cout << "FactorMode:  " << (full_factors ? "full (f_add, f_rescale, f_error)" : "scalar (delta, vl)") << std::endl;
    const char* mode_names[] = {"RECONSTRUCTION", "UNBIASED", "PLAIN"};
    if (!full_factors)
        std::cout << "DeltaMode:   " << mode_names[delta_mode] << std::endl;

    // Load test set if provided
    FloatRowMat test_data_mat;
    std::vector<float> h_test_padded;
    size_t N_test = 0;
    if (TEST_PATH) {
        load_vecs<float, FloatRowMat>(TEST_PATH, test_data_mat);
        N_test = test_data_mat.rows();
        size_t test_dim = test_data_mat.cols();
        if (test_dim != DIM) {
            std::cerr << "Test dim " << test_dim << " != data dim " << DIM << std::endl;
            return 1;
        }
        std::cout << "Test set:  " << TEST_PATH << " (" << N_test << " vectors)" << std::endl;
    }

    // Copy base data to GPU (unpadded)
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), N * DIM * sizeof(float), cudaMemcpyHostToDevice));

    RotatorGPU rotator(static_cast<uint32_t>(DIM), rota_type);

    // Warm-up to absorb CUDA context init before any timed section.
    {
        float* d_warm_in = nullptr;
        float* d_warm_out = nullptr;
        uint16_t* d_warm_code = nullptr;
        float* d_warm_delta = nullptr;
        float* d_warm_vl = nullptr;
        CUDA_CHECK(cudaMalloc(&d_warm_in,    D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_warm_out,   D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_warm_code,  D * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_warm_delta, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_warm_vl,    sizeof(float)));
        CUDA_CHECK(cudaMemset(d_warm_in, 0, D * sizeof(float)));

        rotator.rotate(d_warm_in, d_warm_out, 1);
        if (ex_bits > 0) (void)rabitq_get_const_scaling_factor_gpu(D, ex_bits);
        standalone_quantize_fused_on_residuals(
            d_warm_out, 1, D, ex_bits, 1.0f, fast,
            d_warm_code, d_warm_delta, d_warm_vl, delta_mode);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_warm_in));
        CUDA_CHECK(cudaFree(d_warm_out));
        CUDA_CHECK(cudaFree(d_warm_code));
        CUDA_CHECK(cudaFree(d_warm_delta));
        CUDA_CHECK(cudaFree(d_warm_vl));
    }

    // Rotate test set (for full-pairwise IP — rotation is orthogonal so IP is preserved)
    if (N_test > 0) {
        h_test_padded.assign(N_test * D, 0.0f);
        for (size_t i = 0; i < N_test; ++i)
            for (size_t d = 0; d < DIM; ++d)
                h_test_padded[i * D + d] = test_data_mat(i, d);

        float* d_test_padded = nullptr;
        float* d_test_rotated = nullptr;
        CUDA_CHECK(cudaMalloc(&d_test_padded,  N_test * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_test_rotated, N_test * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_test_padded, h_test_padded.data(),
                              N_test * D * sizeof(float), cudaMemcpyHostToDevice));
        rotator.rotate(d_test_padded, d_test_rotated, N_test);
        CUDA_CHECK(cudaMemcpy(h_test_padded.data(), d_test_rotated,
                              N_test * D * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_test_padded));
        CUDA_CHECK(cudaFree(d_test_rotated));
        std::cout << "Test vectors rotated with GPU rotator." << std::endl;
    }

    // Const scaling factor (timed)
    float const_sf = 0.0f;
    float const_sf_ms = 0.0f;
    if (fast && ex_bits > 0) {
        auto t0 = std::chrono::high_resolution_clock::now();
        const_sf = rabitq_get_const_scaling_factor_gpu(D, ex_bits);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        const_sf_ms = static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::cout << "const_scaling_factor: " << const_sf
              << " (computed in " << const_sf_ms << " ms)" << std::endl;

    // Step 1: compute rotated residuals (ground truth for error analysis)
    std::cout << "\nPreparing residuals..." << std::flush;
    float* d_residuals = nullptr;
    CUDA_CHECK(cudaMalloc(&d_residuals, N * D * sizeof(float)));
    pad_and_rotate(d_data, N, DIM, D, rotator, d_residuals);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << " done." << std::endl;

    std::vector<float> h_residuals(N * D);
    CUDA_CHECK(cudaMemcpy(h_residuals.data(), d_residuals, N * D * sizeof(float), cudaMemcpyDeviceToHost));

    // Step 2: benchmark quantization + collect outputs for error analysis.
    float ms_no_rot   = 0.0f;
    float ms_with_rot = 0.0f;

    std::vector<uint16_t> h_codes(N * D);
    std::vector<float>    h_delta, h_vl, h_factors;

    uint16_t* d_code  = nullptr;
    float*    d_delta = nullptr;
    float*    d_vl    = nullptr;
    float*    d_factors = nullptr;

    // Zero rotated centroid buffer (used by full-factor kernel).
    float* d_zero_centroid = nullptr;
    if (full_factors) {
        CUDA_CHECK(cudaMalloc(&d_zero_centroid, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_zero_centroid, 0, D * sizeof(float)));
    }

    auto quantize_once = [&](const float* d_res) {
        if (full_factors) {
            standalone_quantize_full_on_residuals(
                d_res, d_zero_centroid, N, D, ex_bits, const_sf, fast,
                d_code, d_factors);
        } else {
            standalone_quantize_fused_on_residuals(
                d_res, N, D, ex_bits, const_sf, fast,
                d_code, d_delta, d_vl, delta_mode);
        }
    };

    std::cout << "Running GPU..." << std::flush;

    // --- no rotation ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMalloc(&d_code, N * D * sizeof(uint16_t)));
        if (full_factors) {
            CUDA_CHECK(cudaMalloc(&d_factors, N * 3 * sizeof(float)));
        } else {
            CUDA_CHECK(cudaMalloc(&d_delta, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_vl,    N * sizeof(float)));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float malloc_ms = static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count());

        float kernel_ms = benchmark_gpu([&]() { quantize_once(d_residuals); });
        ms_no_rot = const_sf_ms + malloc_ms + kernel_ms;
    }

    // --- with rotation ---
    float* d_tmp_res = nullptr;
    {
        CUDA_CHECK(cudaFree(d_code)); d_code = nullptr;
        if (full_factors) {
            CUDA_CHECK(cudaFree(d_factors)); d_factors = nullptr;
        } else {
            CUDA_CHECK(cudaFree(d_delta)); d_delta = nullptr;
            CUDA_CHECK(cudaFree(d_vl));    d_vl    = nullptr;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMalloc(&d_tmp_res, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_code, N * D * sizeof(uint16_t)));
        if (full_factors) {
            CUDA_CHECK(cudaMalloc(&d_factors, N * 3 * sizeof(float)));
        } else {
            CUDA_CHECK(cudaMalloc(&d_delta, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_vl,    N * sizeof(float)));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float malloc_ms = static_cast<float>(std::chrono::duration<double, std::milli>(t1 - t0).count());

        float kernel_ms = benchmark_gpu([&]() {
            pad_and_rotate(d_data, N, DIM, D, rotator, d_tmp_res);
            quantize_once(d_tmp_res);
        });
        ms_with_rot = const_sf_ms + malloc_ms + kernel_ms;
    }
    CUDA_CHECK(cudaFree(d_tmp_res));

    // Re-run on residuals to produce final outputs for analysis
    quantize_once(d_residuals);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_codes.data(), d_code, N * D * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    if (full_factors) {
        h_factors.resize(N * 3);
        CUDA_CHECK(cudaMemcpy(h_factors.data(), d_factors, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_factors));
    } else {
        h_delta.resize(N);
        h_vl.resize(N);
        CUDA_CHECK(cudaMemcpy(h_delta.data(), d_delta, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vl.data(),    d_vl,    N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_delta));
        CUDA_CHECK(cudaFree(d_vl));
    }
    CUDA_CHECK(cudaFree(d_code));
    if (d_zero_centroid) CUDA_CHECK(cudaFree(d_zero_centroid));
    std::cout << " done." << std::endl;

    // Step 3: analysis
    std::cout << "\n=== Error Analysis ===" << std::endl;
    if (full_factors) {
        // Sanity: check all factors are finite; summarize ranges.
        size_t nan_count = 0;
        double f_add_sum = 0, f_resc_sum = 0, f_err_sum = 0;
        double f_add_min = 1e30, f_add_max = -1e30;
        double f_resc_min = 1e30, f_resc_max = -1e30;
        double f_err_min = 1e30,  f_err_max = -1e30;
        for (size_t i = 0; i < N; ++i) {
            double fa = h_factors[i * 3 + 0];
            double fr = h_factors[i * 3 + 1];
            double fe = h_factors[i * 3 + 2];
            if (!std::isfinite(fa) || !std::isfinite(fr) || !std::isfinite(fe)) { ++nan_count; continue; }
            f_add_sum += fa; f_resc_sum += fr; f_err_sum += fe;
            f_add_min = std::min(f_add_min, fa); f_add_max = std::max(f_add_max, fa);
            f_resc_min = std::min(f_resc_min, fr); f_resc_max = std::max(f_resc_max, fr);
            f_err_min = std::min(f_err_min, fe);  f_err_max  = std::max(f_err_max, fe);
        }
        size_t good = N - nan_count;
        std::cout << "(full-factor mode: " << nan_count << " non-finite / " << N << " vectors)" << std::endl;
        std::cout << std::scientific << std::setprecision(4)
                  << "f_add     mean=" << (good ? f_add_sum  / good : 0) << "  range=[" << f_add_min  << ", " << f_add_max  << "]" << std::endl
                  << "f_rescale mean=" << (good ? f_resc_sum / good : 0) << "  range=[" << f_resc_min << ", " << f_resc_max << "]" << std::endl
                  << "f_error   mean=" << (good ? f_err_sum  / good : 0) << "  range=[" << f_err_min  << ", " << f_err_max  << "]" << std::endl;
    } else {
        if (N_test > 0)
            std::cout << "(IP error: full pairwise with " << N_test << " test vectors, block=128)" << std::endl;
        else
            std::cout << "(IP error: diagonal self-similarity)" << std::endl;

        auto stats = compute_errors_cpu(h_residuals.data(), h_codes.data(),
                                        h_delta.data(), h_vl.data(), N, D);
        if (N_test > 0) {
            compute_ip_with_test(
                h_test_padded.data(), N_test,
                h_residuals.data(), h_codes.data(),
                h_delta.data(), h_vl.data(),
                N, D, 128,
                stats.ip_mean, stats.ip_std, stats.ip_max);
        }
        print_stats("GPU", stats);
    }

    // Step 4: performance summary
    std::cout << "\n=== Performance (N=" << N << ", D=" << D << ") ===" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Method"
              << std::setw(22) << "No rotation"
              << std::setw(22) << "With rotation" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    std::cout << std::left << std::setw(20) << "GPU"
              << std::fixed << std::setprecision(3)
              << std::setw(8) << ms_no_rot << " ms ("
              << std::setw(7) << static_cast<float>(N) / (ms_no_rot * 1000.0f) << " Mv/s)  "
              << std::setw(8) << ms_with_rot << " ms ("
              << std::setw(7) << static_cast<float>(N) / (ms_with_rot * 1000.0f) << " Mv/s)"
              << std::endl;

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_residuals));

    std::cout << "\nDone." << std::endl;
    return 0;
}

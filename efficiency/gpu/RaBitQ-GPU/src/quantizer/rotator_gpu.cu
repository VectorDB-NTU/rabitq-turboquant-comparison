//
// RotatorGPU — Matrix and FhtKac implementations.
// Random orthogonal matrix is generated on CPU via modified Gram-Schmidt so
// this file has no Eigen dependency.
//

#include "quantizer/rotator_gpu.cuh"
#include "quantizer/fht_cuda.cuh"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

static inline size_t rd_up(size_t dim, size_t mult) {
    return ((dim + mult - 1) / mult) * mult;
}

static inline size_t floor_log2(size_t x) {
    size_t r = 0;
    while (x >>= 1) { ++r; }
    return r;
}

// ---------------------------------------------------------------------------
// Modified Gram-Schmidt on a random Gaussian D×D matrix. Produces Q that is
// orthonormal in rows; equivalent to Q from a QR decomposition modulo sign
// conventions. Numerically stable enough for the few-thousand dimensions used
// by RaBitQ. Output is row-major.
// ---------------------------------------------------------------------------
static std::vector<float> random_orthogonal_matrix(size_t D) {
    std::vector<float> M(D * D);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < D * D; ++i) M[i] = dist(gen);

    // Modified Gram-Schmidt on rows (in double for stability).
    std::vector<double> row(D);
    for (size_t i = 0; i < D; ++i) {
        for (size_t k = 0; k < D; ++k) row[k] = M[i * D + k];

        for (size_t j = 0; j < i; ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < D; ++k) dot += row[k] * M[j * D + k];
            for (size_t k = 0; k < D; ++k) row[k] -= dot * M[j * D + k];
        }

        double norm_sq = 0.0;
        for (size_t k = 0; k < D; ++k) norm_sq += row[k] * row[k];
        double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (size_t k = 0; k < D; ++k) M[i * D + k] = static_cast<float>(row[k] * inv_norm);
    }
    return M;
}

// ---------------------------------------------------------------------------
// Matrix-rotator init
// ---------------------------------------------------------------------------
void RotatorGPU::init_matrix(uint32_t dim) {
    D = rd_up(dim, 64);

    std::vector<float> hostP = random_orthogonal_matrix(D);
    CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * D * D));
    CUDA_CHECK(cudaMemcpy(d_P, hostP.data(), sizeof(float) * D * D, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate(&m_handle));
}

// ---------------------------------------------------------------------------
// FhtKac-rotator init
// ---------------------------------------------------------------------------
void RotatorGPU::init_fht_kac(uint32_t dim) {
    dim_ = dim;
    D = rd_up(dim, 64);

    size_t bottom_log = floor_log2(dim);
    trunc_dim_ = 1ULL << bottom_log;
    log_N_ = static_cast<int>(bottom_log);
    fac_ = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));

    size_t flip_bytes = 4 * D / 8;
    std::vector<uint8_t> h_flip(flip_bytes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& b : h_flip) b = static_cast<uint8_t>(dist(gen));

    CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
    CUDA_CHECK(cudaMemcpy(d_flip_, h_flip.data(), flip_bytes, cudaMemcpyHostToDevice));
}

void RotatorGPU::free_resources() {
    if (d_P)    { cudaFree(d_P);    d_P    = nullptr; }
    if (d_flip_){ cudaFree(d_flip_); d_flip_ = nullptr; }
    if (type_ == RotatorType::Matrix && m_handle) {
        cublasDestroy(m_handle);
        m_handle = {};
    }
}

RotatorGPU::RotatorGPU(uint32_t dim, RotatorType type) : type_(type) {
    if (type == RotatorType::Matrix) init_matrix(dim);
    else                             init_fht_kac(dim);
}

RotatorGPU::~RotatorGPU() {
    free_resources();
}

RotatorGPU& RotatorGPU::operator=(const RotatorGPU& other) {
    if (this == &other) return *this;
    free_resources();

    type_ = other.type_;
    D = other.D;

    if (type_ == RotatorType::Matrix) {
        CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * D * D));
        CUDA_CHECK(cudaMemcpy(d_P, other.d_P, sizeof(float) * D * D, cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasCreate(&m_handle));
    } else {
        dim_ = other.dim_;
        trunc_dim_ = other.trunc_dim_;
        fac_ = other.fac_;
        log_N_ = other.log_N_;

        size_t flip_bytes = 4 * D / 8;
        CUDA_CHECK(cudaMalloc(&d_flip_, flip_bytes));
        CUDA_CHECK(cudaMemcpy(d_flip_, other.d_flip_, flip_bytes, cudaMemcpyDeviceToDevice));
    }
    return *this;
}

void RotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const {
    if (type_ == RotatorType::Matrix) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major row-of-A (size D) is reinterpreted as column-of-A in column-major;
        // compute C = P · A by calling sgemm with leading dims = D in both inputs.
        CUBLAS_CHECK(cublasSgemm(m_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 static_cast<int>(D), static_cast<int>(N), static_cast<int>(D),
                                 &alpha,
                                 d_P, static_cast<int>(D),
                                 d_A, static_cast<int>(D),
                                 &beta,
                                 d_RAND_A, static_cast<int>(D)));
    } else {
        cudaStream_t stream = 0;
        if (trunc_dim_ == D) {
            float total_scale = fac_ * fac_ * fac_ * fac_;
            fht::dispatch_fused_rotate(d_A, d_RAND_A, d_flip_,
                                       static_cast<int>(N), log_N_,
                                       total_scale, stream);
        } else {
            fht::dispatch_fused_rotate_nonpow2(d_A, d_RAND_A, d_flip_,
                                                static_cast<int>(N), log_N_,
                                                static_cast<int>(D),
                                                fac_, 0.25f, stream);
        }
    }
}

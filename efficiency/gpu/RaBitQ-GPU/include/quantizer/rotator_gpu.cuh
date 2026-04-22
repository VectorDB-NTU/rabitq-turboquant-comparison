//
// GPU rotator supporting two implementations:
//   - Matrix : full D×D random orthogonal matrix via cuBLAS sgemm, O(N*D^2)
//   - FhtKac : Fast Hadamard Transform + Kac's walk, O(N*D*logD)
//
// Both types are interchangeable at construction time; the public interface
// is identical.
//

#ifndef RABITQ_GPU_ROTATOR_GPU_CUH
#define RABITQ_GPU_ROTATOR_GPU_CUH

#include <cstdint>
#include <cublas_v2.h>
#include "utils/utils_cuda.cuh"

enum class RotatorType : uint8_t {
    Matrix = 0,
    FhtKac = 1,
};

class RotatorGPU {
private:
    RotatorType type_;
    size_t D = 0;            // Padded dimension (multiple of 64)

    // Matrix-rotator state
    float* d_P = nullptr;
    cublasHandle_t m_handle{};

    // FhtKac-rotator state
    size_t dim_ = 0;
    size_t trunc_dim_ = 0;
    float fac_ = 0;
    int log_N_ = 0;
    uint8_t* d_flip_ = nullptr;

    void init_matrix(uint32_t dim);
    void init_fht_kac(uint32_t dim);
    void free_resources();

public:
    explicit RotatorGPU(uint32_t dim, RotatorType type = RotatorType::FhtKac);
    RotatorGPU() : type_(RotatorType::FhtKac) {}
    ~RotatorGPU();

    RotatorGPU& operator=(const RotatorGPU& other);

    size_t size() const { return D; }
    RotatorType rotator_type() const { return type_; }

    /// Rotate N vectors of D floats. In-place aliasing (d_A == d_RAND_A) is
    /// permitted only for FhtKac.
    void rotate(const float* d_A, float* d_RAND_A, size_t N) const;
    bool supports_inplace_rotate() const { return type_ == RotatorType::FhtKac; }
};

#endif // RABITQ_GPU_ROTATOR_GPU_CUH

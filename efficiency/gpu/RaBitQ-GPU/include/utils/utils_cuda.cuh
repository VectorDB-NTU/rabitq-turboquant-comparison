#ifndef RABITQ_GPU_UTILS_CUDA_CUH
#define RABITQ_GPU_UTILS_CUDA_CUH

#include <iostream>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " " << cudaGetErrorString(err)\
                      << std::endl;                               \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#define CUBLAS_CHECK(expr)                                                    \
    do {                                                                     \
        cublasStatus_t _st = (expr);                                         \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " - " << _st << "\n";                               \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#endif // RABITQ_GPU_UTILS_CUDA_CUH

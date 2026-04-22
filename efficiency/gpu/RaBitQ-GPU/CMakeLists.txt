cmake_minimum_required(VERSION 3.18)

# Common data-center / workstation targets:
#   80  = A100
#   86  = A5000 / A6000 / RTX 30xx
#   89  = RTX 40xx / L40(S)
#   120 = RTX PRO 6000 Blackwell
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80;86;89;120)
endif()

project(RaBitQ_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Prefer CUDA from the active conda/mamba env when available.
if(DEFINED ENV{CONDA_PREFIX})
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Host compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Device flags.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# ----------------------------------------------------------------------------
# Library: rabitq_gpu
# ----------------------------------------------------------------------------
add_library(rabitq_gpu STATIC
    src/quantizer/rotator_gpu.cu
    src/quantizer/rescale_search_gpu.cu
    src/quantizer/quantizer_standalone.cu
)

target_include_directories(rabitq_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rabitq_gpu
    PUBLIC
    CUDA::cudart
    CUDA::cublas
)

# ----------------------------------------------------------------------------
# Tests / benchmark executables
# ----------------------------------------------------------------------------
add_executable(test_quantizer_standalone
    tests/test_quantizer_standalone.cu
)

target_link_libraries(test_quantizer_standalone
    PRIVATE
    rabitq_gpu
    OpenMP::OpenMP_CXX
)

# Give the CUDA compiler access to OpenMP on the host side.
target_compile_options(test_quantizer_standalone
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

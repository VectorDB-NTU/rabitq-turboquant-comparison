cmake_minimum_required(VERSION 3.10)

project(ann_recall_reproduction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../RaBitQ-Library/include)

add_executable(rabitq rabitq.cpp)
target_compile_options(
    rabitq
    PRIVATE
        -Wall
        -Ofast
        -Wextra
        -march=native
        -ftree-vectorize
        -fexceptions
)
target_link_libraries(rabitq PRIVATE OpenMP::OpenMP_CXX rt)

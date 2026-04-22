#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

//
// Minimal row-major matrix type replacing Eigen::Matrix for this project.
// Supports the tiny API used by utils/IO.hpp (rows, cols, data, (i,j) indexing).
//
template <typename T>
class RowMajorMatrix {
public:
    using Scalar = T;

    RowMajorMatrix() : rows_(0), cols_(0) {}
    RowMajorMatrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), buf_(rows * cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T* data() { return buf_.data(); }
    const T* data() const { return buf_.data(); }

    // Eigen-compatible (row, col) indexing used by IO.hpp.
    T& operator()(size_t r, size_t c)        { return buf_[r * cols_ + c]; }
    T  operator()(size_t r, size_t c) const  { return buf_[r * cols_ + c]; }

    // Eigen also accepts signed arguments in some call sites.
    T& operator()(long r, long c)            { return buf_[static_cast<size_t>(r) * cols_ + static_cast<size_t>(c)]; }
    T  operator()(long r, long c) const      { return buf_[static_cast<size_t>(r) * cols_ + static_cast<size_t>(c)]; }

private:
    size_t rows_;
    size_t cols_;
    std::vector<T> buf_;
};

using FloatRowMat = RowMajorMatrix<float>;
using PID = uint32_t;

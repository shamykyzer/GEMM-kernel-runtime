#include "tensor.h"

#include <algorithm>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

namespace tile_runtime {

static size_t checked_size(size_t rows, size_t cols) {
    // Test cols first: it avoids dividing by zero, and a zero dimension is legal.
    if (cols != 0 && rows > std::numeric_limits<size_t>::max() / cols) {
        throw std::length_error("Tensor: rows * cols overflows size_t (" +
            std::to_string(rows) + " x " + std::to_string(cols) + ")");
    }
    return rows * cols;
}

Tensor::Tensor() : rows_(0), cols_(0) {}

Tensor::Tensor(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(checked_size(rows, cols), 0.0f) {}

Tensor::Tensor(size_t rows, size_t cols, float init_value)
    : rows_(rows), cols_(cols), data_(checked_size(rows, cols), init_value) {}

Tensor::Tensor(Tensor&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_.clear();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_.clear();
    return *this;
}

void Tensor::bounds_check(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        std::ostringstream oss;
        oss << "Tensor index (" << i << ", " << j
            << ") out of range for shape (" << rows_ << ", " << cols_ << ")";
        throw std::out_of_range(oss.str());
    }
}

float& Tensor::at(size_t i, size_t j) {
    bounds_check(i, j);
    return data_[i * cols_ + j];
}

const float& Tensor::at(size_t i, size_t j) const {
    bounds_check(i, j);
    return data_[i * cols_ + j];
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero() {
    fill(0.0f);
}

void Tensor::randomize(unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data_) {
        v = dist(gen);
    }
}

}  // namespace tile_runtime

// LightwatchAI2 API Contract: Tensor
// Defined by: Phase 03
// Consumers: 04, 05, 08, 09, 11-19, 21-25, 31-36
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include <vector>
#include <initializer_list>
#include <memory>
#include <cstddef>

namespace lightwatch {

using Shape = std::vector<size_t>;

template<typename T>
class Tensor {
public:
    // Construction
    Tensor();
    explicit Tensor(const Shape& shape);
    Tensor(const Shape& shape, const T* data);
    Tensor(const Shape& shape, std::initializer_list<T> data);

    // Static factories
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor full(const Shape& shape, T value);
    static Tensor randn(const Shape& shape);  // Normal distribution N(0,1)
    static Tensor rand(const Shape& shape);   // Uniform [0,1)

    // Element access
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    // Properties
    const Shape& shape() const;
    size_t size(int dim) const;  // Negative dims count from end
    size_t numel() const;        // Total elements
    size_t ndim() const;
    T* data();
    const T* data() const;

    // Shape operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const;  // May share data
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    Tensor contiguous() const;
    bool is_contiguous() const;

    // Reductions
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim = -1, bool keepdim = false) const;
    Tensor min(int dim = -1, bool keepdim = false) const;
    Tensor var(int dim = -1, bool keepdim = false) const;
    T item() const;  // For scalar tensors

    // Element-wise ops (return new tensor)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Hadamard product
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(T exponent) const;

    // Scalar ops
    Tensor operator+(T scalar) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(T scalar) const;
    Tensor operator/(T scalar) const;

    // In-place ops (return reference for chaining)
    Tensor& fill_(T value);
    Tensor& zero_();
    Tensor& add_(const Tensor& other);
    Tensor& sub_(const Tensor& other);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);

    // Comparison (return boolean tensor)
    Tensor<bool> operator==(const Tensor& other) const;
    Tensor<bool> operator!=(const Tensor& other) const;
    Tensor<bool> operator<(const Tensor& other) const;
    Tensor<bool> operator<=(const Tensor& other) const;
    Tensor<bool> operator>(const Tensor& other) const;
    Tensor<bool> operator>=(const Tensor& other) const;

    // Utilities
    Tensor clone() const;

private:
    Shape shape_;
    std::shared_ptr<T[]> data_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
};

// Free functions
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int dim);

template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int dim);

template<typename T>
Tensor<T> where(const Tensor<bool>& condition, const Tensor<T>& x, const Tensor<T>& y);

}  // namespace lightwatch

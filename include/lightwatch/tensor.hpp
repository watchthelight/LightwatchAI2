#pragma once

#include <vector>
#include <initializer_list>
#include <memory>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

#include <lightwatch/memory/aligned.hpp>

namespace lightwatch {

using Shape = std::vector<size_t>;

namespace detail {

// Compute strides for row-major layout
inline std::vector<size_t> compute_strides(const Shape& shape) {
    if (shape.empty()) return {};
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Normalize negative dimension
inline int normalize_dim(int dim, size_t ndim) {
    if (dim < 0) {
        dim += static_cast<int>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        throw std::out_of_range("Dimension out of range");
    }
    return dim;
}

// Compute linear index from multi-dimensional indices
inline size_t compute_offset(const std::vector<size_t>& indices,
                             const std::vector<size_t>& strides,
                             size_t base_offset) {
    size_t offset = base_offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

// Compute total number of elements
inline size_t compute_numel(const Shape& shape) {
    if (shape.empty()) return 1;  // Scalar
    return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

// Global random engine for reproducibility
inline std::mt19937& get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

}  // namespace detail

// Forward declaration for comparison operators
template<typename T> class Tensor;

template<typename T>
class Tensor {
public:
    // Default constructor - creates empty tensor
    Tensor() : offset_(0) {}

    // Construct with shape (uninitialized data)
    explicit Tensor(const Shape& shape)
        : shape_(shape),
          strides_(detail::compute_strides(shape)),
          offset_(0) {
        size_t n = detail::compute_numel(shape);
        if (n > 0) {
            T* ptr = memory::aligned_new<T>(n, 64);
            data_ = std::shared_ptr<T[]>(ptr, [n](T* p) { memory::aligned_delete(p, n); });
        }
    }

    // Construct with shape and data pointer (copies data)
    Tensor(const Shape& shape, const T* data)
        : Tensor(shape) {
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            data_.get()[i] = data[i];
        }
    }

    // Construct with shape and initializer list
    Tensor(const Shape& shape, std::initializer_list<T> data)
        : Tensor(shape) {
        size_t n = std::min(numel(), data.size());
        auto it = data.begin();
        for (size_t i = 0; i < n; ++i, ++it) {
            data_.get()[i] = *it;
        }
    }

    // Static factories
    static Tensor zeros(const Shape& shape) {
        Tensor t(shape);
        t.fill_(T{0});
        return t;
    }

    static Tensor ones(const Shape& shape) {
        Tensor t(shape);
        t.fill_(T{1});
        return t;
    }

    static Tensor full(const Shape& shape, T value) {
        Tensor t(shape);
        t.fill_(value);
        return t;
    }

    static Tensor randn(const Shape& shape) {
        Tensor t(shape);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        size_t n = t.numel();
        T* ptr = t.data_.get();
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<T>(dist(detail::get_rng()));
        }
        return t;
    }

    static Tensor rand(const Shape& shape) {
        Tensor t(shape);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        size_t n = t.numel();
        T* ptr = t.data_.get();
        for (size_t i = 0; i < n; ++i) {
            ptr[i] = static_cast<T>(dist(detail::get_rng()));
        }
        return t;
    }

    // Element access with bounds checking
    T& at(const std::vector<size_t>& indices) {
        check_indices(indices);
        return data_.get()[detail::compute_offset(indices, strides_, offset_)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        check_indices(indices);
        return data_.get()[detail::compute_offset(indices, strides_, offset_)];
    }

    // Element access without bounds checking
    T& operator()(const std::vector<size_t>& indices) {
        return data_.get()[detail::compute_offset(indices, strides_, offset_)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data_.get()[detail::compute_offset(indices, strides_, offset_)];
    }

    // Properties
    const Shape& shape() const { return shape_; }

    size_t size(int dim) const {
        int d = detail::normalize_dim(dim, ndim());
        return shape_[d];
    }

    size_t numel() const {
        return detail::compute_numel(shape_);
    }

    size_t ndim() const {
        return shape_.size();
    }

    T* data() {
        return is_contiguous() ? (data_.get() + offset_) : nullptr;
    }

    const T* data() const {
        return is_contiguous() ? (data_.get() + offset_) : nullptr;
    }

    // Check if memory is contiguous
    bool is_contiguous() const {
        if (shape_.empty()) return true;
        std::vector<size_t> expected = detail::compute_strides(shape_);
        return strides_ == expected;
    }

    // Make contiguous copy if needed
    Tensor contiguous() const {
        if (is_contiguous()) return *this;
        Tensor result(shape_);
        copy_to(result);
        return result;
    }

    // Reshape (returns view if possible)
    Tensor reshape(const Shape& new_shape) const {
        // Validate numel matches
        size_t new_numel = detail::compute_numel(new_shape);
        if (new_numel != numel()) {
            throw std::invalid_argument("reshape: number of elements must match");
        }

        if (is_contiguous()) {
            // Can return a view
            Tensor result;
            result.shape_ = new_shape;
            result.strides_ = detail::compute_strides(new_shape);
            result.data_ = data_;
            result.offset_ = offset_;
            return result;
        } else {
            // Must copy
            return contiguous().reshape(new_shape);
        }
    }

    // View (same as reshape, shares data)
    Tensor view(const Shape& new_shape) const {
        return reshape(new_shape);
    }

    // Transpose two dimensions
    Tensor transpose(int dim0, int dim1) const {
        int d0 = detail::normalize_dim(dim0, ndim());
        int d1 = detail::normalize_dim(dim1, ndim());

        Tensor result;
        result.shape_ = shape_;
        result.strides_ = strides_;
        result.data_ = data_;
        result.offset_ = offset_;

        std::swap(result.shape_[d0], result.shape_[d1]);
        std::swap(result.strides_[d0], result.strides_[d1]);

        return result;
    }

    // Permute dimensions
    Tensor permute(const std::vector<int>& dims) const {
        if (dims.size() != ndim()) {
            throw std::invalid_argument("permute: dims size must match ndim");
        }

        Tensor result;
        result.shape_.resize(ndim());
        result.strides_.resize(ndim());
        result.data_ = data_;
        result.offset_ = offset_;

        for (size_t i = 0; i < dims.size(); ++i) {
            int d = detail::normalize_dim(dims[i], ndim());
            result.shape_[i] = shape_[d];
            result.strides_[i] = strides_[d];
        }

        return result;
    }

    // Squeeze dimension (remove size-1 dims)
    Tensor squeeze(int dim = -1) const {
        Shape new_shape;
        std::vector<size_t> new_strides;

        if (dim == -1) {
            // Remove all size-1 dimensions
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (shape_[i] != 1) {
                    new_shape.push_back(shape_[i]);
                    new_strides.push_back(strides_[i]);
                }
            }
        } else {
            int d = detail::normalize_dim(dim, ndim());
            if (shape_[d] != 1) {
                throw std::invalid_argument("squeeze: dimension size must be 1");
            }
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (static_cast<int>(i) != d) {
                    new_shape.push_back(shape_[i]);
                    new_strides.push_back(strides_[i]);
                }
            }
        }

        if (new_shape.empty()) {
            new_shape = {};
            new_strides = {};
        }

        Tensor result;
        result.shape_ = new_shape;
        result.strides_ = new_strides;
        result.data_ = data_;
        result.offset_ = offset_;
        return result;
    }

    // Unsqueeze dimension (add size-1 dim)
    Tensor unsqueeze(int dim) const {
        // Allow dim == ndim() for inserting at end
        int d = dim;
        if (d < 0) d += static_cast<int>(ndim()) + 1;
        if (d < 0 || d > static_cast<int>(ndim())) {
            throw std::out_of_range("unsqueeze: dimension out of range");
        }

        Shape new_shape = shape_;
        std::vector<size_t> new_strides = strides_;

        size_t stride = (d < static_cast<int>(strides_.size())) ? strides_[d] : 1;
        new_shape.insert(new_shape.begin() + d, 1);
        new_strides.insert(new_strides.begin() + d, stride);

        Tensor result;
        result.shape_ = new_shape;
        result.strides_ = new_strides;
        result.data_ = data_;
        result.offset_ = offset_;
        return result;
    }

    // Slice along dimension
    Tensor slice(int dim, size_t start, size_t end) const {
        int d = detail::normalize_dim(dim, ndim());
        if (start >= end || end > shape_[d]) {
            throw std::out_of_range("slice: invalid range");
        }

        Tensor result;
        result.shape_ = shape_;
        result.shape_[d] = end - start;
        result.strides_ = strides_;
        result.data_ = data_;
        result.offset_ = offset_ + start * strides_[d];
        return result;
    }

    // Reduction operations
    Tensor sum(int dim = -1, bool keepdim = false) const {
        return reduce_op(dim, keepdim, [](T a, T b) { return a + b; }, T{0});
    }

    Tensor mean(int dim = -1, bool keepdim = false) const {
        if (dim == -1) {
            Tensor s = sum(-1, keepdim);
            T n = static_cast<T>(numel());
            return s / n;
        }
        int d = detail::normalize_dim(dim, ndim());
        Tensor s = sum(dim, keepdim);
        T n = static_cast<T>(shape_[d]);
        return s / n;
    }

    Tensor max(int dim = -1, bool keepdim = false) const {
        return reduce_op(dim, keepdim, [](T a, T b) { return (a > b) ? a : b; },
                        std::numeric_limits<T>::lowest());
    }

    Tensor min(int dim = -1, bool keepdim = false) const {
        return reduce_op(dim, keepdim, [](T a, T b) { return (a < b) ? a : b; },
                        std::numeric_limits<T>::max());
    }

    Tensor var(int dim = -1, bool keepdim = false) const {
        Tensor m = mean(dim, true);
        Tensor diff = *this - broadcast_to(m, shape_);
        Tensor sq = diff * diff;
        return sq.mean(dim, keepdim);
    }

    // Get scalar value
    T item() const {
        if (numel() != 1) {
            throw std::runtime_error("item() requires exactly one element");
        }
        return at(std::vector<size_t>(ndim(), 0));
    }

    // Element-wise operations
    Tensor operator+(const Tensor& other) const {
        return binary_op(other, [](T a, T b) { return a + b; });
    }

    Tensor operator-(const Tensor& other) const {
        return binary_op(other, [](T a, T b) { return a - b; });
    }

    Tensor operator*(const Tensor& other) const {
        return binary_op(other, [](T a, T b) { return a * b; });
    }

    Tensor operator/(const Tensor& other) const {
        return binary_op(other, [](T a, T b) { return a / b; });
    }

    Tensor operator-() const {
        return unary_op([](T a) { return -a; });
    }

    Tensor abs() const {
        return unary_op([](T a) { return std::abs(a); });
    }

    Tensor sqrt() const {
        return unary_op([](T a) { return std::sqrt(a); });
    }

    Tensor exp() const {
        return unary_op([](T a) { return std::exp(a); });
    }

    Tensor log() const {
        return unary_op([](T a) { return std::log(a); });
    }

    Tensor pow(T exponent) const {
        return unary_op([exponent](T a) { return std::pow(a, exponent); });
    }

    // Scalar operations
    Tensor operator+(T scalar) const {
        return unary_op([scalar](T a) { return a + scalar; });
    }

    Tensor operator-(T scalar) const {
        return unary_op([scalar](T a) { return a - scalar; });
    }

    Tensor operator*(T scalar) const {
        return unary_op([scalar](T a) { return a * scalar; });
    }

    Tensor operator/(T scalar) const {
        return unary_op([scalar](T a) { return a / scalar; });
    }

    // In-place operations
    Tensor& fill_(T value) {
        iterate([value](T& elem) { elem = value; });
        return *this;
    }

    Tensor& zero_() {
        return fill_(T{0});
    }

    Tensor& add_(const Tensor& other) {
        binary_op_inplace(other, [](T& a, T b) { a += b; });
        return *this;
    }

    Tensor& sub_(const Tensor& other) {
        binary_op_inplace(other, [](T& a, T b) { a -= b; });
        return *this;
    }

    Tensor& mul_(const Tensor& other) {
        binary_op_inplace(other, [](T& a, T b) { a *= b; });
        return *this;
    }

    Tensor& div_(const Tensor& other) {
        binary_op_inplace(other, [](T& a, T b) { a /= b; });
        return *this;
    }

    // Comparison operators (return Tensor<bool>)
    Tensor<bool> operator==(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a == b; });
    }

    Tensor<bool> operator!=(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a != b; });
    }

    Tensor<bool> operator<(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a < b; });
    }

    Tensor<bool> operator<=(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a <= b; });
    }

    Tensor<bool> operator>(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a > b; });
    }

    Tensor<bool> operator>=(const Tensor& other) const {
        return compare_op(other, [](T a, T b) { return a >= b; });
    }

    // Clone (deep copy)
    Tensor clone() const {
        Tensor result(shape_);
        copy_to(result);
        return result;
    }

private:
    Shape shape_;
    std::shared_ptr<T[]> data_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;

    // Bounds checking
    void check_indices(const std::vector<size_t>& indices) const {
        if (indices.size() != ndim()) {
            throw std::out_of_range("Wrong number of indices");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
    }

    // Iterate over all elements
    template<typename F>
    void iterate(F&& func) {
        std::vector<size_t> indices(ndim(), 0);
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            func((*this)(indices));
            // Increment indices
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    // Const iterate
    template<typename F>
    void iterate(F&& func) const {
        std::vector<size_t> indices(ndim(), 0);
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            func((*this)(indices));
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    // Copy to another tensor
    void copy_to(Tensor& dest) const {
        std::vector<size_t> indices(ndim(), 0);
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            dest(indices) = (*this)(indices);
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    // Unary operation
    template<typename F>
    Tensor unary_op(F&& func) const {
        Tensor result(shape_);
        std::vector<size_t> indices(ndim(), 0);
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            result(indices) = func((*this)(indices));
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
        return result;
    }

    // Binary operation with broadcasting
    template<typename F>
    Tensor binary_op(const Tensor& other, F&& func) const {
        Shape broadcast_shape = compute_broadcast_shape(shape_, other.shape_);
        Tensor result(broadcast_shape);

        std::vector<size_t> indices(broadcast_shape.size(), 0);
        size_t n = detail::compute_numel(broadcast_shape);
        for (size_t i = 0; i < n; ++i) {
            T a = get_broadcast_value(*this, indices, shape_, broadcast_shape);
            T b = get_broadcast_value(other, indices, other.shape_, broadcast_shape);
            result(indices) = func(a, b);
            for (int d = static_cast<int>(broadcast_shape.size()) - 1; d >= 0; --d) {
                if (++indices[d] < broadcast_shape[d]) break;
                indices[d] = 0;
            }
        }
        return result;
    }

    // Binary operation in-place
    template<typename F>
    void binary_op_inplace(const Tensor& other, F&& func) {
        std::vector<size_t> indices(ndim(), 0);
        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            T b = get_broadcast_value(other, indices, other.shape_, shape_);
            func((*this)(indices), b);
            for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    // Comparison operation
    template<typename F>
    Tensor<bool> compare_op(const Tensor& other, F&& func) const {
        Shape broadcast_shape = compute_broadcast_shape(shape_, other.shape_);
        Tensor<bool> result(broadcast_shape);

        std::vector<size_t> indices(broadcast_shape.size(), 0);
        size_t n = detail::compute_numel(broadcast_shape);
        for (size_t i = 0; i < n; ++i) {
            T a = get_broadcast_value(*this, indices, shape_, broadcast_shape);
            T b = get_broadcast_value(other, indices, other.shape_, broadcast_shape);
            result(indices) = func(a, b);
            for (int d = static_cast<int>(broadcast_shape.size()) - 1; d >= 0; --d) {
                if (++indices[d] < broadcast_shape[d]) break;
                indices[d] = 0;
            }
        }
        return result;
    }

    // Reduction operation
    template<typename F>
    Tensor reduce_op(int dim, bool keepdim, F&& func, T init) const {
        if (dim == -1) {
            // Global reduction
            T result = init;
            iterate([&result, &func](const T& elem) {
                result = func(result, elem);
            });
            if (keepdim) {
                Shape ones_shape(ndim(), 1);
                Tensor r(ones_shape);
                r.fill_(result);
                return r;
            }
            return Tensor::full({}, result);
        }

        int d = detail::normalize_dim(dim, ndim());
        Shape result_shape;
        for (size_t i = 0; i < ndim(); ++i) {
            if (static_cast<int>(i) == d) {
                if (keepdim) result_shape.push_back(1);
            } else {
                result_shape.push_back(shape_[i]);
            }
        }

        Tensor result = Tensor::full(result_shape, init);

        std::vector<size_t> src_indices(ndim(), 0);
        std::vector<size_t> dst_indices(result_shape.size(), 0);

        size_t n = numel();
        for (size_t i = 0; i < n; ++i) {
            // Map source indices to dest indices
            size_t di = 0;
            for (size_t si = 0; si < ndim(); ++si) {
                if (static_cast<int>(si) == d) {
                    if (keepdim) {
                        dst_indices[di++] = 0;
                    }
                } else {
                    dst_indices[di++] = src_indices[si];
                }
            }

            result(dst_indices) = func(result(dst_indices), (*this)(src_indices));

            // Increment indices
            for (int id = static_cast<int>(ndim()) - 1; id >= 0; --id) {
                if (++src_indices[id] < shape_[id]) break;
                src_indices[id] = 0;
            }
        }

        return result;
    }

    // Broadcasting helpers
    static Shape compute_broadcast_shape(const Shape& a, const Shape& b) {
        size_t ndim = std::max(a.size(), b.size());
        Shape result(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            size_t da = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
            size_t db = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];
            if (da != db && da != 1 && db != 1) {
                throw std::invalid_argument("Shapes not broadcastable");
            }
            result[i] = std::max(da, db);
        }
        return result;
    }

    static T get_broadcast_value(const Tensor& t, const std::vector<size_t>& indices,
                                 const Shape& t_shape, const Shape& broadcast_shape) {
        std::vector<size_t> t_indices(t_shape.size());
        size_t offset = broadcast_shape.size() - t_shape.size();
        for (size_t i = 0; i < t_shape.size(); ++i) {
            t_indices[i] = (t_shape[i] == 1) ? 0 : indices[i + offset];
        }
        return t(t_indices);
    }

    static Tensor broadcast_to(const Tensor& t, const Shape& target_shape) {
        Tensor result(target_shape);
        std::vector<size_t> indices(target_shape.size(), 0);
        size_t n = detail::compute_numel(target_shape);
        for (size_t i = 0; i < n; ++i) {
            result(indices) = get_broadcast_value(t, indices, t.shape_, target_shape);
            for (int d = static_cast<int>(target_shape.size()) - 1; d >= 0; --d) {
                if (++indices[d] < target_shape[d]) break;
                indices[d] = 0;
            }
        }
        return result;
    }

    // Allow Tensor<bool> to access private members
    template<typename U> friend class Tensor;
};

// Free functions

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    // 2D matrix multiplication
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    if (a.size(1) != b.size(0)) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    size_t M = a.size(0);
    size_t K = a.size(1);
    size_t N = b.size(1);

    Tensor<T> result = Tensor<T>::zeros({M, N});

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T{0};
            for (size_t k = 0; k < K; ++k) {
                sum += a({i, k}) * b({k, j});
            }
            result({i, j}) = sum;
        }
    }

    return result;
}

template<typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat: empty tensor list");
    }

    const Tensor<T>& first = tensors[0];
    int d = detail::normalize_dim(dim, first.ndim());

    // Validate shapes
    size_t total_dim = 0;
    for (const auto& t : tensors) {
        if (t.ndim() != first.ndim()) {
            throw std::invalid_argument("concat: all tensors must have same ndim");
        }
        for (size_t i = 0; i < first.ndim(); ++i) {
            if (static_cast<int>(i) != d && t.size(static_cast<int>(i)) != first.size(static_cast<int>(i))) {
                throw std::invalid_argument("concat: shapes must match except in concat dim");
            }
        }
        total_dim += t.size(d);
    }

    // Create result shape
    Shape result_shape = first.shape();
    result_shape[d] = total_dim;
    Tensor<T> result(result_shape);

    // Copy data
    size_t offset = 0;
    for (const auto& t : tensors) {
        std::vector<size_t> src_indices(t.ndim(), 0);
        std::vector<size_t> dst_indices(t.ndim(), 0);
        size_t n = t.numel();

        for (size_t i = 0; i < n; ++i) {
            dst_indices = src_indices;
            dst_indices[d] += offset;
            result(dst_indices) = t(src_indices);

            for (int id = static_cast<int>(t.ndim()) - 1; id >= 0; --id) {
                if (++src_indices[id] < t.shape()[id]) break;
                src_indices[id] = 0;
            }
        }
        offset += t.size(d);
    }

    return result;
}

template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("stack: empty tensor list");
    }

    // Unsqueeze all tensors and concat
    std::vector<Tensor<T>> unsqueezed;
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return concat(unsqueezed, dim);
}

template<typename T>
Tensor<T> where(const Tensor<bool>& condition, const Tensor<T>& x, const Tensor<T>& y) {
    if (condition.shape() != x.shape() || condition.shape() != y.shape()) {
        throw std::invalid_argument("where: all tensors must have same shape");
    }

    Tensor<T> result(x.shape());
    std::vector<size_t> indices(x.ndim(), 0);
    size_t n = x.numel();

    for (size_t i = 0; i < n; ++i) {
        result(indices) = condition(indices) ? x(indices) : y(indices);
        for (int d = static_cast<int>(x.ndim()) - 1; d >= 0; --d) {
            if (++indices[d] < x.shape()[d]) break;
            indices[d] = 0;
        }
    }

    return result;
}

// Scalar on left side
template<typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& t) {
    return t + scalar;
}

template<typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& t) {
    return t * scalar;
}

template<typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& t) {
    return Tensor<T>::full(t.shape(), scalar) - t;
}

template<typename T>
Tensor<T> operator/(T scalar, const Tensor<T>& t) {
    return Tensor<T>::full(t.shape(), scalar) / t;
}

}  // namespace lightwatch

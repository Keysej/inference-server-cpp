#pragma once

#include <vector>
#include <span>
#include <concepts>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <string>
#include <cstdint>

#include <onnxruntime_cxx_api.h>

namespace inference {

// ── C++20 concept ─────────────────────────────────────────────────────────────
//
// Constraining the template at the concept level — rather than adding runtime
// checks inside the body — means an illegal type (e.g. std::string) is caught
// at the call site during compilation, not at test time or in production.

template<typename T>
concept TensorElement =
    std::same_as<T, float>    ||
    std::same_as<T, double>   ||
    std::same_as<T, int32_t>  ||
    std::same_as<T, int64_t>  ||
    std::same_as<T, uint8_t>;

// ── Compile-time type → ONNXTensorElementDataType mapping ────────────────────
//
// if constexpr lets us embed the mapping in a single function without any
// runtime dispatch.  A static_assert at the bottom makes compilation fail fast
// if someone satisfies TensorElement with a new type but forgets to add a case.

template<TensorElement T>
constexpr ONNXTensorElementDataType ort_element_type() {
    // else-if chain is required: a bare static_assert after independent
    // if constexpr returns fires even for the matching branch.
    if constexpr      (std::same_as<T, float>)   return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if constexpr (std::same_as<T, double>)  return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    else if constexpr (std::same_as<T, int32_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    else if constexpr (std::same_as<T, int64_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    else if constexpr (std::same_as<T, uint8_t>) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    else static_assert(sizeof(T) == 0, "ort_element_type: unhandled TensorElement type");
}

// ── Owning, typed Tensor ──────────────────────────────────────────────────────

template<TensorElement T>
class Tensor {
public:
    Tensor(std::vector<int64_t> shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data))
    {
        if (element_count() != static_cast<int64_t>(data_.size()))
            throw std::invalid_argument("Tensor: shape/data size mismatch");
    }

    // Factory: allocate an uninitialized tensor of the given shape.
    static Tensor zeroed(std::vector<int64_t> shape) {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return Tensor(std::move(shape), std::vector<T>(n, T{}));
    }

    // std::span gives zero-copy access — the caller never needs to copy the
    // buffer just to read or pass it to ONNX Runtime.
    [[nodiscard]] std::span<const T>          view()  const noexcept { return data_; }
    [[nodiscard]] std::span<T>                view()        noexcept { return data_; }
    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] T*       data()       noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    // std::multiplies<> (heterogeneous) avoids an extra int64_t cast per element.
    [[nodiscard]] int64_t element_count() const noexcept {
        return std::accumulate(shape_.begin(), shape_.end(),
                               int64_t{1}, std::multiplies<int64_t>{});
    }

    [[nodiscard]] constexpr ONNXTensorElementDataType ort_dtype() const noexcept {
        return ort_element_type<T>();
    }

private:
    std::vector<int64_t> shape_;
    std::vector<T>       data_;
};

// ── Convenience aliases ───────────────────────────────────────────────────────

using FloatTensor = Tensor<float>;
using Int64Tensor = Tensor<int64_t>;

// ── ORT bridge ───────────────────────────────────────────────────────────────
//
// to_ort_value   — wraps our buffer in an Ort::Value with zero copies.
//                  IMPORTANT: the Tensor must outlive the returned Ort::Value.
//
// from_ort_value — copies ORT output data into an owning Tensor after a run().

template<TensorElement T>
[[nodiscard]] Ort::Value to_ort_value(Tensor<T>& tensor) {
    // CPU memory info is stack-allocated here; ORT copies what it needs from it.
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<T>(
        mem,
        tensor.data(),
        static_cast<size_t>(tensor.element_count()),
        tensor.shape().data(),
        tensor.shape().size()
    );
}

template<TensorElement T>
[[nodiscard]] Tensor<T> from_ort_value(Ort::Value& val) {
    // ConstTensorTypeAndShapeInfo is an unowned view — keep val alive.
    auto shape_info = val.GetTensorTypeAndShapeInfo();
    auto shape      = shape_info.GetShape();
    auto n          = static_cast<size_t>(shape_info.GetElementCount());
    const T* ptr    = val.GetTensorData<T>();
    return Tensor<T>(std::move(shape), std::vector<T>(ptr, ptr + n));
}

} // namespace inference

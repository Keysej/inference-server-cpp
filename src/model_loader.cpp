#include "model_loader.h"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string_view>  // zero-copy constant string handling

namespace inference {

// ── TensorInfo helpers ────────────────────────────────────────────────────────

int64_t TensorInfo::static_element_count() const noexcept {
    int64_t n = 1;
    for (auto d : shape) {
        if (d < 0) return -1;
        n *= d;
    }
    return n;
}

std::string TensorInfo::shape_str() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += (shape[i] < 0) ? "?" : std::to_string(shape[i]);
    }
    return s + "]";
}

// ── ModelLoader ───────────────────────────────────────────────────────────────

ModelLoader::ModelLoader(const std::filesystem::path& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "inference_server"),
      session_{nullptr}   // initialised below after path check
{
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path.string());
    }

    file_bytes_ = std::filesystem::file_size(model_path);
    name_       = model_path.stem().string();

    // Tune the session for low-latency single-request inference:
    //   • 1 intra-op thread  — avoids thread-spawn overhead on small models
    //   • graph optimisation level ORT_ENABLE_ALL — fuses ops at load time
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), opts);
    read_tensor_info();
}

void ModelLoader::read_tensor_info() {
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n_in  = session_.GetInputCount();
    const size_t n_out = session_.GetOutputCount();

    inputs_.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        auto name_ptr  = session_.GetInputNameAllocated(i, alloc);
        // type_info must stay alive — ConstTensorTypeAndShapeInfo is an unowned
        // view into it, so chaining on one line would leave a dangling pointer.
        auto type_info = session_.GetInputTypeInfo(i);
        auto ti        = type_info.GetTensorTypeAndShapeInfo();
        inputs_.push_back({
            .name  = name_ptr.get(),
            .dtype = ti.GetElementType(),
            .shape = ti.GetShape()
        });
    }

    outputs_.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        auto name_ptr  = session_.GetOutputNameAllocated(i, alloc);
        auto type_info = session_.GetOutputTypeInfo(i);
        auto ti        = type_info.GetTensorTypeAndShapeInfo();
        outputs_.push_back({
            .name  = name_ptr.get(),
            .dtype = ti.GetElementType(),
            .shape = ti.GetShape()
        });
    }
}

void ModelLoader::log_metadata() const {
    // File size as human-readable string
    auto size_str = [](std::uintmax_t b) -> std::string {
        if (b < 1024)       return std::to_string(b) + " B";
        if (b < 1024*1024)  return std::to_string(b / 1024) + " KB";
        return std::to_string(b / (1024*1024)) + " MB";
    };

    std::cout << "\n┌──────────────────────────────────────────\n";
    std::cout << "│  Model   : " << name_ << '\n';
    std::cout << "│  Size    : " << size_str(file_bytes_) << '\n';
    std::cout << "│\n";

    std::cout << "│  Inputs  (" << inputs_.size() << ")\n";
    for (const auto& t : inputs_) {
        std::cout << "│    [" << t.name << "]"
                  << "  " << dtype_name(t.dtype)
                  << "  " << t.shape_str();
        auto n = t.static_element_count();
        if (n > 0) std::cout << "  (" << n << " elements)";
        std::cout << '\n';
    }

    std::cout << "│\n│  Outputs (" << outputs_.size() << ")\n";
    for (const auto& t : outputs_) {
        std::cout << "│    [" << t.name << "]"
                  << "  " << dtype_name(t.dtype)
                  << "  " << t.shape_str();
        auto n = t.static_element_count();
        if (n > 0) std::cout << "  (" << n << " elements)";
        std::cout << '\n';
    }

    std::cout << "└──────────────────────────────────────────\n\n";
}

std::string ModelLoader::dtype_name(ONNXTensorElementDataType t) {
    // string_view avoids a heap allocation for these short constant strings.
    using sv = std::string_view;
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return std::string(sv{"float32"});
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return std::string(sv{"float64"});
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return std::string(sv{"int32"});
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return std::string(sv{"int64"});
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return std::string(sv{"uint8"});
        default:                                     return std::string(sv{"unknown"});
    }
}

} // namespace inference

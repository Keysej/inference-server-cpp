#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>

#include <onnxruntime_cxx_api.h>

namespace inference {

// Describes one tensor input or output — name, element type, shape.
// shape[i] == -1 means that dimension is dynamic (unknown at load time).
struct TensorInfo {
    std::string              name;
    ONNXTensorElementDataType dtype;
    std::vector<int64_t>     shape;

    // Returns static element count, or -1 if any dimension is dynamic.
    [[nodiscard]] int64_t static_element_count() const noexcept;

    // Human-readable shape: "[-1, 4]", "[1, 3, 224, 224]", etc.
    [[nodiscard]] std::string shape_str() const;
};

// ModelLoader owns the OrtSession and exposes typed tensor metadata.
//
// Single-responsibility: this class only opens the file and reads shapes.
// It never runs a forward pass — that belongs to InferenceEngine (Step 4).
class ModelLoader {
public:
    explicit ModelLoader(const std::filesystem::path& model_path);

    // Non-copyable — OrtSession holds non-shareable runtime state.
    ModelLoader(const ModelLoader&)            = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader(ModelLoader&&)                 = default;
    ModelLoader& operator=(ModelLoader&&)      = default;

    [[nodiscard]] const std::vector<TensorInfo>& inputs()     const noexcept { return inputs_;  }
    [[nodiscard]] const std::vector<TensorInfo>& outputs()    const noexcept { return outputs_; }
    [[nodiscard]] const std::string& model_name()             const noexcept { return name_;    }
    [[nodiscard]] std::uintmax_t     model_file_bytes()       const noexcept { return file_bytes_; }

    // Grants InferenceEngine direct session access without an extra wrapper.
    Ort::Session& session() noexcept { return session_; }

    // Pretty-prints model metadata — called once on server startup.
    void log_metadata() const;

    static std::string dtype_name(ONNXTensorElementDataType t);

private:
    Ort::Env     env_;
    Ort::Session session_;
    std::string  name_;
    std::uintmax_t file_bytes_{0};
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;

    void read_tensor_info();
};

} // namespace inference

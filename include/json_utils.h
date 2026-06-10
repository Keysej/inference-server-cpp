#pragma once

// JSON ↔ Tensor conversion lives here so main.cpp stays focused on routing.
// This is the system boundary: unvalidated JSON comes in, typed tensors go out.
// All validation happens here — the InferenceEngine never sees raw JSON.

#include <string_view>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "tensor.h"

namespace inference {

// Parse {"data": [1.0, 2.0, ...], "shape": [1, 4]} → FloatTensor.
// Throws std::invalid_argument on missing fields or type errors.
// Throws std::invalid_argument (re-thrown from Tensor ctor) on shape mismatch.
inline FloatTensor tensor_from_json(const nlohmann::json& j) {
    if (!j.contains("data") || !j.contains("shape"))
        throw std::invalid_argument("request body must contain 'data' and 'shape'");

    if (!j["data"].is_array())
        throw std::invalid_argument("'data' must be a JSON array");

    if (!j["shape"].is_array())
        throw std::invalid_argument("'shape' must be a JSON array");

    auto data  = j["data"].get<std::vector<float>>();
    auto shape = j["shape"].get<std::vector<int64_t>>();

    // Tensor ctor validates that product(shape) == data.size()
    return FloatTensor(std::move(shape), std::move(data));
}

// FloatTensor → {"data": [...], "shape": [...], "model": "..."}
// model_name is optional; omitted from output if empty.
inline nlohmann::json tensor_to_json(const FloatTensor& t,
                                     std::string_view model_name = "") {
    // std::span lets us construct the JSON array without an extra copy.
    auto view = t.view();
    nlohmann::json j;
    j["data"]  = std::vector<float>(view.begin(), view.end());
    j["shape"] = t.shape();
    if (!model_name.empty())
        j["model"] = model_name;
    return j;
}

// Convenience: build a JSON error body.
inline nlohmann::json error_json(std::string_view msg) {
    return nlohmann::json{{"error", msg}};
}

} // namespace inference

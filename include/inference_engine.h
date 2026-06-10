#pragma once

#include <vector>
#include <string>
#include <stdexcept>

#include "model_loader.h"
#include "tensor.h"

namespace inference {

// Raised when the caller passes a tensor whose shape is incompatible with the
// model's declared input — caught and surfaced as a 400 by the HTTP layer.
struct ShapeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// InferenceEngine wraps a loaded ORT session and runs synchronous forward passes.
//
// Single-responsibility: this class only knows how to call session.Run().
// It holds no HTTP state and no JSON logic — those live in the HTTP layer (Step 5).
// Step 6 will add a coroutine wrapper around run() for concurrent request handling.
class InferenceEngine {
public:
    // Takes ownership of the loader so the session lifetime is tied to the engine.
    explicit InferenceEngine(ModelLoader loader);

    // Non-copyable — session state is not safely shareable.
    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&)                 = default;
    InferenceEngine& operator=(InferenceEngine&&)      = default;

    // Synchronous inference.  Returns a FloatTensor for every output node.
    // Throws ShapeError if the input shape is incompatible with the model.
    //
    // Step 6 wraps this in a coroutine so the calling thread is never blocked
    // while other requests proceed concurrently.
    [[nodiscard]] std::vector<FloatTensor> run(FloatTensor input);

    // Convenience: single-output models (the common case).
    [[nodiscard]] FloatTensor run_single(FloatTensor input);

    [[nodiscard]] const ModelLoader& model() const noexcept { return loader_; }

private:
    ModelLoader loader_;

    // Cached name pointers — avoids a string lookup on every request.
    // These point into the TensorInfo strings owned by loader_, so they
    // stay valid as long as loader_ is alive.
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    void build_name_cache();
    void validate_input(const FloatTensor& t) const;
};

} // namespace inference

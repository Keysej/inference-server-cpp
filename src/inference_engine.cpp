#include "inference_engine.h"

#include <sstream>
#include <ranges>   // std::views::transform to build name-pointer vectors

namespace inference {

// ── Construction ──────────────────────────────────────────────────────────────

InferenceEngine::InferenceEngine(ModelLoader loader)
    : loader_(std::move(loader))
{
    build_name_cache();
}

// Cache const char* pointers into the TensorInfo name strings.
// We use a C++20 range transform instead of a raw loop to express intent
// clearly: map TensorInfo → const char* with no temporary containers.
void InferenceEngine::build_name_cache() {
    auto to_cstr = [](const TensorInfo& t) { return t.name.c_str(); };

    auto in_view  = loader_.inputs()  | std::views::transform(to_cstr);
    auto out_view = loader_.outputs() | std::views::transform(to_cstr);

    input_names_.assign(in_view.begin(),  in_view.end());
    output_names_.assign(out_view.begin(), out_view.end());
}

// ── Input validation ──────────────────────────────────────────────────────────

void InferenceEngine::validate_input(const FloatTensor& t) const {
    if (loader_.inputs().empty())
        throw ShapeError("model has no declared inputs");

    const auto& expected = loader_.inputs()[0];

    // Rank check — mismatched number of dimensions is always wrong.
    if (t.shape().size() != expected.shape.size()) {
        std::ostringstream ss;
        ss << "input rank mismatch: got " << t.shape().size()
           << " dims, model expects " << expected.shape.size();
        throw ShapeError(ss.str());
    }

    // Dimension check — skip dynamic dims (== -1), validate static ones.
    for (size_t i = 0; i < expected.shape.size(); ++i) {
        if (expected.shape[i] < 0) continue;   // dynamic: any size is valid
        if (t.shape()[i] != expected.shape[i]) {
            std::ostringstream ss;
            ss << "input dim[" << i << "] mismatch: got " << t.shape()[i]
               << ", model expects " << expected.shape[i];
            throw ShapeError(ss.str());
        }
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::vector<FloatTensor> InferenceEngine::run(FloatTensor input) {
    validate_input(input);

    // Wrap the input buffer in an Ort::Value without copying it.
    // The vector keeps it alive for the duration of session.Run().
    Ort::Value ort_input = to_ort_value(input);

    // session.Run() is synchronous and thread-safe for concurrent calls on the
    // same session (ORT uses internal locking).  Step 6 adds a coroutine wrapper
    // so the calling thread yields while waiting, allowing other requests to run.
    auto ort_outputs = loader_.session().Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &ort_input,
        input_names_.size(),
        output_names_.data(),
        output_names_.size()
    );

    // Materialise each output into an owning FloatTensor.
    std::vector<FloatTensor> results;
    results.reserve(ort_outputs.size());
    for (auto& ov : ort_outputs)
        results.push_back(from_ort_value<float>(ov));

    return results;
}

FloatTensor InferenceEngine::run_single(FloatTensor input) {
    auto outputs = run(std::move(input));
    if (outputs.empty())
        throw ShapeError("model produced no outputs");
    return std::move(outputs[0]);
}

} // namespace inference

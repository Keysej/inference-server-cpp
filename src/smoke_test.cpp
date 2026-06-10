// smoke_test.cpp — validates C++20 features and the ORT tensor bridge.
// No HTTP server; runs in under a second, no model file needed.

#include <iostream>
#include <vector>
#include <ranges>
#include <span>
#include <concepts>
#include <string_view>
#include <stdexcept>
#include <cstdlib>

#include "tensor.h"
#include "model_loader.h"

using namespace inference;

// CHECK replaces assert: works in Release builds (where NDEBUG silences assert).
#define CHECK(expr) \
    do { if (!(expr)) { \
        std::cerr << "FAIL  " << #expr << "  (" << __FILE__ << ':' << __LINE__ << ")\n"; \
        std::exit(1); \
    } } while(0)

static void pass(std::string_view label) {
    std::cout << "  [OK] " << label << '\n';
}

// ── concept enforcement ───────────────────────────────────────────────────────

void test_concepts() {
    static_assert(TensorElement<float>);
    static_assert(TensorElement<double>);
    static_assert(TensorElement<int32_t>);
    static_assert(TensorElement<int64_t>);
    static_assert(TensorElement<uint8_t>);
    // Uncomment to prove the concept rejects disallowed types at compile time:
    // static_assert(TensorElement<std::string>);
    pass("TensorElement concept enforces allowed types at compile time");
}

// ── compile-time ORT dtype mapping ───────────────────────────────────────────

void test_ort_type_traits() {
    // else-if constexpr chain means these resolve at compile time with no
    // runtime branching — the compiler sees a single constant per instantiation.
    static_assert(ort_element_type<float>()   == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    static_assert(ort_element_type<double>()  == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    static_assert(ort_element_type<int32_t>() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    static_assert(ort_element_type<int64_t>() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    static_assert(ort_element_type<uint8_t>() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    pass("ort_element_type<T>() resolves at compile time via else-if constexpr");
}

// ── Tensor<T> basics ─────────────────────────────────────────────────────────

void test_tensor() {
    FloatTensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(t.element_count() == 6);

    // std::span: zero-copy view — pointer + length, no memcpy.
    std::span<const float> v = t.view();
    CHECK(v[0] == 1.f);
    CHECK(v[5] == 6.f);

    // Ranges pipeline: filter > 3, accumulate.
    float sum = 0;
    for (float x : v | std::views::filter([](float f){ return f > 3.f; }))
        sum += x;
    CHECK(sum == 4.f + 5.f + 6.f);

    // zeroed factory
    auto z = FloatTensor::zeroed({3, 3});
    CHECK(z.element_count() == 9);
    CHECK(z.view()[4] == 0.f);

    pass("Tensor<T>: shape, element_count, span view, ranges filter, zeroed factory");
}

// ── Shape/data mismatch guard ─────────────────────────────────────────────────

void test_shape_mismatch() {
    bool threw = false;
    try { FloatTensor bad({2, 3}, {1.f, 2.f}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    pass("Tensor<T> throws on shape/data size mismatch");
}

// ── ORT bridge round-trip ─────────────────────────────────────────────────────

void test_ort_bridge() {
    FloatTensor src({1, 4}, {10.f, 20.f, 30.f, 40.f});
    {
        // to_ort_value borrows src's buffer — no copy.
        Ort::Value ov = to_ort_value(src);

        auto si    = ov.GetTensorTypeAndShapeInfo();
        auto shape = si.GetShape();

        CHECK(si.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        CHECK(shape.size() == 2);
        CHECK(shape[0] == 1 && shape[1] == 4);

        // from_ort_value copies into an owning Tensor.
        FloatTensor dst = from_ort_value<float>(ov);
        CHECK(dst.element_count() == 4);
        CHECK(dst.view()[0] == 10.f);
        CHECK(dst.view()[3] == 40.f);
    }
    // src still owns its memory — the borrow (ov) is gone.
    CHECK(src.view()[0] == 10.f);
    pass("to_ort_value / from_ort_value: zero-copy borrow + owning round-trip");
}

// ── string_view zero-copy label handling ─────────────────────────────────────

void test_string_view() {
    // string_view is a non-owning pointer into an existing buffer — no heap.
    std::string_view label = "input:0";
    CHECK(label.starts_with("input"));   // starts_with: C++20 addition to string_view
    CHECK(label.size() == 7);
    pass("string_view: zero-copy label, starts_with (C++20 API)");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n=== ai-inference-server smoke test (Step 3) ===\n\n";
    test_concepts();
    test_ort_type_traits();
    test_tensor();
    test_shape_mismatch();
    test_ort_bridge();
    test_string_view();
    std::cout << "\n[PASS] All checks passed — tensor utilities and ORT bridge ready.\n\n";
    return 0;
}

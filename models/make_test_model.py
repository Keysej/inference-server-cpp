"""
Generates a tiny ONNX model for local testing:
  input:  float32[batch, 4]   (e.g. 4-feature input like Iris)
  output: float32[batch, 3]   (3-class logit scores)

Architecture: Linear(4→8) → ReLU → Linear(8→3)
No training — random weights are fine for inference plumbing tests.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(42)

# ── Weights (random, fixed seed for reproducibility) ─────────────────────────
W1 = rng.standard_normal((8, 4)).astype(np.float32)   # (out, in)
b1 = rng.standard_normal((8,)).astype(np.float32)
W2 = rng.standard_normal((3, 8)).astype(np.float32)
b2 = rng.standard_normal((3,)).astype(np.float32)

# ── Initializers (constants embedded in the graph) ───────────────────────────
init_W1 = numpy_helper.from_array(W1, name="W1")
init_b1 = numpy_helper.from_array(b1, name="b1")
init_W2 = numpy_helper.from_array(W2, name="W2")
init_b2 = numpy_helper.from_array(b2, name="b2")

# ── Graph nodes ───────────────────────────────────────────────────────────────
# Gemm: Y = alpha * A * B^T + beta * C  →  linear layer
gemm1 = helper.make_node("Gemm", ["input", "W1", "b1"], ["relu_in"],
                          transB=1, alpha=1.0, beta=1.0)
relu  = helper.make_node("Relu", ["relu_in"], ["relu_out"])
gemm2 = helper.make_node("Gemm", ["relu_out", "W2", "b2"], ["output"],
                          transB=1, alpha=1.0, beta=1.0)

# ── I/O tensor descriptors ────────────────────────────────────────────────────
# batch dimension is dynamic (-1) to stress-test the shape printer.
input_tensor  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [-1, 4])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [-1, 3])

# ── Assemble graph and model ──────────────────────────────────────────────────
graph = helper.make_graph(
    [gemm1, relu, gemm2],
    "tiny_classifier",
    [input_tensor],
    [output_tensor],
    initializer=[init_W1, init_b1, init_W2, init_b2],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.doc_string = "Tiny 2-layer classifier for inference server testing"
onnx.checker.check_model(model)

out_path = "tiny_classifier.onnx"
onnx.save(model, out_path)
print(f"[OK] Saved {out_path}")
print(f"     input  : float32[-1, 4]")
print(f"     output : float32[-1, 3]")
print(f"     ops    : Gemm → Relu → Gemm")

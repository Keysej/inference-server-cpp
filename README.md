# AI Inference Server · C++20

A high-performance HTTP inference server built from scratch in **C++20** using ONNX Runtime and Crow. Designed as a portfolio project targeting AI/ML backend engineering roles.

**~132 000 req/s · p50 < 0.05 ms · p99 < 0.2 ms** (Apple M-series, tiny model, loopback)

---

## C++20 features used

| Feature | Where |
|---|---|
| **Coroutines** (`co_await`, `co_return`, `promise_type`) | [`include/task.h`](include/task.h), [`include/async_engine.h`](include/async_engine.h) |
| **`std::jthread` + `std::stop_token`** | [`include/thread_pool.h`](include/thread_pool.h) |
| **`condition_variable_any::wait(lock, stop_token, pred)`** | [`include/thread_pool.h`](include/thread_pool.h) |
| **Concepts** (`TensorElement`) | [`include/tensor.h`](include/tensor.h) |
| **`std::views::iota`** | [`src/main.cpp`](src/main.cpp) — benchmark loop |
| **`std::ranges::sort`** | [`src/main.cpp`](src/main.cpp) — latency percentiles |
| **`constexpr std::string_view`** | [`src/main.cpp`](src/main.cpp) — zero-cost ANSI color constants |
| **`std::span`** | [`include/tensor.h`](include/tensor.h) — zero-copy data view |

---

## Architecture

```
HTTP Client
    │
    ▼
┌──────────────────────────────────────┐
│         Crow HTTP Server             │  multithreaded · port 8080
│  POST /predict                       │
│  GET  /health                        │
│  GET  /benchmark                     │
└───────────────┬──────────────────────┘
                │  co_await PoolAwaitable
                ▼
┌──────────────────────────────────────┐
│  AsyncInferenceEngine (coroutine)    │  Task<FloatTensor>
│  run_async() suspends caller,        │
│  posts work to pool, resumes on done │
└───────────────┬──────────────────────┘
                │  submit_with_future / submit
                ▼
┌──────────────────────────────────────┐
│  ThreadPool  (N × std::jthread)      │  N = hardware_concurrency()
│  stop_token cooperative shutdown     │
│  condition_variable_any wakeup       │
└───────────────┬──────────────────────┘
                │  session.Run()
                ▼
┌──────────────────────────────────────┐
│  ONNX Runtime  (ORT_ENABLE_ALL)      │  thread-safe inference
└──────────────────────────────────────┘
```

**Key design point:** Crow threads and inference threads are fully decoupled. A Crow thread offloads `session.Run()` to the inference pool via a C++20 coroutine, then blocks only on a lightweight mutex/CV wakeup — it is never inside ORT while waiting. Under load, Crow threads overlap network I/O with inference.

---

## Prerequisites

**macOS (Homebrew — tested on Apple Silicon)**

```bash
brew install onnxruntime crow nlohmann-json asio cmake
```

**Python** (to generate the test model)

```bash
pip install onnx numpy
```

> **Linux:** replace `HOMEBREW_PREFIX` with your sysroot, e.g.  
> `cmake -DHOMEBREW_PREFIX=/usr ...` after installing the same deps via vcpkg or your package manager.

---

## Build

```bash
git clone https://github.com/Keysej/inference-server-cpp.git
cd inference-server-cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Two binaries are produced:

| Binary | Purpose |
|---|---|
| `build/inference_server` | Main HTTP server |
| `build/smoke_test` | Headless tensor + engine unit test (no model file needed) |

---

## Quick start

```bash
# 1. Generate the test ONNX model (requires: pip install onnx numpy)
cd models && python3 make_test_model.py && cd ..

# 2. Run the server
./build/inference_server models/tiny_classifier.onnx 8080
```

Or run the all-in-one demo script:

```bash
chmod +x demo.sh && ./demo.sh
```

---

## API

### `GET /health`

```bash
curl http://localhost:8080/health
```
```json
{"model": "tiny_classifier", "status": "ok"}
```

---

### `POST /predict`

Accepts a flat float32 array + shape. Shape must match the model's expected input (`[-1, 4]` for the test model).

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0, 4.0], "shape": [1, 4]}'
```
```json
{
  "data": [-0.527, 1.203, -0.441],
  "model": "tiny_classifier",
  "shape": [1, 3]
}
```

---

### `GET /benchmark`

Fires `n` concurrent inference requests through the thread pool, measures per-request latency, and returns statistics. `n` defaults to 200; max 10 000.

```bash
curl "http://localhost:8080/benchmark?n=500"
```
```json
{
  "model": "tiny_classifier",
  "requests": 500,
  "wall_ms": 3.8,
  "rps": 131578,
  "latency_ms": {
    "mean": 0.044,
    "min":  0.031,
    "p50":  0.048,
    "p95":  0.136,
    "p99":  0.171,
    "max":  0.392
  }
}
```

---

## Project structure

```
.
├── CMakeLists.txt
├── include/
│   ├── tensor.h          # Tensor<T> type + TensorElement concept
│   ├── model_loader.h    # ONNX session + metadata
│   ├── inference_engine.h
│   ├── json_utils.h      # system boundary: JSON ↔ Tensor
│   ├── task.h            # Task<T> coroutine return type
│   ├── thread_pool.h     # fixed-size pool (jthread + stop_token)
│   └── async_engine.h    # PoolAwaitable + AsyncInferenceEngine
├── src/
│   ├── main.cpp          # Crow routes + startup banner
│   ├── model_loader.cpp
│   ├── inference_engine.cpp
│   └── smoke_test.cpp    # headless tensor test
└── models/
    └── make_test_model.py  # generates tiny_classifier.onnx
```

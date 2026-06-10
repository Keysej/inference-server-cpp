cmake_minimum_required(VERSION 3.22)
project(ai_inference_server VERSION 0.1.0 LANGUAGES CXX)

# Require C++20 — we use coroutines, concepts, ranges, jthread, span
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Homebrew prefix (Apple Silicon default; override via -DHOMEBREW_PREFIX=…) ──
if(NOT DEFINED HOMEBREW_PREFIX)
  set(HOMEBREW_PREFIX "/opt/homebrew")
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${HOMEBREW_PREFIX}/opt/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include/onnxruntime")

# nlohmann/json (header-only)
set(JSON_INCLUDE "${HOMEBREW_PREFIX}/opt/nlohmann-json/include")

# Crow (header-only HTTP framework)
set(CROW_INCLUDE "${HOMEBREW_PREFIX}/opt/crow/include")

# Asio (standalone — required by Crow, ships with boost but also available standalone)
# Crow bundles it; add homebrew asio as fallback
find_path(ASIO_INCLUDE asio.hpp
  PATHS
    "${HOMEBREW_PREFIX}/opt/asio/include"
    "${HOMEBREW_PREFIX}/include"
    "${CROW_INCLUDE}"
)

# ── Shared interface library for common flags ─────────────────────────────────
add_library(project_options INTERFACE)
target_compile_options(project_options INTERFACE
  -Wall -Wextra -Wpedantic
  -O2
)
target_include_directories(project_options INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE}
  ${JSON_INCLUDE}
  ${CROW_INCLUDE}
  ${ASIO_INCLUDE}
)
target_link_libraries(project_options INTERFACE ${ONNXRUNTIME_LIB})

# ── Main server executable ────────────────────────────────────────────────────
add_executable(inference_server
  src/main.cpp
  src/model_loader.cpp
  src/inference_engine.cpp
)
target_link_libraries(inference_server PRIVATE project_options)

# ── Smoke-test executable (no HTTP, no model file needed) ─────────────────────
add_executable(smoke_test
  src/smoke_test.cpp
)
target_link_libraries(smoke_test PRIVATE project_options)

# ── Print where libs were found ───────────────────────────────────────────────
message(STATUS "ONNX Runtime lib : ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime inc : ${ONNXRUNTIME_INCLUDE}")
message(STATUS "nlohmann/json    : ${JSON_INCLUDE}")
message(STATUS "Crow             : ${CROW_INCLUDE}")

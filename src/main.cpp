#include <iostream>
#include <string>
#include <vector>
#include <future>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <ranges>
#include <cmath>

#define CROW_MAIN
#include <crow.h>
#include <nlohmann/json.hpp>

#include "model_loader.h"
#include "inference_engine.h"
#include "async_engine.h"
#include "json_utils.h"

using namespace inference;
using json = nlohmann::json;

// ── ANSI color helpers ────────────────────────────────────────────────────────
// constexpr string_view: zero runtime cost, zero heap allocation (C++20).

namespace ansi {
    constexpr std::string_view rst  = "\033[0m";
    constexpr std::string_view bold = "\033[1m";
    constexpr std::string_view grn  = "\033[1;32m";  // bold green
    constexpr std::string_view cyn  = "\033[1;36m";  // bold cyan
}

// ── Response helpers ──────────────────────────────────────────────────────────
// Crow's response(int, string) sets text/plain.  We always want application/json.

static crow::response json_ok(const json& body) {
    crow::response r{200, body.dump()};
    r.set_header("Content-Type", "application/json");
    return r;
}

static crow::response json_err(int code, std::string_view msg) {
    crow::response r{code, error_json(msg).dump()};
    r.set_header("Content-Type", "application/json");
    return r;
}

// ── Startup banner ────────────────────────────────────────────────────────────

static void print_banner(const InferenceEngine& engine, int port, size_t workers) {
    const auto& m = engine.model();

    // Human-readable byte count.
    auto size_str = [](std::uintmax_t b) -> std::string {
        if (b < 1024)        return std::to_string(b) + " B";
        if (b < 1024 * 1024) return std::to_string(b / 1024) + " KB";
        return std::to_string(b / (1024 * 1024)) + " MB";
    };

    // Shape with "batch" for dynamic (-1) dims: [-1, 4] → "[batch × 4]".
    auto shape_str = [](const std::vector<int64_t>& s) -> std::string {
        std::string out = "[";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) out += " × ";
            out += (s[i] < 0) ? "batch" : std::to_string(s[i]);
        }
        return out + "]";
    };

    const std::string_view HR  = "├────────────────────────────────────────────────\n";
    const std::string_view TOP = "┌────────────────────────────────────────────────\n";
    const std::string_view BOT = "└────────────────────────────────────────────────\n";

    using std::cout;
    cout << '\n' << TOP;
    cout << "│  " << ansi::cyn << "AI Inference Server" << ansi::rst
         << "  (C++20 · ONNX Runtime)\n";
    cout << HR;
    cout << "│  " << ansi::cyn << "model   " << ansi::rst
         << ansi::bold << m.model_name() << ansi::rst
         << "   ·   " << size_str(m.model_file_bytes()) << '\n';
    cout << "│  " << ansi::cyn << "session " << ansi::rst
         << "ORT_ENABLE_ALL  ·  1 intra-op thread\n";
    cout << "│  " << ansi::cyn << "workers " << ansi::rst
         << workers << " inference threads\n";
    cout << HR;
    for (const auto& t : m.inputs()) {
        cout << "│  " << ansi::cyn << "in  " << ansi::rst
             << ansi::bold << t.name << ansi::rst
             << "   " << ModelLoader::dtype_name(t.dtype)
             << "  " << shape_str(t.shape) << '\n';
    }
    for (const auto& t : m.outputs()) {
        cout << "│  " << ansi::cyn << "out " << ansi::rst
             << ansi::bold << t.name << ansi::rst
             << "   " << ModelLoader::dtype_name(t.dtype)
             << "  " << shape_str(t.shape) << '\n';
    }
    cout << HR;
    cout << "│  " << ansi::bold << "POST" << ansi::rst << "  /predict    →  run inference\n";
    cout << "│  " << ansi::bold << "GET " << ansi::rst << "  /health     →  liveness check\n";
    cout << "│  " << ansi::bold << "GET " << ansi::rst << "  /benchmark  →  latency profile  (?n=N)\n";
    cout << HR;
    cout << "│  " << ansi::grn << "✓" << ansi::rst
         << "  http://0.0.0.0:" << port << "  ready\n";
    cout << BOT << '\n' << std::flush;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: inference_server <model.onnx> [port]\n";
        return 1;
    }

    const int port = (argc > 2) ? std::stoi(argv[2]) : 8080;

    // ── Load model & build engine ─────────────────────────────────────────────
    ModelLoader  loader{argv[1]};
    std::string  model_name = loader.model_name();
    InferenceEngine engine{std::move(loader)};

    // ── Async coroutine layer ─────────────────────────────────────────────────
    const size_t pool_threads = std::thread::hardware_concurrency();
    ThreadPool           pool{pool_threads};
    AsyncInferenceEngine async_engine{engine, pool};

    // ── Wire up Crow ──────────────────────────────────────────────────────────
    crow::SimpleApp app;
    app.loglevel(crow::LogLevel::Warning);

    // ── GET /health ───────────────────────────────────────────────────────────
    CROW_ROUTE(app, "/health")
    ([&model_name]() {
        return json_ok({{"status", "ok"}, {"model", model_name}});
    });

    // ── GET /benchmark ────────────────────────────────────────────────────────
    // Query params: ?n=200   (number of requests; default 200, max 10 000)
    //
    // Fires n inference requests concurrently via the inference thread pool,
    // collects per-request latencies, and returns p50/p95/p99 plus RPS.
    //
    // C++20 highlights used here:
    //   std::views::iota   — range of ints without a raw loop index
    //   std::ranges::sort  — ADL-safe sort that works directly on a container
    CROW_ROUTE(app, "/benchmark")
    ([&engine, &pool, &model_name](const crow::request& req) -> crow::response {
        try {
            const char* n_param = req.url_params.get("n");
            int n = n_param ? std::clamp(std::stoi(n_param), 1, 10'000) : 200;

            // Fire n inference requests concurrently — all land in the pool queue
            // immediately, so the pool's workers pick them up in parallel.
            std::vector<std::future<double>> futures;
            futures.reserve(n);

            auto wall_start = std::chrono::steady_clock::now();

            for (auto _ : std::views::iota(0, n)) {
                futures.push_back(pool.submit_with_future([&engine] {
                    FloatTensor input{{1, 4}, {1.0f, 2.0f, 3.0f, 4.0f}};
                    auto t0 = std::chrono::high_resolution_clock::now();
                    (void)engine.run_single(std::move(input));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double, std::milli>(t1 - t0).count();
                }));
            }

            std::vector<double> latencies;
            latencies.reserve(n);
            for (auto& f : futures)
                latencies.push_back(f.get());   // throws if inference threw

            auto wall_end = std::chrono::steady_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(
                                 wall_end - wall_start).count();

            // std::ranges::sort: C++20 — no need to pass begin/end iterators.
            std::ranges::sort(latencies);

            double sum  = std::accumulate(latencies.begin(), latencies.end(), 0.0);
            double mean = sum / n;

            // Nearest-rank percentile (index clamped to [0, n-1]).
            auto pct = [&](double p) {
                size_t idx = static_cast<size_t>(std::floor(p * n));
                return latencies[std::min(idx, latencies.size() - 1)];
            };

            double rps = wall_ms > 0.0 ? (n * 1000.0 / wall_ms) : 0.0;

            return json_ok({
                {"model",    model_name},
                {"requests", n},
                {"wall_ms",  std::round(wall_ms * 10) / 10},
                {"rps",      std::round(rps)},
                {"latency_ms", {
                    {"mean", std::round(mean    * 1000) / 1000},
                    {"min",  std::round(latencies.front() * 1000) / 1000},
                    {"p50",  std::round(pct(0.50) * 1000) / 1000},
                    {"p95",  std::round(pct(0.95) * 1000) / 1000},
                    {"p99",  std::round(pct(0.99) * 1000) / 1000},
                    {"max",  std::round(latencies.back()  * 1000) / 1000},
                }}
            });

        } catch (const std::exception& e) {
            return json_err(500, e.what());
        }
    });

    // ── POST /predict ─────────────────────────────────────────────────────────
    // Step 6: coroutine-based handler.
    //
    // run_async() suspends at co_await and offloads session.Run() to the
    // inference thread pool.  sync_wait() blocks this Crow thread until the
    // coroutine completes — but because inference runs on a separate pool, the
    // Crow thread count and the inference concurrency are independently tuned.
    //
    // In a fully async HTTP stack (Seastar, io_uring) the handler itself would
    // be a coroutine and would co_await run_async() without blocking any thread.
    CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::Post)
    ([&async_engine, &model_name](const crow::request& req) -> crow::response {
        try {
            auto body   = json::parse(req.body);
            auto input  = tensor_from_json(body);
            auto output = async_engine.run_async(std::move(input)).sync_wait();
            return json_ok(tensor_to_json(output, model_name));

        } catch (const json::parse_error& e) {
            return json_err(400, std::string("JSON parse error: ") + e.what());
        } catch (const ShapeError& e) {
            return json_err(400, e.what());
        } catch (const std::invalid_argument& e) {
            return json_err(400, e.what());
        } catch (const std::exception& e) {
            return json_err(500, e.what());
        }
    });

    // ── Start server ──────────────────────────────────────────────────────────
    print_banner(engine, port, pool_threads);

    app.port(port).multithreaded().run();
    return 0;
}

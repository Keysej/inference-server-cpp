#pragma once

// AsyncInferenceEngine — coroutine wrapper around InferenceEngine::run_single.
//
// Architecture:
//   Crow thread (synchronous)
//     └─ calls async_engine.run_async(input)    ← returns Task<FloatTensor>
//         └─ coroutine hits co_await PoolAwaitable{...}
//             └─ suspends caller, posts session.Run() to ThreadPool
//                 └─ pool worker runs blocking ORT call
//                 └─ resumes coroutine with result
//     └─ calls task.sync_wait()                 ← blocks until coroutine done
//
// The Crow thread pool and the inference thread pool are decoupled: Crow
// threads are never inside session.Run() — they block only on a fast mutex/CV
// wakeup.  Under load, Crow threads overlap network I/O with inference.

#include <coroutine>
#include <functional>
#include <optional>
#include <exception>

#include "inference_engine.h"
#include "task.h"
#include "thread_pool.h"

namespace inference {

// PoolAwaitable: the suspension point that offloads work to a thread pool.
//
// C++20 guarantees the coroutine is SUSPENDED before await_suspend() is called,
// so continuation.resume() from a pool thread is always safe — even if the work
// completes before await_suspend() returns on the calling thread.
struct PoolAwaitable {
    ThreadPool&                    pool;
    std::function<FloatTensor()>   work;

    std::optional<FloatTensor> result_{};
    std::exception_ptr         exception_{};
    std::coroutine_handle<>    continuation_{};

    // Never skip suspension — the whole point is to yield the caller.
    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> h) {
        continuation_ = h;
        pool.submit([this] {
            try   { result_    = work(); }
            catch (...) { exception_ = std::current_exception(); }
            // Resume the Task coroutine on this worker thread.
            // The coroutine will then execute co_return and notify sync_wait().
            continuation_.resume();
        });
    }

    FloatTensor await_resume() {
        if (exception_) std::rethrow_exception(exception_);
        return std::move(*result_);
    }
};

// Wraps InferenceEngine with a coroutine API.
//
// Borrows both engine and pool — both must outlive this object.
class AsyncInferenceEngine {
public:
    AsyncInferenceEngine(InferenceEngine& engine, ThreadPool& pool) noexcept
        : engine_(engine), pool_(pool) {}

    // Returns a Task<FloatTensor>.  Calling sync_wait() on it blocks until
    // the coroutine completes.  In a fully async HTTP stack you would
    // co_await it from the handler instead.
    [[nodiscard]] Task<FloatTensor> run_async(FloatTensor input) {
        // co_await suspends this coroutine and posts session.Run() to the pool.
        // When the pool worker finishes, it resumes us here with the result.
        auto result = co_await PoolAwaitable{
            pool_,
            [this, t = std::move(input)]() mutable {
                return engine_.run_single(std::move(t));
            }
        };
        co_return result;
    }

private:
    InferenceEngine& engine_;
    ThreadPool&      pool_;
};

} // namespace inference

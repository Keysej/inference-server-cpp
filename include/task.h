#pragma once

// Task<T>: minimal C++20 coroutine return type.
//
// A Task<T> represents an asynchronous computation that will eventually
// produce a value of type T (or throw).  The coroutine starts eagerly
// (initial_suspend: never) and its frame is kept alive after completion
// (final_suspend: always) so the caller can read the result via sync_wait().
//
// Interview context: this is the same foundation used by every async C++
// library (cppcoro, libunifex, folly::coro).  They add scheduling,
// cancellation, and executor propagation on top of exactly these ~40 lines.

#include <coroutine>
#include <exception>
#include <optional>
#include <mutex>
#include <condition_variable>
#include <utility>

namespace inference {

template<typename T>
class Task {
public:
    // ── promise_type ───────────────────────────────────────────────────────
    // The compiler instantiates one promise_type per coroutine invocation.
    // It stores the result and synchronises the coroutine body with the
    // caller waiting in sync_wait().
    struct promise_type {
        std::optional<T>        result_;
        std::exception_ptr      exception_;
        std::mutex              mtx_;
        std::condition_variable cv_;
        bool                    done_{false};

        Task get_return_object() noexcept {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        // Eager start: coroutine body runs immediately on first call.
        std::suspend_never  initial_suspend() noexcept { return {}; }

        // Keep frame alive after co_return so sync_wait() can read the result.
        std::suspend_always final_suspend()   noexcept { return {}; }

        void return_value(T value) {
            std::unique_lock lock{mtx_};
            result_ = std::move(value);
            done_   = true;
            cv_.notify_all();
        }

        void unhandled_exception() {
            std::unique_lock lock{mtx_};
            exception_ = std::current_exception();
            done_      = true;
            cv_.notify_all();
        }
    };

    using Handle = std::coroutine_handle<promise_type>;

    explicit Task(Handle h) noexcept : handle_(h) {}
    Task(const Task&) = delete;
    Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    ~Task() { if (handle_) handle_.destroy(); }

    // Block the calling thread until the coroutine completes, then return
    // the value (or rethrow the exception).
    //
    // Used here because Crow's HTTP layer is synchronous.  In a fully async
    // HTTP stack (e.g. Seastar, libuv, io_uring) the handler would itself be
    // a coroutine and would co_await this Task instead of calling sync_wait().
    T sync_wait() {
        auto& p = handle_.promise();
        std::unique_lock lock{p.mtx_};
        p.cv_.wait(lock, [&p] { return p.done_; });
        if (p.exception_) std::rethrow_exception(p.exception_);
        return std::move(*p.result_);
    }

private:
    Handle handle_;
};

} // namespace inference

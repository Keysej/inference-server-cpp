#pragma once

// Fixed-size thread pool using two C++20 additions:
//
//   std::jthread     — auto-joins on destruction; owns a built-in stop_source
//   stop_token       — cooperative cancellation without a shared flag variable
//   condition_variable_any::wait(lock, stop_token, pred)
//                    — registers a stop callback that calls notify_all() when
//                      stop is requested, so workers wake without a manual kick

#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stop_token>
#include <thread>

namespace inference {

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency()) {
        workers_.reserve(thread_count);
        for (size_t i = 0; i < thread_count; ++i)
            workers_.emplace_back([this](std::stop_token st) { worker_loop(st); });
    }

    // Post a callable; it runs on one worker thread.
    template<typename F>
    void submit(F&& f) {
        {
            std::unique_lock lock{mtx_};
            queue_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

    // Post a callable and return a std::future for its result.
    // packaged_task is non-copyable, so we share_ptr it to satisfy
    // std::function's copyability requirement.
    template<typename F>
    [[nodiscard]] auto submit_with_future(F&& f) {
        using R   = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        submit([task = std::move(task)] { (*task)(); });
        return fut;
    }

    // std::jthread destructors call request_stop() + join().
    // condition_variable_any registers a stop callback, so sleeping workers
    // wake automatically — no manual notify needed here.
    ~ThreadPool() = default;

private:
    void worker_loop(std::stop_token st) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock{mtx_};
                // Returns false when stop requested and queue is still empty.
                if (!cv_.wait(lock, st, [this] { return !queue_.empty(); }))
                    return;
                task = std::move(queue_.front());
                queue_.pop();
            }
            task();
        }
    }

    std::vector<std::jthread>           workers_;
    std::queue<std::function<void()>>   queue_;
    std::mutex                          mtx_;
    std::condition_variable_any         cv_;
};

} // namespace inference

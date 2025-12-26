// Phase 39: Performance Profiling
// Profiling infrastructure for performance analysis

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace lightwatch {

// High-resolution timer
class Timer {
public:
    Timer() : running_(false) {}

    void start() {
        start_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    void stop() {
        end_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }

    double elapsed_ms() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_;
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_us() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_;
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

    double elapsed_s() const {
        auto end = running_ ? std::chrono::high_resolution_clock::now() : end_;
        return std::chrono::duration<double>(end - start_).count();
    }

    bool is_running() const { return running_; }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
    bool running_;
};

// Profiling statistics
struct ProfileStats {
    size_t count = 0;
    double total_ms = 0.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double std_ms = 0.0;
};

// Global profiler singleton
class Profiler {
public:
    // Singleton access
    static Profiler& instance() {
        static Profiler profiler;
        return profiler;
    }

    // Start timing a named scope
    void start_scope(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        active_timers_[name].start();
    }

    // End timing a named scope
    void end_scope(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = active_timers_.find(name);
        if (it != active_timers_.end()) {
            it->second.stop();
            timings_[name].push_back(it->second.elapsed_ms());
            active_timers_.erase(it);
        }
    }

    // Get statistics for a named scope
    ProfileStats get_stats(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ProfileStats stats;

        auto it = timings_.find(name);
        if (it == timings_.end() || it->second.empty()) {
            return stats;
        }

        const auto& times = it->second;
        stats.count = times.size();

        stats.min_ms = times[0];
        stats.max_ms = times[0];
        stats.total_ms = 0.0;

        for (double t : times) {
            stats.total_ms += t;
            stats.min_ms = std::min(stats.min_ms, t);
            stats.max_ms = std::max(stats.max_ms, t);
        }

        stats.mean_ms = stats.total_ms / static_cast<double>(stats.count);

        // Compute standard deviation
        double sum_sq = 0.0;
        for (double t : times) {
            double diff = t - stats.mean_ms;
            sum_sq += diff * diff;
        }
        stats.std_ms = std::sqrt(sum_sq / static_cast<double>(stats.count));

        return stats;
    }

    // Print formatted report
    void print_report(std::ostream& out = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);

        out << "\n=== Profiler Report ===" << std::endl;
        out << std::setw(30) << std::left << "Scope"
            << std::setw(10) << std::right << "Count"
            << std::setw(15) << "Total (ms)"
            << std::setw(15) << "Mean (ms)"
            << std::setw(15) << "Min (ms)"
            << std::setw(15) << "Max (ms)"
            << std::endl;
        out << std::string(100, '-') << std::endl;

        // Sort scopes by total time (descending)
        std::vector<std::pair<std::string, double>> sorted;
        for (const auto& [name, times] : timings_) {
            double total = 0.0;
            for (double t : times) total += t;
            sorted.emplace_back(name, total);
        }
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [name, _] : sorted) {
            auto stats = get_stats_unlocked(name);
            out << std::setw(30) << std::left << name
                << std::setw(10) << std::right << stats.count
                << std::setw(15) << std::fixed << std::setprecision(2) << stats.total_ms
                << std::setw(15) << stats.mean_ms
                << std::setw(15) << stats.min_ms
                << std::setw(15) << stats.max_ms
                << std::endl;
        }

        out << std::string(100, '-') << std::endl;
    }

    // Reset all profiling data
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        timings_.clear();
        active_timers_.clear();
    }

    // Get all scope names
    std::vector<std::string> get_scope_names() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        for (const auto& [name, _] : timings_) {
            names.push_back(name);
        }
        return names;
    }

    // Check if scope exists
    bool has_scope(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return timings_.find(name) != timings_.end();
    }

    // RAII scope helper
    class Scope {
    public:
        explicit Scope(const std::string& name) : name_(name) {
            Profiler::instance().start_scope(name_);
        }

        ~Scope() {
            Profiler::instance().end_scope(name_);
        }

        // Non-copyable
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        std::string name_;
    };

private:
    Profiler() = default;

    // Non-copyable singleton
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Get stats without locking (for internal use when already locked)
    ProfileStats get_stats_unlocked(const std::string& name) const {
        ProfileStats stats;

        auto it = timings_.find(name);
        if (it == timings_.end() || it->second.empty()) {
            return stats;
        }

        const auto& times = it->second;
        stats.count = times.size();

        stats.min_ms = times[0];
        stats.max_ms = times[0];
        stats.total_ms = 0.0;

        for (double t : times) {
            stats.total_ms += t;
            stats.min_ms = std::min(stats.min_ms, t);
            stats.max_ms = std::max(stats.max_ms, t);
        }

        stats.mean_ms = stats.total_ms / static_cast<double>(stats.count);

        return stats;
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<double>> timings_;
    std::unordered_map<std::string, Timer> active_timers_;
};

// Macro for easy profiling
#ifndef LIGHTWATCH_DISABLE_PROFILING
    #define PROFILE_SCOPE(name) \
        ::lightwatch::Profiler::Scope _profile_scope_##__LINE__(name)
    #define PROFILE_FUNCTION() \
        ::lightwatch::Profiler::Scope _profile_scope_func(__func__)
#else
    #define PROFILE_SCOPE(name) ((void)0)
    #define PROFILE_FUNCTION() ((void)0)
#endif

// Convenience functions
inline void profile_start(const std::string& name) {
    Profiler::instance().start_scope(name);
}

inline void profile_end(const std::string& name) {
    Profiler::instance().end_scope(name);
}

inline void profile_reset() {
    Profiler::instance().reset();
}

inline void profile_report() {
    Profiler::instance().print_report();
}

}  // namespace lightwatch

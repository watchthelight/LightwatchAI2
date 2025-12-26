// Phase 39: Profiler Tests

#include <lightwatch/profiler.hpp>
#include <iostream>
#include <sstream>
#include <thread>
#include <cmath>

using namespace lightwatch;

bool float_eq(double a, double b, double eps = 1.0) {
    return std::abs(a - b) < eps;
}

// Test 1: Timer basic operation
bool test_phase_39_timer() {
    Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();

    double elapsed = timer.elapsed_ms();

    // Should be approximately 10ms (allow some variance)
    if (elapsed < 5.0 || elapsed > 50.0) {
        std::cerr << "timer: unexpected elapsed time " << elapsed << " ms" << std::endl;
        return false;
    }

    // Test microseconds
    double us = timer.elapsed_us();
    if (us < 5000.0 || us > 50000.0) {
        std::cerr << "timer: unexpected elapsed_us " << us << std::endl;
        return false;
    }

    std::cout << "test_phase_39_timer: PASSED (elapsed=" << elapsed << "ms)" << std::endl;
    return true;
}

// Test 2: Profiler basic operation
bool test_phase_39_profiler() {
    Profiler::instance().reset();

    // Profile a scope manually
    Profiler::instance().start_scope("test_scope");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    Profiler::instance().end_scope("test_scope");

    // Check stats
    auto stats = Profiler::instance().get_stats("test_scope");

    if (stats.count != 1) {
        std::cerr << "profiler: wrong count " << stats.count << std::endl;
        return false;
    }

    if (stats.total_ms < 1.0 || stats.total_ms > 50.0) {
        std::cerr << "profiler: unexpected total_ms " << stats.total_ms << std::endl;
        return false;
    }

    std::cout << "test_phase_39_profiler: PASSED" << std::endl;
    return true;
}

// Test 3: Report output
bool test_phase_39_report() {
    Profiler::instance().reset();

    // Add some timings
    for (int i = 0; i < 3; ++i) {
        Profiler::instance().start_scope("report_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Profiler::instance().end_scope("report_test");
    }

    // Capture report output
    std::ostringstream oss;
    Profiler::instance().print_report(oss);

    std::string report = oss.str();

    // Should contain headers
    if (report.find("Profiler Report") == std::string::npos) {
        std::cerr << "report: missing header" << std::endl;
        return false;
    }

    if (report.find("report_test") == std::string::npos) {
        std::cerr << "report: missing scope name" << std::endl;
        return false;
    }

    if (report.find("Count") == std::string::npos) {
        std::cerr << "report: missing Count column" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_report: PASSED" << std::endl;
    return true;
}

// Test 4: Nested scopes
bool test_phase_39_nested() {
    Profiler::instance().reset();

    // Outer scope
    Profiler::instance().start_scope("outer");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    // Inner scope
    Profiler::instance().start_scope("inner");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Profiler::instance().end_scope("inner");

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Profiler::instance().end_scope("outer");

    // Check both recorded
    auto outer_stats = Profiler::instance().get_stats("outer");
    auto inner_stats = Profiler::instance().get_stats("inner");

    if (outer_stats.count != 1) {
        std::cerr << "nested: outer count wrong" << std::endl;
        return false;
    }

    if (inner_stats.count != 1) {
        std::cerr << "nested: inner count wrong" << std::endl;
        return false;
    }

    // Outer should be longer than inner
    if (outer_stats.total_ms <= inner_stats.total_ms) {
        std::cerr << "nested: outer should be longer than inner" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_nested: PASSED" << std::endl;
    return true;
}

// Test 5: RAII Scope
bool test_phase_39_raii_scope() {
    Profiler::instance().reset();

    {
        Profiler::Scope scope("raii_scope");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto stats = Profiler::instance().get_stats("raii_scope");

    if (stats.count != 1) {
        std::cerr << "raii_scope: wrong count" << std::endl;
        return false;
    }

    if (stats.total_ms < 1.0) {
        std::cerr << "raii_scope: time not recorded" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_raii_scope: PASSED" << std::endl;
    return true;
}

// Test 6: Multiple measurements
bool test_phase_39_multiple() {
    Profiler::instance().reset();

    const int N = 5;
    for (int i = 0; i < N; ++i) {
        PROFILE_SCOPE("multiple_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto stats = Profiler::instance().get_stats("multiple_test");

    if (stats.count != static_cast<size_t>(N)) {
        std::cerr << "multiple: wrong count " << stats.count << std::endl;
        return false;
    }

    // Mean should be approximately total/count
    double expected_mean = stats.total_ms / static_cast<double>(N);
    if (!float_eq(stats.mean_ms, expected_mean, 0.1)) {
        std::cerr << "multiple: mean mismatch" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_multiple: PASSED" << std::endl;
    return true;
}

// Test 7: Stats calculation
bool test_phase_39_stats() {
    Profiler::instance().reset();

    // Create timings with known variance
    for (int i = 0; i < 3; ++i) {
        Profiler::instance().start_scope("stats_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + i * 2));  // 2, 4, 6 ms
        Profiler::instance().end_scope("stats_test");
    }

    auto stats = Profiler::instance().get_stats("stats_test");

    if (stats.count != 3) {
        std::cerr << "stats: wrong count" << std::endl;
        return false;
    }

    // Min should be less than max
    if (stats.min_ms >= stats.max_ms) {
        std::cerr << "stats: min should be < max" << std::endl;
        return false;
    }

    // Mean should be between min and max
    if (stats.mean_ms < stats.min_ms || stats.mean_ms > stats.max_ms) {
        std::cerr << "stats: mean should be between min and max" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_stats: PASSED" << std::endl;
    return true;
}

// Test 8: Reset functionality
bool test_phase_39_reset() {
    Profiler::instance().reset();

    // Add some data
    Profiler::instance().start_scope("reset_test");
    Profiler::instance().end_scope("reset_test");

    // Verify it exists
    if (!Profiler::instance().has_scope("reset_test")) {
        std::cerr << "reset: scope should exist before reset" << std::endl;
        return false;
    }

    // Reset
    Profiler::instance().reset();

    // Should not exist anymore
    if (Profiler::instance().has_scope("reset_test")) {
        std::cerr << "reset: scope should not exist after reset" << std::endl;
        return false;
    }

    std::cout << "test_phase_39_reset: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 39: Profiler Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_39_timer()) ++failures;
    if (!test_phase_39_profiler()) ++failures;
    if (!test_phase_39_report()) ++failures;
    if (!test_phase_39_nested()) ++failures;
    if (!test_phase_39_raii_scope()) ++failures;
    if (!test_phase_39_multiple()) ++failures;
    if (!test_phase_39_stats()) ++failures;
    if (!test_phase_39_reset()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 39 tests passed (8/8) ===" << std::endl;
    return 0;
}

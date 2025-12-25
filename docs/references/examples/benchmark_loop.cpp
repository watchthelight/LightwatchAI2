// File: docs/references/examples/benchmark_loop.cpp
// Part of: LightwatchAI2 Master Prompt Reference Files
// Referenced by: Master_Prompt.md > BENCHMARK SPECIFICATION

// Pseudocode for benchmark measurement
// Actual implementation will use std::chrono

#include <chrono>

using namespace std::chrono;

void run_benchmark(int warmup, int iterations, int tokens_per_iter) {
    // Warmup: discard these iterations
    for (int i = 0; i < warmup; i++) {
        generate(prompt, tokens_per_iter);  // Discard
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        generate(prompt, tokens_per_iter);
    }
    auto end = high_resolution_clock::now();

    double seconds = duration<double>(end - start).count();
    double tokens_per_second = (iterations * tokens_per_iter) / seconds;

    // Report results
    std::cout << "tokens_per_second: " << tokens_per_second << std::endl;
}

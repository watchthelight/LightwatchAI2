// Phase 38: CLI Tests

#include <lightwatch/cli.hpp>
#include <iostream>
#include <sstream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::cli;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Generate options defaults
bool test_phase_38_generate_options() {
    GenerateOptions opts;

    if (opts.max_tokens != 100) {
        std::cerr << "generate_options: wrong default max_tokens" << std::endl;
        return false;
    }

    if (!float_eq(opts.temperature, 0.8f)) {
        std::cerr << "generate_options: wrong default temperature" << std::endl;
        return false;
    }

    if (opts.top_k != 40) {
        std::cerr << "generate_options: wrong default top_k" << std::endl;
        return false;
    }

    if (!float_eq(opts.top_p, 0.9f)) {
        std::cerr << "generate_options: wrong default top_p" << std::endl;
        return false;
    }

    if (opts.json_output != false) {
        std::cerr << "generate_options: wrong default json_output" << std::endl;
        return false;
    }

    if (opts.stream != true) {
        std::cerr << "generate_options: wrong default stream" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_generate_options: PASSED" << std::endl;
    return true;
}

// Test 2: Benchmark options defaults
bool test_phase_38_benchmark_options() {
    BenchmarkOptions opts;

    if (opts.prompt_tokens != 128) {
        std::cerr << "benchmark_options: wrong default prompt_tokens" << std::endl;
        return false;
    }

    if (opts.generate_tokens != 128) {
        std::cerr << "benchmark_options: wrong default generate_tokens" << std::endl;
        return false;
    }

    if (opts.warmup != 5) {
        std::cerr << "benchmark_options: wrong default warmup" << std::endl;
        return false;
    }

    if (opts.iterations != 100) {
        std::cerr << "benchmark_options: wrong default iterations" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_benchmark_options: PASSED" << std::endl;
    return true;
}

// Test 3: JSON escape
bool test_phase_38_json_escape() {
    std::string input = "Hello\n\"World\"\t\\";
    std::string expected = "Hello\\n\\\"World\\\"\\t\\\\";
    std::string result = json_escape(input);

    if (result != expected) {
        std::cerr << "json_escape: got '" << result << "' expected '" << expected << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_json_escape: PASSED" << std::endl;
    return true;
}

// Test 4: Generate command execution
bool test_phase_38_generate() {
    GenerateOptions opts;
    opts.prompt = "Hello";
    opts.max_tokens = 10;
    opts.stream = false;
    opts.json_output = false;

    // Redirect stdout
    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());

    int result = run_generate(opts);

    std::cout.rdbuf(old_cout);

    if (result != 0) {
        std::cerr << "generate: returned error " << result << std::endl;
        return false;
    }

    // Should have produced some output
    if (capture.str().empty()) {
        std::cerr << "generate: no output" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_generate: PASSED" << std::endl;
    return true;
}

// Test 5: Generate with JSON output
bool test_phase_38_generate_json() {
    GenerateOptions opts;
    opts.prompt = "Test";
    opts.max_tokens = 5;
    opts.stream = false;
    opts.json_output = true;

    // Redirect stdout
    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());

    int result = run_generate(opts);

    std::cout.rdbuf(old_cout);

    if (result != 0) {
        std::cerr << "generate_json: returned error " << result << std::endl;
        return false;
    }

    std::string output = capture.str();

    // Should contain JSON structure
    if (output.find("{") == std::string::npos) {
        std::cerr << "generate_json: no opening brace" << std::endl;
        return false;
    }
    if (output.find("}") == std::string::npos) {
        std::cerr << "generate_json: no closing brace" << std::endl;
        return false;
    }
    if (output.find("\"prompt\"") == std::string::npos) {
        std::cerr << "generate_json: missing prompt field" << std::endl;
        return false;
    }
    if (output.find("\"generated_tokens\"") == std::string::npos) {
        std::cerr << "generate_json: missing generated_tokens field" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_generate_json: PASSED" << std::endl;
    return true;
}

// Test 6: Benchmark command execution (reduced iterations for fast testing)
bool test_phase_38_benchmark() {
    BenchmarkOptions opts;
    opts.prompt_tokens = 4;
    opts.generate_tokens = 2;
    opts.warmup = 1;
    opts.iterations = 1;
    opts.json_output = false;

    // Redirect stdout
    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());

    int result = run_benchmark(opts);

    std::cout.rdbuf(old_cout);

    if (result != 0) {
        std::cerr << "benchmark: returned error " << result << std::endl;
        return false;
    }

    std::string output = capture.str();

    // Should report tokens/second
    if (output.find("tok/s") == std::string::npos) {
        std::cerr << "benchmark: missing tok/s" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_benchmark: PASSED" << std::endl;
    return true;
}

// Test 7: Info command execution
bool test_phase_38_info() {
    // Redirect stdout
    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());

    int result = run_info("", false);

    std::cout.rdbuf(old_cout);

    if (result != 0) {
        std::cerr << "info: returned error " << result << std::endl;
        return false;
    }

    std::string output = capture.str();

    // Should contain model info
    if (output.find("Model") == std::string::npos) {
        std::cerr << "info: missing Model" << std::endl;
        return false;
    }
    if (output.find("parameters") == std::string::npos) {
        std::cerr << "info: missing parameters" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_info: PASSED" << std::endl;
    return true;
}

// Test 8: Temperature effect (lower = less diverse)
bool test_phase_38_temperature() {
    GenerateOptions opts1;
    opts1.prompt = "Test";
    opts1.max_tokens = 5;
    opts1.temperature = 0.1f;  // Low temperature
    opts1.stream = false;
    opts1.json_output = false;

    GenerateOptions opts2;
    opts2.prompt = "Test";
    opts2.max_tokens = 5;
    opts2.temperature = 2.0f;  // High temperature
    opts2.stream = false;
    opts2.json_output = false;

    // Both should succeed (basic smoke test)
    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture1, capture2;

    std::cout.rdbuf(capture1.rdbuf());
    int r1 = run_generate(opts1);
    std::cout.rdbuf(capture2.rdbuf());
    int r2 = run_generate(opts2);
    std::cout.rdbuf(old_cout);

    if (r1 != 0 || r2 != 0) {
        std::cerr << "temperature: generation failed" << std::endl;
        return false;
    }

    // Both should produce output
    if (capture1.str().empty() || capture2.str().empty()) {
        std::cerr << "temperature: empty output" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_temperature: PASSED" << std::endl;
    return true;
}

// Test 9: Top-k effect
bool test_phase_38_top_k() {
    GenerateOptions opts;
    opts.prompt = "Test";
    opts.max_tokens = 10;
    opts.top_k = 10;  // Constrained vocabulary
    opts.stream = false;
    opts.json_output = false;

    std::streambuf* old_cout = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());

    int result = run_generate(opts);

    std::cout.rdbuf(old_cout);

    if (result != 0) {
        std::cerr << "top_k: generation failed" << std::endl;
        return false;
    }

    if (capture.str().empty()) {
        std::cerr << "top_k: empty output" << std::endl;
        return false;
    }

    std::cout << "test_phase_38_top_k: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 38: CLI Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_38_generate_options()) ++failures;
    if (!test_phase_38_benchmark_options()) ++failures;
    if (!test_phase_38_json_escape()) ++failures;
    if (!test_phase_38_generate()) ++failures;
    if (!test_phase_38_generate_json()) ++failures;
    if (!test_phase_38_benchmark()) ++failures;
    if (!test_phase_38_info()) ++failures;
    if (!test_phase_38_temperature()) ++failures;
    if (!test_phase_38_top_k()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 38 tests passed (9/9) ===" << std::endl;
    return 0;
}

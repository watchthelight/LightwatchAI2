// Phase 38: CLI/REPL Interface
// Command-line interface for generation, benchmarking, and interactive use

#pragma once

#include <lightwatch/models/gpt.hpp>
#include <lightwatch/generate.hpp>
#include <lightwatch/serialize.hpp>
#include <lightwatch/tokenizer/bpe.hpp>
#include <lightwatch/init.hpp>
#include <lightwatch/autograd.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>

namespace lightwatch::cli {

struct GenerateOptions {
    std::string prompt;
    std::string model_path;
    size_t max_tokens = 100;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    bool json_output = false;
    bool stream = true;
};

struct BenchmarkOptions {
    std::string model_path;
    size_t prompt_tokens = 128;
    size_t generate_tokens = 128;
    size_t warmup = 5;
    size_t iterations = 100;
    bool json_output = false;
};

struct InfoOptions {
    std::string model_path;
    bool json_output = false;
};

// Helper: escape string for JSON
inline std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

// Create a small test model for testing
inline models::GPT2 create_test_model() {
    models::GPT2Config cfg;
    cfg.vocab_size = 100;  // Tiny vocab for fast testing
    cfg.max_seq_len = 32;
    cfg.embed_dim = 32;
    cfg.num_heads = 2;
    cfg.num_layers = 1;  // Single layer for speed
    cfg.ffn_dim = 64;
    cfg.dropout_p = 0.0f;

    models::GPT2 model(cfg);

    init::InitConfig init_cfg;
    init_cfg.seed = 42;
    init::init_gpt2_weights(model, init_cfg);

    return model;
}

// Run generate command
inline int run_generate(const GenerateOptions& opts) {
    try {
        // Create model
        models::GPT2 model = create_test_model();
        model.eval();

        // Load weights if specified
        if (!opts.model_path.empty()) {
            load_weights(opts.model_path, model);
        }

        // Tokenize prompt (simple char-based for testing)
        std::vector<TokenId> prompt_tokens;
        auto vocab_size = model.config().vocab_size;
        if (!opts.prompt.empty()) {
            for (size_t i = 0; i < opts.prompt.size() && prompt_tokens.size() < 20; ++i) {
                if (opts.prompt[i] != ' ') {
                    prompt_tokens.push_back(static_cast<TokenId>(opts.prompt[i] % vocab_size));
                }
            }
        }
        if (prompt_tokens.empty()) {
            prompt_tokens.push_back(1);  // Default start token
        }

        // Setup sampling config
        SamplingConfig config;
        config.max_new_tokens = opts.max_tokens;
        config.temperature = opts.temperature;
        config.top_k = opts.top_k;
        config.top_p = opts.top_p;
        config.do_sample = true;
        config.seed = static_cast<unsigned int>(std::time(nullptr));
        config.eos_token_id = vocab_size - 1;  // Use last token as EOS

        autograd::NoGradGuard no_grad;

        // Generate tokens
        std::vector<TokenId> generated;

        if (opts.stream && !opts.json_output) {
            // Streaming output
            generate_sample_streaming(model, prompt_tokens,
                [&generated, vocab_size](TokenId token) {
                    generated.push_back(token);
                    // Print token as character (simple representation)
                    char c = static_cast<char>('A' + (token % 26));
                    std::cout << c << std::flush;
                    (void)vocab_size;
                    return true;
                }, config);
            std::cout << std::endl;
        } else {
            // Non-streaming
            generated = generate_sample(model, prompt_tokens, config);
        }

        // Get only new tokens
        std::vector<TokenId> new_tokens(generated.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
                                        generated.end());

        // Build output string
        std::string output_text;
        for (TokenId t : new_tokens) {
            output_text += static_cast<char>('A' + (t % 26));
        }

        if (opts.json_output) {
            std::cout << "{\n";
            std::cout << "  \"prompt\": \"" << json_escape(opts.prompt) << "\",\n";
            std::cout << "  \"prompt_tokens\": " << prompt_tokens.size() << ",\n";
            std::cout << "  \"generated_tokens\": " << new_tokens.size() << ",\n";
            std::cout << "  \"output\": \"" << json_escape(output_text) << "\",\n";
            std::cout << "  \"temperature\": " << opts.temperature << ",\n";
            std::cout << "  \"top_k\": " << opts.top_k << ",\n";
            std::cout << "  \"top_p\": " << opts.top_p << "\n";
            std::cout << "}" << std::endl;
        } else if (!opts.stream) {
            std::cout << output_text << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Run benchmark command
inline int run_benchmark(const BenchmarkOptions& opts) {
    try {
        // Create model
        models::GPT2 model = create_test_model();
        model.eval();

        // Load weights if specified
        if (!opts.model_path.empty()) {
            load_weights(opts.model_path, model);
        }

        // Create prompt of specified length
        std::vector<TokenId> prompt(opts.prompt_tokens, 1);

        SamplingConfig config;
        config.max_new_tokens = opts.generate_tokens;
        config.temperature = 0.8f;
        config.do_sample = true;
        config.seed = 42;

        autograd::NoGradGuard no_grad;

        if (!opts.json_output) {
            std::cout << "Warmup (" << opts.warmup << " iterations)..." << std::endl;
        }

        // Warmup
        for (size_t i = 0; i < opts.warmup; ++i) {
            generate_sample(model, prompt, config);
        }

        if (!opts.json_output) {
            std::cout << "Benchmarking (" << opts.iterations << " iterations)..." << std::endl;
        }

        // Timed runs
        std::vector<double> times;
        times.reserve(opts.iterations);

        for (size_t i = 0; i < opts.iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            generate_sample(model, prompt, config);
            auto end = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms);
        }

        // Compute statistics
        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg_ms = sum / static_cast<double>(times.size());

        double tokens_per_sec = opts.generate_tokens * 1000.0 / avg_ms;

        if (opts.json_output) {
            std::cout << "{\n";
            std::cout << "  \"prompt_tokens\": " << opts.prompt_tokens << ",\n";
            std::cout << "  \"generate_tokens\": " << opts.generate_tokens << ",\n";
            std::cout << "  \"iterations\": " << opts.iterations << ",\n";
            std::cout << "  \"avg_ms\": " << std::fixed << std::setprecision(2) << avg_ms << ",\n";
            std::cout << "  \"min_ms\": " << min_time << ",\n";
            std::cout << "  \"max_ms\": " << max_time << ",\n";
            std::cout << "  \"tokens_per_second\": " << std::fixed << std::setprecision(1) << tokens_per_sec << "\n";
            std::cout << "}" << std::endl;
        } else {
            std::cout << "=== Benchmark Results ===" << std::endl;
            std::cout << "Prompt tokens: " << opts.prompt_tokens << std::endl;
            std::cout << "Generated tokens: " << opts.generate_tokens << std::endl;
            std::cout << "Iterations: " << opts.iterations << std::endl;
            std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_ms << " ms" << std::endl;
            std::cout << "Min time: " << min_time << " ms" << std::endl;
            std::cout << "Max time: " << max_time << " ms" << std::endl;
            std::cout << "Tokens/second: " << std::fixed << std::setprecision(1) << tokens_per_sec << " tok/s" << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Run REPL command
inline int run_repl(const std::string& model_path) {
    try {
        std::cout << "Lightwatch REPL" << std::endl;
        std::cout << "Type your prompt and press Enter (empty line to quit)" << std::endl;
        std::cout << std::endl;

        // Create model
        models::GPT2 model = create_test_model();
        model.eval();

        // Load weights if specified
        if (!model_path.empty()) {
            load_weights(model_path, model);
            std::cout << "Loaded weights from " << model_path << std::endl;
        }

        auto vocab_size = model.config().vocab_size;

        SamplingConfig config;
        config.max_new_tokens = 20;  // Shorter for faster REPL
        config.temperature = 0.8f;
        config.top_k = 40;
        config.top_p = 0.9f;
        config.do_sample = true;
        config.eos_token_id = vocab_size - 1;

        autograd::NoGradGuard no_grad;

        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) {
                break;
            }

            if (line.empty()) {
                break;
            }

            // Tokenize
            std::vector<TokenId> prompt_tokens;
            for (size_t i = 0; i < line.size() && prompt_tokens.size() < 20; ++i) {
                if (line[i] != ' ') {
                    prompt_tokens.push_back(static_cast<TokenId>(line[i] % vocab_size));
                }
            }
            if (prompt_tokens.empty()) {
                prompt_tokens.push_back(1);
            }

            config.seed = static_cast<unsigned int>(std::time(nullptr));

            // Generate
            std::cout << "< " << std::flush;
            generate_sample_streaming(model, prompt_tokens,
                [](TokenId token) {
                    char c = static_cast<char>('A' + (token % 26));
                    std::cout << c << std::flush;
                    return true;
                }, config);
            std::cout << std::endl << std::endl;
        }

        std::cout << "Goodbye!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Run info command
inline int run_info(const std::string& model_path, bool json_output = false) {
    try {
        if (model_path.empty()) {
            // Show info for default model
            models::GPT2 model = create_test_model();
            auto state = model.state_dict();

            size_t total_params = 0;
            for (const auto& [name, tensor] : state) {
                total_params += tensor.numel();
            }

            if (json_output) {
                std::cout << "{\n";
                std::cout << "  \"model_type\": \"gpt2\",\n";
                std::cout << "  \"num_tensors\": " << state.size() << ",\n";
                std::cout << "  \"total_params\": " << total_params << ",\n";
                std::cout << "  \"memory_mb\": " << std::fixed << std::setprecision(2)
                          << (total_params * 4.0 / 1024.0 / 1024.0) << "\n";
                std::cout << "}" << std::endl;
            } else {
                std::cout << "=== Model Info ===" << std::endl;
                std::cout << "Model type: GPT-2" << std::endl;
                std::cout << "Number of tensors: " << state.size() << std::endl;
                std::cout << "Total parameters: " << total_params << std::endl;
                std::cout << "Memory: " << std::fixed << std::setprecision(2)
                          << (total_params * 4.0 / 1024.0 / 1024.0) << " MB" << std::endl;
            }
            return 0;
        }

        // Check if file is valid
        if (!is_valid_lwbin(model_path)) {
            std::cerr << "Error: Invalid or corrupted .lwbin file" << std::endl;
            return 1;
        }

        auto header = read_header(model_path);
        auto tensors = inspect_weights(model_path);

        size_t total_params = 0;
        for (const auto& meta : tensors) {
            size_t numel = 1;
            for (int64_t dim : meta.shape) {
                numel *= static_cast<size_t>(dim);
            }
            total_params += numel;
        }

        if (json_output) {
            std::cout << "{\n";
            std::cout << "  \"file\": \"" << json_escape(model_path) << "\",\n";
            std::cout << "  \"version\": " << header.version << ",\n";
            std::cout << "  \"num_tensors\": " << header.tensor_count << ",\n";
            std::cout << "  \"total_params\": " << total_params << ",\n";
            std::cout << "  \"memory_mb\": " << std::fixed << std::setprecision(2)
                      << (total_params * 4.0 / 1024.0 / 1024.0) << ",\n";
            std::cout << "  \"tensors\": [\n";
            for (size_t i = 0; i < tensors.size(); ++i) {
                const auto& meta = tensors[i];
                std::cout << "    {\"name\": \"" << json_escape(meta.name) << "\", \"shape\": [";
                for (size_t j = 0; j < meta.shape.size(); ++j) {
                    std::cout << meta.shape[j];
                    if (j < meta.shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]}";
                if (i < tensors.size() - 1) std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "  ]\n";
            std::cout << "}" << std::endl;
        } else {
            std::cout << "=== Model Info ===" << std::endl;
            std::cout << "File: " << model_path << std::endl;
            std::cout << "Format version: " << header.version << std::endl;
            std::cout << "Number of tensors: " << header.tensor_count << std::endl;
            std::cout << "Total parameters: " << total_params << std::endl;
            std::cout << "Memory: " << std::fixed << std::setprecision(2)
                      << (total_params * 4.0 / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << std::endl;
            std::cout << "Tensors:" << std::endl;
            for (const auto& meta : tensors) {
                std::cout << "  " << meta.name << ": [";
                for (size_t i = 0; i < meta.shape.size(); ++i) {
                    std::cout << meta.shape[i];
                    if (i < meta.shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

// Parse command line arguments
inline void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  generate    Generate text from a prompt\n";
    std::cout << "  benchmark   Run performance benchmark\n";
    std::cout << "  repl        Interactive REPL mode\n";
    std::cout << "  info        Show model information\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --help                Show this help\n";
    std::cout << "  --model PATH          Model weights file (.lwbin)\n";
    std::cout << "  --prompt TEXT         Input prompt for generation\n";
    std::cout << "  --max-tokens N        Maximum tokens to generate (default: 100)\n";
    std::cout << "  --temperature T       Sampling temperature (default: 0.8)\n";
    std::cout << "  --top-k K             Top-k sampling (default: 40)\n";
    std::cout << "  --top-p P             Top-p (nucleus) sampling (default: 0.9)\n";
    std::cout << "  --json                Output as JSON\n";
    std::cout << "  --no-stream           Don't stream output\n";
    std::cout << "  --prompt-tokens N     Prompt size for benchmark (default: 128)\n";
    std::cout << "  --generate-tokens N   Tokens to generate for benchmark (default: 128)\n";
    std::cout << "  --warmup N            Warmup iterations (default: 5)\n";
    std::cout << "  --iterations N        Benchmark iterations (default: 100)\n";
}

inline int parse_and_run(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "--help" || command == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    // Parse options
    GenerateOptions gen_opts;
    BenchmarkOptions bench_opts;
    std::string model_path;
    bool json_output = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
            gen_opts.model_path = model_path;
            bench_opts.model_path = model_path;
        } else if (arg == "--prompt" && i + 1 < argc) {
            gen_opts.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            gen_opts.max_tokens = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (arg == "--temperature" && i + 1 < argc) {
            gen_opts.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--top-k" && i + 1 < argc) {
            gen_opts.top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            gen_opts.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (arg == "--json") {
            json_output = true;
            gen_opts.json_output = true;
            bench_opts.json_output = true;
        } else if (arg == "--no-stream") {
            gen_opts.stream = false;
        } else if (arg == "--prompt-tokens" && i + 1 < argc) {
            bench_opts.prompt_tokens = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (arg == "--generate-tokens" && i + 1 < argc) {
            bench_opts.generate_tokens = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            bench_opts.warmup = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (arg == "--iterations" && i + 1 < argc) {
            bench_opts.iterations = static_cast<size_t>(std::atoi(argv[++i]));
        }
    }

    if (command == "generate") {
        return run_generate(gen_opts);
    } else if (command == "benchmark") {
        return run_benchmark(bench_opts);
    } else if (command == "repl") {
        return run_repl(model_path);
    } else if (command == "info") {
        return run_info(model_path, json_output);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}

}  // namespace lightwatch::cli

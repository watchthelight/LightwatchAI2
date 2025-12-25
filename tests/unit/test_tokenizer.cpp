// Phase 06: BPE Tokenizer Tests

#include <lightwatch/tokenizer/bpe.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace lightwatch::tokenizer;

// Test roundtrip encoding/decoding
bool test_phase_06_tokenizer_roundtrip() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "Hello, world!";
    auto tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);

    if (decoded != text) {
        std::cerr << "tokenizer_roundtrip: expected '" << text
                  << "' got '" << decoded << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_roundtrip: PASSED" << std::endl;
    return true;
}

// Test special EOS token
bool test_phase_06_tokenizer_special() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    if (tokenizer.eos_id() != 50256) {
        std::cerr << "tokenizer_special: EOS should be 50256, got "
                  << tokenizer.eos_id() << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_special: PASSED" << std::endl;
    return true;
}

// Test unicode text
bool test_phase_06_tokenizer_unicode() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "日本語";
    auto tokens = tokenizer.encode(text);

    if (tokens.empty()) {
        std::cerr << "tokenizer_unicode: should produce tokens" << std::endl;
        return false;
    }

    std::string decoded = tokenizer.decode(tokens);
    if (decoded != text) {
        std::cerr << "tokenizer_unicode: roundtrip failed, got '"
                  << decoded << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_unicode: PASSED" << std::endl;
    return true;
}

// Test empty string
bool test_phase_06_tokenizer_empty() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    auto tokens = tokenizer.encode("");

    if (!tokens.empty()) {
        std::cerr << "tokenizer_empty: should return empty vector" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_empty: PASSED" << std::endl;
    return true;
}

// Test vocab size
bool test_phase_06_tokenizer_vocab_size() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    if (tokenizer.vocab_size() != 50257) {
        std::cerr << "tokenizer_vocab_size: expected 50257, got "
                  << tokenizer.vocab_size() << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_vocab_size: PASSED" << std::endl;
    return true;
}

// Test whitespace handling
bool test_phase_06_tokenizer_whitespace() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "  leading and trailing  ";
    auto tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);

    if (decoded != text) {
        std::cerr << "tokenizer_whitespace: roundtrip failed" << std::endl;
        std::cerr << "  expected: '" << text << "'" << std::endl;
        std::cerr << "  got: '" << decoded << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_whitespace: PASSED" << std::endl;
    return true;
}

// Test numbers
bool test_phase_06_tokenizer_numbers() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "12345";
    auto tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);

    if (decoded != text) {
        std::cerr << "tokenizer_numbers: roundtrip failed, got '"
                  << decoded << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_numbers: PASSED" << std::endl;
    return true;
}

// Test long text
bool test_phase_06_tokenizer_long_text() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    // Generate long text
    std::string text;
    for (int i = 0; i < 100; ++i) {
        text += "The quick brown fox jumps over the lazy dog. ";
    }

    auto tokens = tokenizer.encode(text);

    if (tokens.empty()) {
        std::cerr << "tokenizer_long_text: should produce tokens" << std::endl;
        return false;
    }

    // Check all token IDs are valid
    for (TokenId id : tokens) {
        if (id < 0 || static_cast<size_t>(id) >= tokenizer.vocab_size()) {
            std::cerr << "tokenizer_long_text: invalid token id " << id << std::endl;
            return false;
        }
    }

    std::string decoded = tokenizer.decode(tokens);
    if (decoded != text) {
        std::cerr << "tokenizer_long_text: roundtrip failed" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_long_text: PASSED" << std::endl;
    return true;
}

// Test emoji
bool test_phase_06_tokenizer_emoji() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "Hello 🌍 World";
    auto tokens = tokenizer.encode(text);

    if (tokens.empty()) {
        std::cerr << "tokenizer_emoji: should produce tokens" << std::endl;
        return false;
    }

    std::string decoded = tokenizer.decode(tokens);
    if (decoded != text) {
        std::cerr << "tokenizer_emoji: roundtrip failed" << std::endl;
        std::cerr << "  expected: '" << text << "'" << std::endl;
        std::cerr << "  got: '" << decoded << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_emoji: PASSED" << std::endl;
    return true;
}

// Test newlines
bool test_phase_06_tokenizer_newlines() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    std::string text = "line1\nline2\r\nline3";
    auto tokens = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(tokens);

    if (decoded != text) {
        std::cerr << "tokenizer_newlines: roundtrip failed" << std::endl;
        return false;
    }

    std::cout << "test_phase_06_tokenizer_newlines: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_06_tokenizer_roundtrip()) ++failures;
    if (!test_phase_06_tokenizer_special()) ++failures;
    if (!test_phase_06_tokenizer_unicode()) ++failures;
    if (!test_phase_06_tokenizer_empty()) ++failures;
    if (!test_phase_06_tokenizer_vocab_size()) ++failures;
    if (!test_phase_06_tokenizer_whitespace()) ++failures;
    if (!test_phase_06_tokenizer_numbers()) ++failures;
    if (!test_phase_06_tokenizer_long_text()) ++failures;
    if (!test_phase_06_tokenizer_emoji()) ++failures;
    if (!test_phase_06_tokenizer_newlines()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 06 tests passed (10/10)" << std::endl;
    return 0;
}

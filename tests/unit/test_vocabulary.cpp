// Phase 07: Vocabulary Tests

#include <lightwatch/tokenizer/vocabulary.hpp>
#include <iostream>
#include <string>

using namespace lightwatch::tokenizer;

// Test vocabulary size from GPT-2 encoder
bool test_phase_07_vocab_size() {
    Vocabulary vocab = Vocabulary::from_encoder_json("data/vocab/encoder.json");

    if (vocab.size() != 50257) {
        std::cerr << "vocab_size: expected 50257, got " << vocab.size() << std::endl;
        return false;
    }

    std::cout << "test_phase_07_vocab_size: PASSED" << std::endl;
    return true;
}

// Test token to ID conversion
bool test_phase_07_token_to_id() {
    Vocabulary vocab = Vocabulary::from_encoder_json("data/vocab/encoder.json");

    // "Hello" in GPT-2 vocab - note this is the byte-encoded version
    // The actual token is "Hello" which maps to ID 15496
    TokenId id = vocab.token_to_id("Hello");

    if (id < 0 || static_cast<size_t>(id) >= vocab.size()) {
        std::cerr << "token_to_id: invalid ID for 'Hello'" << std::endl;
        return false;
    }

    // Verify round trip
    std::string token = vocab.id_to_token(id);
    if (token != "Hello") {
        std::cerr << "token_to_id: roundtrip failed, got '" << token << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_07_token_to_id: PASSED" << std::endl;
    return true;
}

// Test ID to token for EOS
bool test_phase_07_id_to_token() {
    Vocabulary vocab = Vocabulary::from_encoder_json("data/vocab/encoder.json");

    // Token 50256 should be <|endoftext|>
    std::string token = vocab.id_to_token(50256);

    if (token != "<|endoftext|>") {
        std::cerr << "id_to_token: expected '<|endoftext|>', got '" << token << "'" << std::endl;
        return false;
    }

    std::cout << "test_phase_07_id_to_token: PASSED" << std::endl;
    return true;
}

// Test EOS ID
bool test_phase_07_eos_id() {
    Vocabulary vocab;

    if (vocab.eos_id() != 50256) {
        std::cerr << "eos_id: expected 50256, got " << vocab.eos_id() << std::endl;
        return false;
    }

    std::cout << "test_phase_07_eos_id: PASSED" << std::endl;
    return true;
}

// Test contains method
bool test_phase_07_contains() {
    Vocabulary vocab = Vocabulary::from_encoder_json("data/vocab/encoder.json");

    // GPT-2 uses byte-level encoding, so "the" with leading space
    // is represented as a specific unicode sequence
    // Let's check for a simple ASCII token
    if (!vocab.contains("the")) {
        // Try without the leading space marker
        if (!vocab.contains("a")) {
            std::cerr << "contains: vocabulary should contain 'a'" << std::endl;
            return false;
        }
    }

    // Check that non-existent token returns false
    if (vocab.contains("xyznonexistent123")) {
        std::cerr << "contains: should not contain 'xyznonexistent123'" << std::endl;
        return false;
    }

    std::cout << "test_phase_07_contains: PASSED" << std::endl;
    return true;
}

// Test add and lookup roundtrip
bool test_phase_07_roundtrip() {
    Vocabulary vocab;

    // Add tokens
    TokenId id1 = vocab.add_token("hello");
    TokenId id2 = vocab.add_token("world");
    TokenId id3 = vocab.add_token("hello");  // Duplicate

    // Check IDs
    if (id1 != 0) {
        std::cerr << "roundtrip: first token should have ID 0" << std::endl;
        return false;
    }
    if (id2 != 1) {
        std::cerr << "roundtrip: second token should have ID 1" << std::endl;
        return false;
    }
    if (id3 != id1) {
        std::cerr << "roundtrip: duplicate token should return same ID" << std::endl;
        return false;
    }

    // Check lookup
    if (vocab.token_to_id("hello") != id1) {
        std::cerr << "roundtrip: token_to_id mismatch" << std::endl;
        return false;
    }
    if (vocab.id_to_token(id1) != "hello") {
        std::cerr << "roundtrip: id_to_token mismatch" << std::endl;
        return false;
    }

    // Check size
    if (vocab.size() != 2) {
        std::cerr << "roundtrip: size should be 2, got " << vocab.size() << std::endl;
        return false;
    }

    std::cout << "test_phase_07_roundtrip: PASSED" << std::endl;
    return true;
}

// Test save and load
bool test_phase_07_save_load() {
    Vocabulary vocab;
    vocab.add_token("hello");
    vocab.add_token("world");
    vocab.add_token("test\nwith\nnewlines");

    // Save to temp file
    std::string path = "/tmp/test_vocab.txt";
    vocab.save(path);

    // Load back
    Vocabulary loaded = Vocabulary::load(path);

    if (loaded.size() != vocab.size()) {
        std::cerr << "save_load: size mismatch" << std::endl;
        return false;
    }

    if (loaded.token_to_id("hello") != vocab.token_to_id("hello")) {
        std::cerr << "save_load: 'hello' ID mismatch" << std::endl;
        return false;
    }

    if (loaded.id_to_token(2) != "test\nwith\nnewlines") {
        std::cerr << "save_load: newline handling failed" << std::endl;
        return false;
    }

    std::cout << "test_phase_07_save_load: PASSED" << std::endl;
    return true;
}

// Test special tokens
bool test_phase_07_special_tokens() {
    Vocabulary vocab;

    // Check special token constants
    if (SpecialTokens::PAD != 50256) {
        std::cerr << "special_tokens: PAD should be 50256" << std::endl;
        return false;
    }
    if (SpecialTokens::EOS != 50256) {
        std::cerr << "special_tokens: EOS should be 50256" << std::endl;
        return false;
    }

    // Check is_special_token
    if (!vocab.is_special_token(50256)) {
        std::cerr << "special_tokens: 50256 should be special" << std::endl;
        return false;
    }

    std::cout << "test_phase_07_special_tokens: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_07_vocab_size()) ++failures;
    if (!test_phase_07_token_to_id()) ++failures;
    if (!test_phase_07_id_to_token()) ++failures;
    if (!test_phase_07_eos_id()) ++failures;
    if (!test_phase_07_contains()) ++failures;
    if (!test_phase_07_roundtrip()) ++failures;
    if (!test_phase_07_save_load()) ++failures;
    if (!test_phase_07_special_tokens()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 07 tests passed (8/8)" << std::endl;
    return 0;
}

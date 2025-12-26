# Unit tests for LightwatchAI2

# Phase 01: Config header test
add_executable(test_config test_config.cpp)
lightwatch_set_compiler_flags(test_config)
add_test(NAME test_phase_01_config_header COMMAND test_config)

# Phase 02: Memory management tests
add_executable(test_memory test_memory.cpp)
lightwatch_set_compiler_flags(test_memory)
add_test(NAME test_phase_02_memory COMMAND test_memory)

# Phase 03: Tensor core tests
add_executable(test_tensor test_tensor.cpp)
lightwatch_set_compiler_flags(test_tensor)
add_test(NAME test_phase_03_tensor COMMAND test_tensor)

# Phase 04: SIMD operations tests
add_executable(test_simd test_simd.cpp)
lightwatch_set_compiler_flags(test_simd)
add_test(NAME test_phase_04_simd COMMAND test_simd)

# Phase 05: Autograd engine tests
add_executable(test_autograd test_autograd.cpp)
lightwatch_set_compiler_flags(test_autograd)
add_test(NAME test_phase_05_autograd COMMAND test_autograd)

# Phase 06: BPE Tokenizer tests
add_executable(test_tokenizer test_tokenizer.cpp)
lightwatch_set_compiler_flags(test_tokenizer)
add_test(NAME test_phase_06_tokenizer COMMAND test_tokenizer)
set_tests_properties(test_phase_06_tokenizer PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Phase 07: Vocabulary tests
add_executable(test_vocabulary test_vocabulary.cpp)
lightwatch_set_compiler_flags(test_vocabulary)
add_test(NAME test_phase_07_vocabulary COMMAND test_vocabulary)
set_tests_properties(test_phase_07_vocabulary PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Phase 08: Embedding layer tests
add_executable(test_embedding test_embedding.cpp)
lightwatch_set_compiler_flags(test_embedding)
add_test(NAME test_phase_08_embedding COMMAND test_embedding)

# Phase 09: Positional encoding tests
add_executable(test_positional test_positional.cpp)
lightwatch_set_compiler_flags(test_positional)
add_test(NAME test_phase_09_positional COMMAND test_positional)

# Phase 11: Linear layer tests
add_executable(test_linear test_linear.cpp)
lightwatch_set_compiler_flags(test_linear)
add_test(NAME test_phase_11_linear COMMAND test_linear)

# Phase 12: Activation functions tests
add_executable(test_activations test_activations.cpp)
lightwatch_set_compiler_flags(test_activations)
add_test(NAME test_phase_12_activations COMMAND test_activations)

# Phase 13: Layer normalization tests
add_executable(test_normalization test_normalization.cpp)
lightwatch_set_compiler_flags(test_normalization)
add_test(NAME test_phase_13_normalization COMMAND test_normalization)

# Phase 14: Dropout tests
add_executable(test_dropout test_dropout.cpp)
lightwatch_set_compiler_flags(test_dropout)
add_test(NAME test_phase_14_dropout COMMAND test_dropout)

# Phase 15: Attention tests
add_executable(test_attention test_attention.cpp)
lightwatch_set_compiler_flags(test_attention)
add_test(NAME test_phase_15_attention COMMAND test_attention)

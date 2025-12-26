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

# Phase 16: Multi-Head Attention tests
add_executable(test_mha test_mha.cpp)
lightwatch_set_compiler_flags(test_mha)
add_test(NAME test_phase_16_mha COMMAND test_mha)

# Phase 17: FFN Block tests
add_executable(test_ffn test_ffn.cpp)
lightwatch_set_compiler_flags(test_ffn)
add_test(NAME test_phase_17_ffn COMMAND test_ffn)

# Phase 18: Transformer Encoder Block tests
add_executable(test_transformer test_transformer.cpp)
lightwatch_set_compiler_flags(test_transformer)
add_test(NAME test_phase_18_transformer COMMAND test_transformer)

# Phase 19: Transformer Decoder Block tests
add_executable(test_decoder test_decoder.cpp)
lightwatch_set_compiler_flags(test_decoder)
add_test(NAME test_phase_19_decoder COMMAND test_decoder)

# Phase 21: Loss Functions tests
add_executable(test_loss test_loss.cpp)
lightwatch_set_compiler_flags(test_loss)
add_test(NAME test_phase_21_loss COMMAND test_loss)

# Phase 22: SGD Optimizer tests
add_executable(test_sgd test_sgd.cpp)
lightwatch_set_compiler_flags(test_sgd)
add_test(NAME test_phase_22_sgd COMMAND test_sgd)

# Phase 23: Adam Optimizer tests
add_executable(test_adam test_adam.cpp)
lightwatch_set_compiler_flags(test_adam)
add_test(NAME test_phase_23_adam COMMAND test_adam)

# Phase 24: LR Scheduler tests
add_executable(test_scheduler test_scheduler.cpp)
lightwatch_set_compiler_flags(test_scheduler)
add_test(NAME test_phase_24_scheduler COMMAND test_scheduler)

# Phase 25: Gradient Clipping tests
add_executable(test_clip test_clip.cpp)
lightwatch_set_compiler_flags(test_clip)
add_test(NAME test_phase_25_clip COMMAND test_clip)

# Phase 26: Checkpointing tests
add_executable(test_checkpoint test_checkpoint.cpp)
lightwatch_set_compiler_flags(test_checkpoint)
add_test(NAME test_phase_26_checkpoint COMMAND test_checkpoint)

# Phase 27: Data Loading tests
add_executable(test_dataloader test_dataloader.cpp)
lightwatch_set_compiler_flags(test_dataloader)
add_test(NAME test_phase_27_dataloader COMMAND test_dataloader)
set_tests_properties(test_phase_27_dataloader PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Phase 28: Batch Processing tests
add_executable(test_collate test_collate.cpp)
lightwatch_set_compiler_flags(test_collate)
add_test(NAME test_phase_28_collate COMMAND test_collate)

# Phase 29: Training Loop tests
add_executable(test_trainer test_trainer.cpp)
lightwatch_set_compiler_flags(test_trainer)
add_test(NAME test_phase_29_trainer COMMAND test_trainer)

# Phase 31: GPT-2 Architecture tests
add_executable(test_gpt test_gpt.cpp)
lightwatch_set_compiler_flags(test_gpt)
add_test(NAME test_phase_31_gpt COMMAND test_gpt)

# Phase 32: Model Config tests
add_executable(test_model_config test_model_config.cpp)
lightwatch_set_compiler_flags(test_model_config)
target_link_libraries(test_model_config PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME test_phase_32_model_config COMMAND test_model_config)
set_tests_properties(test_phase_32_model_config PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Phase 33: Weight Initialization tests
add_executable(test_init test_init.cpp)
lightwatch_set_compiler_flags(test_init)
add_test(NAME test_phase_33_init COMMAND test_init)

# Phase 34: Greedy Decode tests
add_executable(test_generate test_generate.cpp)
lightwatch_set_compiler_flags(test_generate)
add_test(NAME test_phase_34_generate COMMAND test_generate)

# Phase 35: Sampling tests
add_executable(test_sampling test_sampling.cpp)
lightwatch_set_compiler_flags(test_sampling)
add_test(NAME test_phase_35_sampling COMMAND test_sampling)

# Phase 36: KV-Cache tests
add_executable(test_cache test_cache.cpp)
lightwatch_set_compiler_flags(test_cache)
add_test(NAME test_phase_36_cache COMMAND test_cache)

# Phase 37: Serialization tests
add_executable(test_serialize test_serialize.cpp)
lightwatch_set_compiler_flags(test_serialize)
add_test(NAME test_phase_37_serialize COMMAND test_serialize)

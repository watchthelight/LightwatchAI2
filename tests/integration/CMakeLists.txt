# Integration tests for LightwatchAI2

# Phase 10: Foundation integration tests
add_executable(test_foundation test_foundation.cpp)
lightwatch_set_compiler_flags(test_foundation)
add_test(NAME test_phase_10_foundation COMMAND test_foundation)
set_tests_properties(test_phase_10_foundation PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Phase 20: Neural Core integration tests (CHECKPOINT 2)
add_executable(test_neural_core test_neural_core.cpp)
lightwatch_set_compiler_flags(test_neural_core)
add_test(NAME test_phase_20_neural_core COMMAND test_neural_core)

# Phase 30: Training integration tests (CHECKPOINT 3)
add_executable(test_training test_training.cpp)
lightwatch_set_compiler_flags(test_training)
add_test(NAME test_phase_30_training COMMAND test_training)

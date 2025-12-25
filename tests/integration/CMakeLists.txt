# Integration tests for LightwatchAI2

# Phase 10: Foundation integration tests
add_executable(test_foundation test_foundation.cpp)
lightwatch_set_compiler_flags(test_foundation)
add_test(NAME test_phase_10_foundation COMMAND test_foundation)
set_tests_properties(test_phase_10_foundation PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

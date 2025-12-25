# Unit tests for LightwatchAI2

# Phase 01: Config header test
add_executable(test_config test_config.cpp)
lightwatch_set_compiler_flags(test_config)
add_test(NAME test_phase_01_config_header COMMAND test_config)

# Phase 02: Memory management tests
add_executable(test_memory test_memory.cpp)
lightwatch_set_compiler_flags(test_memory)
add_test(NAME test_phase_02_memory COMMAND test_memory)

# Unit tests for LightwatchAI2

# Phase 01: Config header test
add_executable(test_config test_config.cpp)
lightwatch_set_compiler_flags(test_config)
add_test(NAME test_phase_01_config_header COMMAND test_config)

# Smoke test - validates toolchain before main implementation

add_executable(smoke_test smoke_test.cpp)

add_test(NAME test_phase_00_smoke COMMAND smoke_test)

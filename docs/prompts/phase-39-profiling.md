# Phase 39: Profiling

## Objective
Implement profiling infrastructure for performance analysis and optimization.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 38 | CLI with benchmark command |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 38 | src/main.cpp | CLI infrastructure |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/profiler.hpp | Profiler, Timer | Phase 40 |
| src/profiler.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

class Timer {
public:
    void start();
    void stop();
    double elapsed_ms() const;
    double elapsed_us() const;

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

class Profiler {
public:
    static Profiler& instance();

    void start_scope(const std::string& name);
    void end_scope(const std::string& name);

    struct Stats {
        size_t count;
        double total_ms;
        double mean_ms;
        double min_ms;
        double max_ms;
    };

    Stats get_stats(const std::string& name) const;
    void print_report() const;
    void reset();

    // RAII scope helper
    class Scope {
    public:
        Scope(const std::string& name);
        ~Scope();
    private:
        std::string name_;
    };

private:
    std::unordered_map<std::string, std::vector<double>> timings_;
};

#define PROFILE_SCOPE(name) \
    Profiler::Scope _profile_scope_##__LINE__(name)

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Timer**: High-resolution timing using chrono
2. **Profiler**: Collect timings per named scope
3. **Stats**: Compute mean, min, max from collected data
4. **Report**: Print formatted summary of all scopes

### Performance Constraints
- Timer overhead: < 100ns per start/stop
- Profiler: O(1) for start/end scope

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_39_timer` | Time sleep(10ms) | ~10ms elapsed |
| `test_phase_39_profiler` | Profile scope | Stats recorded |
| `test_phase_39_report` | print_report() | Formatted output |
| `test_phase_39_nested` | Nested scopes | Both recorded |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_39" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/profiler.hpp`
- [ ] Profiler accurately measures known operations

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-450 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Use PROFILE_SCOPE macro for easy instrumentation
- Singleton pattern for global profiler access
- Disable in release builds via preprocessor
- Report helps identify optimization opportunities

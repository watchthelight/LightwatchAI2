// LightwatchAI2 Smoke Test
// Validates toolchain before main implementation

#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Test basic C++17 features
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};

    float sum = 0;
    for (auto x : v) sum += x;

    // Test floating point operations
    float expected = 10.0f;
    if (std::abs(sum - expected) > 1e-6f) {
        std::cerr << "ERROR: sum = " << sum << ", expected " << expected << std::endl;
        return 1;
    }

    std::cout << "Smoke test passed: sum = " << sum << std::endl;
    return 0;
}

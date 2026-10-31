// CHECKS: skippy-no-raw-loops
#include <algorithm>
#include <array>
int sum(const std::array<int, 4>& values) {
    int total = 0;
    for (std::size_t i = 0; i < values.size(); ++i) total += values[i];
    // CHECK: [[@LINE-1]]:5: warning: raw loop
    int again = 0;
    for (const int value : values) again += value;
    return total + again;
}

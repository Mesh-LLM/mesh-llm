// CHECKS: skippy-no-throwing-accessor
#include "expected_stub.h"
#include <optional>

std::optional<int> find(int value);
std::expected<int, int> load(int value);

int throwing_optional(int value) { return find(value).value(); }
// CHECK: [[@LINE-1]]:43: warning: '.value()' is the throwing accessor

int throwing_expected(int value) { return load(value).value(); }
// CHECK: [[@LINE-1]]:43: warning: '.value()' is the throwing accessor

int composed(int value) { return find(value).value_or(0); }  // negative: value_or composes

int guarded(int value) {
    const std::optional<int> result = find(value);
    return result ? *result : 0;  // negative: a guarded read of *
}

// CHECKS: skippy-optional-error-polarity,skippy-no-expected-void
#include "expected_stub.h"
#include <optional>
enum class PlanError { Bad };

std::optional<PlanError> validate(int x) { return x < 0 ? std::optional{PlanError::Bad} : std::nullopt; }

std::expected<void, PlanError> old_shape();
// CHECK: [[@LINE-1]]:1: warning: std::expected<void, E> is banned

bool caller(int x) {
    const auto failure = validate(x);
    if (!failure) return true;
    // CHECK: [[@LINE-1]]:10: warning: std::optional<PlanError> is a failure channel (engaged means it failed)
    return failure.has_value();
}

// CHECKS: skippy-no-failure-optional
#include <optional>

enum class PlanError { Bad };
enum class PlanFailure { Worse };

std::optional<PlanError> validate(int value);
// CHECK: [[@LINE-1]]:{{(1|6)}}: warning: std::optional<PlanError> is a failure channel

std::optional<PlanFailure> validate_more(int value);
// CHECK: [[@LINE-1]]:{{(1|6)}}: warning: std::optional<PlanFailure> is a failure channel

std::optional<int> find(int value);  // negative: a value optional, not a failure channel

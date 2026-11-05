#ifndef SKIPPY_API_HOST_EXPERTS_H
#define SKIPPY_API_HOST_EXPERTS_H

/**
 * @file host_experts.h
 * @brief Routed-expert host placement (patch 0013), as a typed boundary value.
 *
 * The native loader accepts a block range whose routed-expert tensors
 * (`ffn_(up|down|gate|gate_up)_(ch|)exps`) load into host memory instead of
 * the accelerator, and silently ignores a range that fails its guards. The
 * facade refuses to forward a silent drop: an invalid range is a caller bug,
 * so it is rejected here with the rule it broke, and only valid ranges reach
 * the ABI.
 */

#include "stage_plan.h"

#include <expected>
#include <optional>

namespace skippy {

/// Mirrors the runtime's maximum host-expert block span: a wider range is
/// refused so placement override construction stays bounded by the model it
/// names.
inline constexpr std::int32_t max_host_expert_block_span = 4096;

/// Why a requested placement was refused.
enum class PlacementError {
    /// The range starts before block zero.
    negative_start,
    /// The range ends before it starts.
    inverted_range,
    /// The range is wider than `max_host_expert_block_span`.
    overspan,
};

/// The blocks whose routed experts load into host memory. Disengaged means no
/// placement was requested and every expert stays on the accelerator.
using HostExpertPlacement = std::optional<LayerRange>;

/// Validates a requested placement. An empty range is "off", not an error;
/// every other malformed range names the rule it broke.
[[nodiscard]] std::expected<HostExpertPlacement, PlacementError>
host_expert_placement(LayerRange requested) noexcept;

/// One line naming the broken rule, for failure messages.
[[nodiscard]] const char* describe(PlacementError error) noexcept;

} // namespace skippy

#endif // SKIPPY_API_HOST_EXPERTS_H

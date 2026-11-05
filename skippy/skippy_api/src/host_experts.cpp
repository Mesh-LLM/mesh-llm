#include "host_experts.h"

namespace skippy {

std::expected<HostExpertPlacement, PlacementError>
host_expert_placement(const LayerRange requested) noexcept {
    if (requested.start < 0) return std::unexpected(PlacementError::negative_start);
    if (requested.end < requested.start) return std::unexpected(PlacementError::inverted_range);
    if (requested.empty()) return HostExpertPlacement{};
    if (requested.count() > max_host_expert_block_span) {
        return std::unexpected(PlacementError::overspan);
    }
    return HostExpertPlacement{requested};
}

const char* describe(const PlacementError error) noexcept {
    switch (error) {
    case PlacementError::negative_start: return "host expert range starts before block zero";
    case PlacementError::inverted_range: return "host expert range ends before it starts";
    case PlacementError::overspan: return "host expert range is wider than the block span cap";
    }
    return "unknown host expert placement error";
}

} // namespace skippy

// T4 (docs/TYPE_DESIGN.md): presence first, count second. The head is the
// cls.output tensor (llama-arch.cpp:536, loaded by bert, modern-bert,
// neo-bert, qwen3, qwen3vl, qwen35, lfm2); n_cls_out defaults to 1 whether
// or not it exists, so it is only ever read as the width of a head proven
// present here. A ModelFamily is never an input to this decision (P0-B).
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "llama.h"
#include "llama/model_inventory.h"
#include "skippy/narrow.h"

namespace skippy::llama {

class ClassifierHead {
public:
    [[nodiscard]] static std::optional<ClassifierHead>
    detect(std::span<const probe::TensorInfo> inventory_sorted_by_name) noexcept {
        static constexpr std::string_view kHead = "cls.output.weight";
        const auto found =
            std::ranges::lower_bound(inventory_sorted_by_name, kHead, {}, &probe::TensorInfo::name);
        if (found == inventory_sorted_by_name.end() || found->name != kHead) return std::nullopt;
        // Width from the tensor itself; n_cls_out only confirms it.
        const auto width = found->dimension_count >= 2 ? narrow<std::uint32_t>(found->dimensions[1])
                                                       : std::nullopt;
        // A guarded read, not transform(): optional::transform calls the
        // throwing accessor internally, and detect is noexcept.
        if (!width.has_value()) return std::nullopt;
        return ClassifierHead{*width};
    }

    [[nodiscard]] constexpr std::uint32_t outputs() const noexcept { return outputs_; }

private:
    constexpr explicit ClassifierHead(std::uint32_t outputs) noexcept : outputs_(outputs) {}
    std::uint32_t outputs_;
};

// Pins (docs/TYPE_DESIGN.md T4): presence is structural, never nominal.
static_assert(!std::is_constructible_v<skippy::llama::ClassifierHead, std::uint32_t>);
static_assert(!std::is_default_constructible_v<skippy::llama::ClassifierHead>);

} // namespace skippy::llama

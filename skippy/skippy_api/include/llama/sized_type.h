// T7 (docs/TYPE_DESIGN.md): a ggml type that ggml can size. The wire value
// is bounds-checked against GGML_TYPE_COUNT before the one cast, and removed
// slots (block <= 0 or element == 0) are rejected, so row_size_bytes can
// never compute with a type ggml cannot size.
#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "ggml.h"

namespace skippy::llama {

class SizedGgmlType {
public:
    [[nodiscard]] static std::optional<SizedGgmlType> parse(std::uint32_t wire) noexcept {
        if (wire >= static_cast<std::uint32_t>(GGML_TYPE_COUNT)) return std::nullopt;
        const auto type = static_cast<ggml_type>(wire); // the one cast, after the bound
        const std::int64_t block = ggml_blck_size(type);
        const std::size_t element = ggml_type_size(type);
        if (block <= 0 || element == 0) return std::nullopt; // removed slots
        return SizedGgmlType{type, block, element};
    }

    [[nodiscard]] ggml_type native() const noexcept { return type_; }
    [[nodiscard]] std::int64_t block() const noexcept { return block_; }
    [[nodiscard]] std::size_t element() const noexcept { return element_; }

private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): block/element widths arrive in ggml's fixed pair order from one validated caller
    SizedGgmlType(ggml_type type, std::int64_t block, std::size_t element) noexcept
        : type_(type), block_(block), element_(element) {}
    ggml_type type_;
    std::int64_t block_;
    std::size_t element_;
};

// Pins (docs/TYPE_DESIGN.md T7): the only constructor is the bounded parse.
static_assert(!std::is_constructible_v<skippy::llama::SizedGgmlType, ggml_type>);
static_assert(!std::is_default_constructible_v<skippy::llama::SizedGgmlType>);

} // namespace skippy::llama

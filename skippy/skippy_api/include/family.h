#ifndef SKIPPY_API_FAMILY_H
#define SKIPPY_API_FAMILY_H

/// Model families this facade knows by name.
///
/// A family is a closed set, not a string: callers branch on the enum and the
/// GGUF's `general.architecture` is parsed once, at the edge.

#include <string_view>

namespace skippy {

enum class ModelFamily : unsigned char {
    /// Anything we do not recognise, including GGUF files with no architecture.
    Unknown = 0,
    Llama = 1,
    Qwen = 2,
    Mistral = 3,
    Gemma = 4,
    Phi = 5,
    Bert = 6,
    Mamba = 7,
    Rwkv = 8,
    T5 = 9,
};

/// Maps a GGUF `general.architecture` value onto a family.
///
/// Versioned names map to their family: `qwen2`, `qwen2vl` and `qwen3` are all
/// `Qwen`. An unrecognised name is `Unknown`, never an error.
[[nodiscard]] ModelFamily family_of(std::string_view architecture) noexcept;

/// Stable name for logs. Never empty.
[[nodiscard]] std::string_view describe(ModelFamily family) noexcept;

/// True when the family is known to hold recurrent state instead of KV.
[[nodiscard]] constexpr bool is_recurrent(ModelFamily family) noexcept {
    return family == ModelFamily::Mamba || family == ModelFamily::Rwkv;
}

} // namespace skippy

#endif // SKIPPY_API_FAMILY_H

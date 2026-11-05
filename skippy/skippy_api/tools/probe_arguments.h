#pragma once

#include <charconv>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <system_error>

namespace skippy::probe {

/// Parses the bounded decimal counters used by the diagnostic probes.
/// Trailing bytes, overflow and values below the argument's minimum are
/// rejected before a range or allocation is built from the number.
inline std::expected<std::int32_t, std::string>
parse_probe_integer(std::string_view text, std::string_view argument, std::int32_t minimum) {
    if (!text.empty()) {
        std::int32_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
            value >= minimum) {
            return value;
        }
    }
    return std::unexpected(std::string(argument) + " must be a decimal integer between " +
                           std::to_string(minimum) + " and 2147483647");
}

} // namespace skippy::probe

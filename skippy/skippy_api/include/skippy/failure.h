// T9 (docs/TYPE_DESIGN.md): out-of-memory is a code, never text.
// out_of_memory() allocates nothing, so a noexcept barrier's bad_alloc arm
// is nothrow by construction.
#pragma once

#include <concepts>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace skippy {

template <typename Code>
    requires std::is_enum_v<Code> && requires {
        Code::OutOfMemory;
        Code::Internal;
    }
struct Failure {
    Code code;
    std::string message; // empty for OutOfMemory by construction

    [[nodiscard]] static Failure out_of_memory() noexcept { return {Code::OutOfMemory, {}}; }
    [[nodiscard]] static Failure internal(std::string_view what) noexcept {
        Failure failure{Code::Internal, {}};
        try {
            failure.message = what; // text is optional; the code is not
        } catch (...) {
        }
        return failure;
    }
};

} // namespace skippy

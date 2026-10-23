// The forbidden spellings, as lines of C++ that fail the build
// (docs/TYPE_DESIGN.md, TYPE-RULES.md 4). Extend this file when closing
// a finding: the rule, pinned. The llama-side types carry their own pins in
// their headers, where llama.h is on the include path.
//
// The pins are trait-based rather than requires-based: a requires-expression
// over a concrete type is a hard error on the older toolchains in the lane
// matrix, and a pin that breaks the build the wrong way protects nothing.

#include "skippy/failure.h"
#include "skippy/geometry.h"
#include "skippy/narrow.h"
#include "skippy/wire.h"
#include "types.h"

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

// ---- T6 narrow/checked_*: the only integer-cast spelling ------------------

static_assert(std::is_same_v<decltype(skippy::narrow<std::int32_t>(std::int64_t{})),
                             std::optional<std::int32_t>>);
static_assert(skippy::narrow<std::uint8_t>(300) == std::nullopt); // loss is a value
static_assert(skippy::narrow<std::uint8_t>(255) == std::uint8_t{255});
static_assert(skippy::checked_mul<std::uint32_t>(0x1'0000U, 0x1'0000U) == std::nullopt);
static_assert(skippy::checked_mul<std::uint32_t>(2U, 3U) == 6U);
static_assert(skippy::checked_add<std::uint32_t>(0xFFFF'FFFFU, 1U) == std::nullopt);
static_assert(skippy::checked_add<std::uint32_t>(1U, 2U) == 3U);

// ---- T2 SessionGeometry: parse, don't validate -----------------------------

static_assert(!std::is_default_constructible_v<skippy::SessionGeometry>);
static_assert(!std::is_constructible_v<skippy::SessionGeometry, skippy::TokensPerSequence,
                                       skippy::SequenceCount>);
static_assert(skippy::SessionGeometry::make({512}, {1}).has_value());
static_assert(!skippy::SessionGeometry::make({0}, {1}).has_value());
static_assert(!skippy::SessionGeometry::make({512}, {0}).has_value());
static_assert(!skippy::SessionGeometry::make({512}, {65}).has_value());
static_assert(!skippy::SessionGeometry::make({0x7FFF'FFFF}, {2}).has_value()); // product would wrap
static_assert(skippy::SessionGeometry::make({512}, {2})->total_tokens() == 1024);
static_assert(std::is_trivially_copyable_v<skippy::SessionGeometry>); // a Value (TYPE-RULES.md 3.1)

// ---- T7 WireScalar ---------------------------------------------------------

static_assert(skippy::WireScalar<std::uint32_t>);
static_assert(!skippy::WireScalar<std::string>); // only integral/floating scalars cross the wire

// ---- T9 Failure<Code>: OOM is a code, never text ----------------------------

namespace contracts {
enum class Code { OutOfMemory, Internal };
static_assert(noexcept(skippy::Failure<Code>::out_of_memory()));
static_assert(noexcept(skippy::Failure<Code>::internal(std::declval<std::string_view>())));
static_assert(std::is_nothrow_move_constructible_v<skippy::Failure<Code>>);
} // namespace contracts

// ---- layout facts the ABI depends on ----------------------------------------

static_assert(std::is_trivially_copyable_v<skippy::Token>);
static_assert(sizeof(skippy::Token) == sizeof(std::int32_t));
static_assert(alignof(skippy::Token) == alignof(std::int32_t));

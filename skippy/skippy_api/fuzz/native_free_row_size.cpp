// The parser only needs the byte-size seam; ggml's real implementation is
// linked into the llama side and exercised by the probes. This F32-flavoured
// mirror of its contract - same rank bounds, same rejection of negative
// extents and wrapping byte totals - keeps the native-free targets (tests and
// fuzzers) pinned to the parser's typed-error path. One stub, not two.
#include "llama/model_inventory.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

namespace skippy::probe {

std::optional<std::uint64_t> row_size_bytes(std::uint32_t /*type*/,
                                            const std::array<std::int64_t, 4>& dimensions,
                                            std::uint32_t rank) {
    if (rank == 0U || rank > 4U) return std::nullopt;
    std::uint64_t bytes = 4; // F32 row element
    for (std::uint32_t dim = 0; dim < rank; ++dim) {
        if (dimensions[dim] < 0) return std::nullopt;
        const auto extent = static_cast<std::uint64_t>(dimensions[dim]);
        if (extent != 0U && bytes > std::numeric_limits<std::uint64_t>::max() / extent) {
            return std::nullopt;
        }
        bytes *= extent;
    }
    return bytes;
}

} // namespace skippy::probe

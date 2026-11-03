#include "llama/model_inventory.h"

#include "ggml.h"

#include <limits>

namespace skippy::probe {

std::optional<std::uint64_t> row_size_bytes(std::uint32_t type,
                                            const std::array<std::int64_t, 4>& dimensions,
                                            std::uint32_t rank) {
    if (rank == 0U || rank > 4U) return std::nullopt;
    if (type >= static_cast<std::uint32_t>(GGML_TYPE_COUNT)) return std::nullopt;
    const auto native = static_cast<ggml_type>(type);
    const std::int64_t block = ggml_blck_size(native);
    const std::uint64_t element = ggml_type_size(native);
    // Removed type slots (Q4_2/Q4_3) report a zero block: the native sizing
    // helpers would then read type_traits out of bounds, divide by zero, or
    // assert on a first dimension that is not a block multiple. For a
    // block-aligned row ggml_row_size is exactly (ne / blck) * type_size —
    // computed here with the wrap checked, because the native form multiplies
    // before it divides and its intermediate can wrap first.
    if (block <= 0 || element == 0U || dimensions[0] < 0 || dimensions[0] % block != 0) {
        return std::nullopt;
    }
    const auto rows = static_cast<std::uint64_t>(dimensions[0] / block);
    if (rows > std::numeric_limits<std::uint64_t>::max() / element) return std::nullopt;
    return checked_tensor_bytes(rows * element, dimensions, rank);
}

} // namespace skippy::probe

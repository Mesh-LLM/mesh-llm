#include "llama/model_inventory.h"

#include "ggml.h"

namespace skippy::probe {

std::uint64_t row_size_bytes(std::uint32_t type, const std::array<std::int64_t, 4>& dimensions,
                             std::uint32_t rank) {
    if (rank == 0) return 0;
    std::uint64_t bytes = ggml_row_size(static_cast<ggml_type>(type), dimensions[0]);
    for (std::uint32_t dim = 1; dim < rank; ++dim) {
        bytes *= static_cast<std::uint64_t>(dimensions[dim]);
    }
    return bytes;
}

} // namespace skippy::probe

#ifndef SKIPPY_API_LLAMA_PREPARED_EXPORT_H
#define SKIPPY_API_LLAMA_PREPARED_EXPORT_H

/// A stage's export buffer, prepared once and reused across calls.
///
/// The frame a stage produces is sized by the stage's own geometry, so the
/// buffer can be owned by the stage and never reallocated after the first call.
/// This is the façade-side half of a prepared activation token: the native
/// prefill still maps the parts into its own buffers, but we stop allocating a
/// fresh frame per boundary and stop rebuilding a descriptor per call.

#include "skippy.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace skippy::llama {

class PreparedExport {
public:
    PreparedExport() = default;
    PreparedExport(PreparedExport&&) noexcept = default;
    PreparedExport& operator=(PreparedExport&&) noexcept = default;
    PreparedExport(const PreparedExport&) = delete;
    PreparedExport& operator=(const PreparedExport&) = delete;

    /// Grows the buffer to `needed` and returns all of it. Capacity is never
    /// released, so a repeated call with the same size performs no allocation.
    [[nodiscard]] std::span<std::byte> reserve(std::size_t needed) {
        if (buffer_.size() < needed) buffer_.resize(needed);
        return std::span<std::byte>{buffer_};
    }

    /// The frame as last produced, trimmed to what the native call reported.
    [[nodiscard]] std::span<const std::byte> written() const noexcept {
        return std::span<const std::byte>{buffer_.data(), written_};
    }

    /// Reports the frame size written by the native call, clamped to the
    /// buffer: a lying descriptor yields a short frame, never a span that
    /// outruns its storage.
    void set_written(std::size_t size) noexcept { written_ = std::min(size, buffer_.size()); }
    [[nodiscard]] std::size_t capacity() const noexcept { return buffer_.size(); }

    /// Descriptor shaped for this stage's export; written by the native call.
    skippy_activation_desc desc{};

private:
    std::vector<std::byte> buffer_;
    std::size_t written_ = 0;
};

} // namespace skippy::llama

#endif // SKIPPY_API_LLAMA_PREPARED_EXPORT_H

#pragma once

#include <atomic>
#include <memory>
#include <version> // __cpp_lib_atomic_shared_ptr

namespace skippy::detail {

/// Atomic handoff for the read-copy-update snapshot/publish pattern
/// (AGENTS.md, "Atomics and CAS before mutexes"): many readers `load` an
/// immutable index, publishers install a copy-on-write replacement with
/// `compare_exchange_weak`. The surface is exactly those two operations —
/// this is not a general atomic wrapper.
#ifdef __cpp_lib_atomic_shared_ptr

template <typename T> class atomic_shared_ptr {
public:
    explicit atomic_shared_ptr(std::shared_ptr<T> initial) noexcept : atom_{std::move(initial)} {}

    [[nodiscard]] std::shared_ptr<T> load(const std::memory_order order) const noexcept {
        return atom_.load(order);
    }

    bool compare_exchange_weak(std::shared_ptr<T>& expected, std::shared_ptr<T> desired,
                               const std::memory_order success,
                               const std::memory_order failure) noexcept {
        return atom_.compare_exchange_weak(expected, std::move(desired), success, failure);
    }

private:
    std::atomic<std::shared_ptr<T>> atom_;
};

#else
// Fallback for standard libraries without the std::atomic<std::shared_ptr>
// template (Xcode's libc++ through Apple clang 21 does not define it). The
// C++20 free functions are the only spelling left, and C++20 deprecates them
// — GCC 14's libstdc++ annotates them, fatal under this tree's -Werror. That
// library defines __cpp_lib_atomic_shared_ptr, so this branch is never
// selected there; a toolchain that both lacks the type and warns gets its
// suppression scoped to exactly these two bodies.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

template <typename T> class atomic_shared_ptr {
public:
    explicit atomic_shared_ptr(std::shared_ptr<T> initial) noexcept : raw_{std::move(initial)} {}

    [[nodiscard]] std::shared_ptr<T> load(const std::memory_order order) const noexcept {
        return std::atomic_load_explicit(&raw_, order);
    }

    bool compare_exchange_weak(std::shared_ptr<T>& expected, std::shared_ptr<T> desired,
                               const std::memory_order success,
                               const std::memory_order failure) noexcept {
        return std::atomic_compare_exchange_weak_explicit(&raw_, &expected, std::move(desired),
                                                          success, failure);
    }

private:
    // Mutable: the free functions take the address of a non-const shared_ptr.
    mutable std::shared_ptr<T> raw_;
};

#ifdef __clang__
#pragma clang diagnostic pop
#elifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

} // namespace skippy::detail

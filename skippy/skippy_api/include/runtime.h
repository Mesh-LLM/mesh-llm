#pragma once

// Ownership registry, generic over a native backend.
//
// Ownership, top down:
//   NativeRuntime<Backend>  owns  Backend                         (by value)
//                           owns  Backend::ModelResource          (unique_ptr, stationary)
//                           owns  Backend::SessionResource        (unique_ptr, stationary)
//   Backend::SessionResource may hold a plain ModelResource& — the runtime
//   guarantees the model outlives it (ModelInUse, shutdown order).
//   Model / Session handles own nothing: they are non-forgeable tokens.
//
// Cleanup is RAII only. A resource destructor performs complete, nonthrowing
// final cleanup, including device synchronization. There is no fallible
// close(): the native free/synchronize entry points cannot report failure.
//
// Nothing here depends on llama.cpp. The backend is the only place that does.

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace skippy {

enum class RuntimeError {
    InvalidModel,
    InvalidSession,
    ModelInUse,
    AllocationFailed,
    LoadFailed,
    SessionCreationFailed,
    UnsupportedConfiguration, // a SessionConfig the backend cannot honour
    CapacityExceeded,
    InvalidBackendResult,
};

// A load failure with the loader's reason. The message is the loader's own
// text (the inventory parser's, say) and may be empty; it travels in the
// value, never in state on the backend.
struct LoadFailure {
    RuntimeError code;
    std::string message;
};

struct ModelConfig {
    std::string path;
};

struct SessionConfig {
    // Initial embedding implementation requires an exclusive context.
    // The backend must reject unsupported configurations, not ignore this.
    bool dedicated_context = true;

    // Longest single input, in tokens. An embedding pass puts the whole input
    // through in one physical batch, so a single-sequence session's context is
    // this size; a batch session's context holds sequences x max_tokens.
    std::uint32_t max_tokens = 512;

    // How many sequences one batch may decode together. 1 keeps an exclusive
    // context; more makes the session batch-capable, sharing max_tokens of
    // K/V budget across the sequences of a single call.
    std::uint32_t sequences = 1;
};

struct ShutdownSummary {
    std::size_t sessions_released{};
    std::size_t models_released{};
};

namespace detail {

struct ModelId {
    std::uint32_t slot{};
    std::uint64_t generation{};
    auto operator<=>(const ModelId&) const = default;
};

struct SessionId {
    std::uint32_t slot{};
    std::uint64_t generation{};
    auto operator<=>(const SessionId&) const = default;
};

} // namespace detail

// A native backend: loads models and opens sessions on them. Resource types
// are the backend's own concrete types; the runtime never erases them.
template <typename B>
concept RuntimeBackend =
    std::is_object_v<B> && requires(B& backend, const ModelConfig& model_config,
                                    B::ModelResource& model, const SessionConfig& session_config) {
        {
            backend.load_model(model_config)
        } -> std::same_as<std::expected<std::unique_ptr<typename B::ModelResource>, LoadFailure>>;
        {
            backend.create_session(model, session_config)
        }
        -> std::same_as<std::expected<std::unique_ptr<typename B::SessionResource>, RuntimeError>>;
    };

template <RuntimeBackend Backend> class NativeRuntime;

// Non-forgeable, move-only access handles. NativeRuntime owns the resources.
// Dropping a handle does not release the resource: release explicitly or drain
// the runtime. No raw pointer is stored, so a stale handle is safe to reject.
class Model {
public:
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept : id_(std::exchange(other.id_, std::nullopt)) {}
    Model& operator=(Model&&) = delete;

private:
    template <RuntimeBackend B> friend class NativeRuntime;
    explicit Model(detail::ModelId id) noexcept : id_(id) {}
    std::optional<detail::ModelId> id_;
};

class Session {
public:
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept : id_(std::exchange(other.id_, std::nullopt)) {}
    Session& operator=(Session&&) = delete;

private:
    template <RuntimeBackend B> friend class NativeRuntime;
    explicit Session(detail::SessionId id) noexcept : id_(id) {}
    std::optional<detail::SessionId> id_;
};

namespace detail {

// Unique across runtime instances in this library instance. Never wrap/reuse.
// Generation 0 is reserved for "no handle". Defined in runtime.cpp.
[[nodiscard]] std::expected<std::uint64_t, RuntimeError> next_generation() noexcept;

// An entry owns a resource while its slot is in use; a null resource marks the
// slot free. Shared by every helper that walks the entry tables.
inline constexpr auto is_live = [](const auto& entry) noexcept {
    return static_cast<bool>(entry.resource);
};

template <typename Entries>
std::expected<std::uint32_t, RuntimeError> reserve_slot(Entries& entries) {
    if (const auto free = std::ranges::find_if_not(entries, is_live); free != entries.end()) {
        return static_cast<std::uint32_t>(std::ranges::distance(entries.begin(), free));
    }
    if (entries.size() >= std::numeric_limits<std::uint32_t>::max()) {
        return std::unexpected(RuntimeError::CapacityExceeded);
    }
    try {
        entries.emplace_back();
    } catch (const std::bad_alloc&) {
        return std::unexpected(RuntimeError::AllocationFailed);
    }
    return static_cast<std::uint32_t>(entries.size() - 1);
}

// A handle resolves when it still names an id (not moved-from or released),
// that id's slot is live, and the generation matches the current occupant.
template <typename Entries, typename Id>
bool contains(const Entries& entries, const std::optional<Id>& id) noexcept {
    return id && id->slot < entries.size() && is_live(entries[id->slot]) &&
           entries[id->slot].generation == id->generation;
}

// Destroys every live resource in the table and returns how many there were.
// Resetting the current entry cannot disturb the filter: it has already been
// yielded, and the view only evaluates the predicate on the elements ahead.
template <typename Entries> std::size_t release_all(Entries& entries) noexcept {
    std::size_t released = 0;
    for (auto& entry : entries | std::views::filter(is_live)) {
        entry.resource.reset();
        ++released;
    }
    return released;
}

// Runs `visit` on a resource and wraps its result, including void.
template <typename Resource, std::invocable<Resource&> Visitor>
std::expected<std::invoke_result_t<Visitor, Resource&>, RuntimeError>
visit_resource(Resource& resource, Visitor&& visit) {
    using Result = std::invoke_result_t<Visitor, Resource&>;
    if constexpr (std::is_void_v<Result>) {
        std::invoke(std::forward<Visitor>(visit), resource);
        return {};
    } else {
        return std::expected<Result, RuntimeError>{
            std::in_place,
            std::invoke(std::forward<Visitor>(visit), resource),
        };
    }
}

} // namespace detail

// Single-threaded ownership registry. The caller serializes ALL access.
// No copy/move: IDs and borrowed backend resources stay tied to this instance.
template <RuntimeBackend Backend> class NativeRuntime {
public:
    using ModelResource = Backend::ModelResource;
    using SessionResource = Backend::SessionResource;

    // Constructs the backend in place; the runtime has no "no backend" state.
    template <typename... Args>
        requires std::constructible_from<Backend, Args&&...>
    explicit NativeRuntime([[maybe_unused]] std::in_place_t tag, Args&&... args) noexcept(
        std::is_nothrow_constructible_v<Backend, Args&&...>)
        : backend_(std::forward<Args>(args)...) {}

    explicit NativeRuntime(Backend backend) noexcept(std::is_nothrow_move_constructible_v<Backend>)
        requires std::move_constructible<Backend>
        : backend_(std::move(backend)) {}

    ~NativeRuntime() noexcept {
        // Sessions, then models (shutdown); the backend is destroyed last by
        // member order, after every resource that might still call into it.
        shutdown();
    }

    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;
    NativeRuntime(NativeRuntime&&) = delete;
    NativeRuntime& operator=(NativeRuntime&&) = delete;

    [[nodiscard]] Backend& backend() noexcept { return backend_; }
    [[nodiscard]] const Backend& backend() const noexcept { return backend_; }

    [[nodiscard]] std::expected<Model, LoadFailure> load_model(const ModelConfig& config);
    [[nodiscard]] std::expected<Session, RuntimeError>
    create_session(const Model& handle, const SessionConfig& config = {});

    // Destroys the session's resource and clears the handle. Fails only when
    // the handle does not name a live session (moved-from, released, foreign).
    [[nodiscard]] std::expected<void, RuntimeError> release_session(Session& handle);

    // As release_session(); additionally fails with ModelInUse while any
    // session of this model is live, leaving model and handle untouched.
    [[nodiscard]] std::expected<void, RuntimeError> unload_model(Model& handle);

    // Destroys every live session, then every live model. Cannot fail.
    // Outstanding handles stop resolving; the runtime stays usable.
    ShutdownSummary shutdown() noexcept;

    // Scoped borrow of the resource behind a handle. `visit` receives the
    // resource by reference for the duration of the call only; nothing
    // escapes. Fails with InvalidModel / InvalidSession without calling it.
    template <std::invocable<ModelResource&> Visitor>
    [[nodiscard]] std::expected<std::invoke_result_t<Visitor, ModelResource&>, RuntimeError>
    with_model(const Model& handle, Visitor&& visit);

    template <std::invocable<const ModelResource&> Visitor>
    [[nodiscard]] std::expected<std::invoke_result_t<Visitor, const ModelResource&>, RuntimeError>
    with_model(const Model& handle, Visitor&& visit) const;

    template <std::invocable<SessionResource&> Visitor>
    [[nodiscard]] std::expected<std::invoke_result_t<Visitor, SessionResource&>, RuntimeError>
    with_session(const Session& handle, Visitor&& visit);

    template <std::invocable<const SessionResource&> Visitor>
    [[nodiscard]] std::expected<std::invoke_result_t<Visitor, const SessionResource&>, RuntimeError>
    with_session(const Session& handle, Visitor&& visit) const;

private:
    struct ModelEntry {
        std::uint64_t generation{};
        std::unique_ptr<ModelResource> resource;
    };
    struct SessionEntry {
        std::uint64_t generation{};
        detail::ModelId model{};
        std::unique_ptr<SessionResource> resource;
    };

    Backend backend_; // first: destroyed last
    std::vector<ModelEntry> models_;
    std::vector<SessionEntry> sessions_;
};

// --- definitions -------------------------------------------------------------

template <RuntimeBackend Backend>
std::expected<Model, LoadFailure> NativeRuntime<Backend>::load_model(const ModelConfig& config) {
    auto slot = detail::reserve_slot(models_);
    if (!slot) return std::unexpected(LoadFailure{.code = slot.error(), .message = {}});
    auto generation = detail::next_generation();
    if (!generation) return std::unexpected(LoadFailure{.code = generation.error(), .message = {}});

    try {
        auto model = backend_.load_model(config);
        if (!model) return std::unexpected(model.error());
        if (!*model)
            return std::unexpected(
                LoadFailure{.code = RuntimeError::InvalidBackendResult, .message = {}});
        models_[*slot] = ModelEntry{*generation, std::move(*model)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(LoadFailure{.code = RuntimeError::AllocationFailed, .message = {}});
    }
    return Model{detail::ModelId{*slot, *generation}};
}

template <RuntimeBackend Backend>
std::expected<Session, RuntimeError>
NativeRuntime<Backend>::create_session(const Model& handle, const SessionConfig& config) {
    const auto model = handle.id_;
    if (!detail::contains(models_, model)) return std::unexpected(RuntimeError::InvalidModel);
    auto slot = detail::reserve_slot(sessions_);
    if (!slot) return std::unexpected(slot.error());
    auto generation = detail::next_generation();
    if (!generation) return std::unexpected(generation.error());

    try {
        auto session = backend_.create_session(*models_[model->slot].resource, config);
        if (!session) return std::unexpected(session.error());
        if (!*session) return std::unexpected(RuntimeError::InvalidBackendResult);
        sessions_[*slot] = SessionEntry{*generation, *model, std::move(*session)};
    } catch (const std::bad_alloc&) {
        return std::unexpected(RuntimeError::AllocationFailed);
    }
    return Session{detail::SessionId{*slot, *generation}};
}

template <RuntimeBackend Backend>
std::expected<void, RuntimeError> NativeRuntime<Backend>::release_session(Session& handle) {
    const auto id = handle.id_;
    if (!detail::contains(sessions_, id)) return std::unexpected(RuntimeError::InvalidSession);
    sessions_[id->slot].resource.reset();
    handle.id_.reset();
    return {};
}

template <RuntimeBackend Backend>
std::expected<void, RuntimeError> NativeRuntime<Backend>::unload_model(Model& handle) {
    const auto id = handle.id_;
    if (!detail::contains(models_, id)) return std::unexpected(RuntimeError::InvalidModel);
    const auto uses_model = [&](const auto& session) noexcept {
        return detail::is_live(session) && session.model == *id;
    };
    if (std::ranges::any_of(sessions_, uses_model)) {
        return std::unexpected(RuntimeError::ModelInUse);
    }
    models_[id->slot].resource.reset();
    handle.id_.reset();
    return {};
}

template <RuntimeBackend Backend> ShutdownSummary NativeRuntime<Backend>::shutdown() noexcept {
    // Sessions first: a session may still touch its model during cleanup.
    const auto sessions = detail::release_all(sessions_);
    const auto models = detail::release_all(models_);
    return ShutdownSummary{sessions, models};
}

template <RuntimeBackend Backend>
template <std::invocable<typename Backend::ModelResource&> Visitor>
std::expected<std::invoke_result_t<Visitor, typename Backend::ModelResource&>, RuntimeError>
NativeRuntime<Backend>::with_model(const Model& handle, Visitor&& visit) {
    const auto id = handle.id_;
    if (!detail::contains(models_, id)) return std::unexpected(RuntimeError::InvalidModel);
    return detail::visit_resource(*models_[id->slot].resource, std::forward<Visitor>(visit));
}

template <RuntimeBackend Backend>
template <std::invocable<const typename Backend::ModelResource&> Visitor>
std::expected<std::invoke_result_t<Visitor, const typename Backend::ModelResource&>, RuntimeError>
NativeRuntime<Backend>::with_model(const Model& handle, Visitor&& visit) const {
    const auto id = handle.id_;
    if (!detail::contains(models_, id)) return std::unexpected(RuntimeError::InvalidModel);
    return detail::visit_resource(std::as_const(*models_[id->slot].resource),
                                  std::forward<Visitor>(visit));
}

template <RuntimeBackend Backend>
template <std::invocable<typename Backend::SessionResource&> Visitor>
std::expected<std::invoke_result_t<Visitor, typename Backend::SessionResource&>, RuntimeError>
NativeRuntime<Backend>::with_session(const Session& handle, Visitor&& visit) {
    const auto id = handle.id_;
    if (!detail::contains(sessions_, id)) return std::unexpected(RuntimeError::InvalidSession);
    return detail::visit_resource(*sessions_[id->slot].resource, std::forward<Visitor>(visit));
}

template <RuntimeBackend Backend>
template <std::invocable<const typename Backend::SessionResource&> Visitor>
std::expected<std::invoke_result_t<Visitor, const typename Backend::SessionResource&>, RuntimeError>
NativeRuntime<Backend>::with_session(const Session& handle, Visitor&& visit) const {
    const auto id = handle.id_;
    if (!detail::contains(sessions_, id)) return std::unexpected(RuntimeError::InvalidSession);
    return detail::visit_resource(std::as_const(*sessions_[id->slot].resource),
                                  std::forward<Visitor>(visit));
}

} // namespace skippy

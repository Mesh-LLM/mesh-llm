#pragma once

/**
 * @file llama/stage_planner.h
 * @brief RAII access to the native stage planner.
 *
 * The fastest way to expose a plan is to not copy it. A native plan already
 * owns its descriptors, identities and strings in one allocation, so this
 * wrapper owns the handle and materializes values only when a caller asks.
 * That beats the C shape (create/free plus one call per field) without
 * changing the ABI: nothing is copied until `materialize()` runs, and a
 * caller that only needs the layer range pays one call.
 */

#include "llama/backend.h"
#include "stage_load.h"
#include "stage_plan.h"

#include "skippy.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace skippy::llama {

struct StagePlanDeleter {
    void operator()(skippy_stage_plan* plan) const noexcept;
};
struct StagePlannerDeleter {
    void operator()(skippy_stage_planner* planner) const noexcept;
};

/// An immutable realized plan. Move-only; owns the native allocation.
class StagePlanHandle {
public:
    StagePlanHandle() = default;
    StagePlanHandle(StagePlanHandle&&) noexcept = default;
    StagePlanHandle& operator=(StagePlanHandle&&) noexcept = default;
    StagePlanHandle(const StagePlanHandle&) = delete;
    StagePlanHandle& operator=(const StagePlanHandle&) = delete;

    [[nodiscard]] bool valid() const noexcept { return plan_ != nullptr; }

    /// Cheapest useful read: the stage's layer window.
    [[nodiscard]] std::expected<LayerRange, PlanFailure> range() const;

    /// Copies the plan into facade values. One pass, exact reservations.
    [[nodiscard]] std::expected<StagePlan, PlanFailure> materialize() const;

    [[nodiscard]] const skippy_stage_plan* native() const noexcept { return plan_.get(); }

private:
    friend class StagePlanner;
    explicit StagePlanHandle(std::unique_ptr<skippy_stage_plan, StagePlanDeleter> plan)
        : plan_(std::move(plan)) {}

    std::unique_ptr<skippy_stage_plan, StagePlanDeleter> plan_;
};

/// Metadata-only planner. Move-only; owns the native allocation.
class StagePlanner {
public:
    StagePlanner() = default;
    StagePlanner(StagePlanner&&) noexcept = default;
    StagePlanner& operator=(StagePlanner&&) noexcept = default;
    StagePlanner(const StagePlanner&) = delete;
    StagePlanner& operator=(const StagePlanner&) = delete;

    [[nodiscard]] static std::expected<StagePlanner, PlanFailure>
    create(const StagePlannerInputs& inputs);

    /// Realizes `range` for every configured profile.
    [[nodiscard]] std::expected<StagePlanHandle, PlanFailure> realize(LayerRange range) const;

    [[nodiscard]] bool valid() const noexcept { return planner_ != nullptr; }

private:
    std::unique_ptr<skippy_stage_planner, StagePlannerDeleter> planner_;
};

/// Plans `ranges` of `path` and returns each stage's loader configuration.
///
/// This is the one entry point a caller needs to split a model: the tensor
/// inventory, the planner, the plan and the config derivation all happen here,
/// in one pass, with no intermediate objects to keep alive. Realized plans are
/// cached in `backend`'s store, which outlives the configs this returns.
[[nodiscard]] std::expected<PlanLease, PlanFailure>
plan_model_stages(LlamaBackend& backend, const std::string& path,
                  std::span<const LayerRange> ranges);

/// Plans `ranges` and returns the plans themselves.
///
/// Use this when you need to validate a chain or read semantics; use
/// `plan_model_stages` when you only need loader configuration. Both share
/// `backend`'s cache, so calling either one first warms the other.
[[nodiscard]] std::expected<std::vector<StagePlan>, PlanFailure>
plan_model_plans(LlamaBackend& backend, const std::string& path,
                 std::span<const LayerRange> ranges);

/// Validates a chain natively, without materializing anything.
[[nodiscard]] std::expected<void, PlanFailure>
validate_chain(std::span<const StagePlanHandle> plans);

} // namespace skippy::llama

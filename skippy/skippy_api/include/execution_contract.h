#ifndef SKIPPY_API_EXECUTION_CONTRACT_H
#define SKIPPY_API_EXECUTION_CONTRACT_H

/**
 * @file execution_contract.h
 * @brief The admitted execution contract, as typed data.
 *
 * The native planner serializes a stage's contract to JSON and carries it
 * opaquely into the runtime. This header is the typed shape behind that text,
 * so a caller can inspect what a stage will actually execute and check it
 * against the tensors the stage ships, without a JSON dependency.
 *
 * Validation here is the pass that does not need a graph: parameter ordering,
 * profile offsets, and the resident-set union.
 */

#include "stage_plan.h"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace skippy {

/// One admitted auxiliary (multi-token-prediction) profile.
struct AuxiliaryProfile {
    std::int32_t offset = 0;
    std::vector<std::string> parameters;
};

/// What one stage is admitted to execute, per profile.
struct ExecutionContract {
    static constexpr std::int32_t kSupportedVersion = 1;

    std::int32_t version = kSupportedVersion;
    LayerRange range;
    std::vector<std::string> decoder;
    /// Present only when the stage admits a raw embedding input.
    std::optional<std::vector<std::string>> decoder_embeddings;
    std::vector<AuxiliaryProfile> auxiliary;
};

enum class ContractError : std::uint8_t {
    UnsupportedVersion,
    RangeMismatch,
    EmptyParameter,
    ParametersNotSorted,
    DuplicateParameter,
    AuxiliaryOffsetNotContiguous,
    AuxiliaryProfileMissing,
    EmbeddingProfileMissing,
    ResidentMismatch,
};

/// Checks the invariants that hold for every admitted contract.
[[nodiscard]] std::expected<void, ContractError> validate(const ExecutionContract& contract);

/// Checks a contract against the stage it was realized for.
[[nodiscard]] std::expected<void, ContractError> validate(const ExecutionContract& contract,
                                                          const LayerRange& stage);

/// Parameters one profile is admitted to realize.
///
/// `auxiliary_offset` indexes `contract.auxiliary` and is ignored unless
/// `auxiliary` is set. An embedding-input profile requires the stage to admit
/// `decoder_embeddings`.
///
/// The returned span borrows from `contract`; it is valid for as long as that
/// object is, so do not pass a temporary.
[[nodiscard]] std::expected<std::span<const std::string>, ContractError>
parameters_for(const ExecutionContract& contract, bool auxiliary, std::int32_t auxiliary_offset,
               bool embedding_input);

/// Every parameter any profile may realize, sorted and unique.
[[nodiscard]] std::vector<TensorId> resident_parameters(const ExecutionContract& contract);

/// Checks a stage's resident tensor set against that union.
[[nodiscard]] std::expected<void, ContractError>
validate_residents(const ExecutionContract& contract, std::span<const TensorId> residents);

} // namespace skippy

#endif // SKIPPY_API_EXECUTION_CONTRACT_H

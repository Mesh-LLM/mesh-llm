#include "execution_contract.h"

#include <algorithm>
#include <set>

namespace skippy {
namespace {

std::expected<void, ContractError> validate_parameters(const std::vector<std::string>& parameters) {
    if (std::ranges::any_of(parameters, [](const std::string& name) { return name.empty(); })) {
        return std::unexpected(ContractError::EmptyParameter);
    }
    const bool sorted = std::ranges::is_sorted(parameters);
    const bool unique = std::ranges::adjacent_find(parameters) == parameters.end();
    if (!unique) return std::unexpected(ContractError::DuplicateParameter);
    if (!sorted) return std::unexpected(ContractError::ParametersNotSorted);
    return {};
}

std::expected<void, ContractError> require_parameters(const std::vector<std::string>& parameters) {
    if (parameters.empty()) return std::unexpected(ContractError::EmptyParameter);
    return validate_parameters(parameters);
}

} // namespace

std::expected<void, ContractError> validate(const ExecutionContract& contract) {
    if (contract.version != ExecutionContract::kSupportedVersion) {
        return std::unexpected(ContractError::UnsupportedVersion);
    }
    if (auto failure = require_parameters(contract.decoder); !failure) return failure;
    if (contract.decoder_embeddings) {
        if (auto failure = require_parameters(*contract.decoder_embeddings); !failure)
            return failure;
    }
    std::int32_t expected_offset = 0;
    for (const AuxiliaryProfile& profile : contract.auxiliary) {
        // Offsets index profiles by position; a gap would make index lookups
        // silently mean something else.
        if (profile.offset != expected_offset++) {
            return std::unexpected(ContractError::AuxiliaryOffsetNotContiguous);
        }
        if (auto failure = require_parameters(profile.parameters); !failure) return failure;
    }
    return {};
}

std::expected<void, ContractError> validate(const ExecutionContract& contract,
                                            const LayerRange& stage) {
    return validate(contract).and_then([&] -> std::expected<void, ContractError> {
        if (contract.range != stage) return std::unexpected(ContractError::RangeMismatch);
        return {};
    });
}

std::expected<std::span<const std::string>, ContractError>
parameters_for(const ExecutionContract& contract, bool auxiliary, std::int32_t auxiliary_offset,
               bool embedding_input) {
    return validate(contract).and_then(
        [&] -> std::expected<std::span<const std::string>, ContractError> {
            if (auxiliary) {
                const auto index = static_cast<std::size_t>(auxiliary_offset);
                if (auxiliary_offset < 0 || index >= contract.auxiliary.size()) {
                    return std::unexpected(ContractError::AuxiliaryProfileMissing);
                }
                return std::span<const std::string>{contract.auxiliary[index].parameters};
            }
            if (!embedding_input) return std::span<const std::string>{contract.decoder};
            if (!contract.decoder_embeddings) {
                return std::unexpected(ContractError::EmbeddingProfileMissing);
            }
            return std::span<const std::string>{*contract.decoder_embeddings};
        });
}

std::vector<TensorId> resident_parameters(const ExecutionContract& contract) {
    std::set<std::string> union_parameters(contract.decoder.begin(), contract.decoder.end());
    if (contract.decoder_embeddings) {
        union_parameters.insert(contract.decoder_embeddings->begin(),
                                contract.decoder_embeddings->end());
    }
    for (const AuxiliaryProfile& profile : contract.auxiliary) {
        union_parameters.insert(profile.parameters.begin(), profile.parameters.end());
    }
    std::vector<TensorId> residents;
    residents.reserve(union_parameters.size());
    std::ranges::transform(union_parameters, std::back_inserter(residents),
                           [](const std::string& name) { return TensorId{name}; });
    return residents;
}

std::expected<void, ContractError> validate_residents(const ExecutionContract& contract,
                                                      std::span<const TensorId> residents) {
    const std::vector<TensorId> expected = resident_parameters(contract);
    if (expected.size() != residents.size() || !std::ranges::equal(expected, residents)) {
        return std::unexpected(ContractError::ResidentMismatch);
    }
    return {};
}

} // namespace skippy

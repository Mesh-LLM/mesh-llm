#include "execution_contract.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

using skippy::AuxiliaryProfile;
using skippy::ContractError;
using skippy::ExecutionContract;
using skippy::LayerRange;
using skippy::TensorId;

constexpr LayerRange kStage{0, 6};

ExecutionContract contract(std::vector<std::string> decoder = {"a", "b"}) {
    return ExecutionContract{
        .version = ExecutionContract::kSupportedVersion,
        .range = kStage,
        .decoder = std::move(decoder),
        .decoder_embeddings = std::nullopt,
        .auxiliary = {},
    };
}

} // namespace

TEST(ExecutionContract, AcceptsASortedUniqueDecoder) {
    EXPECT_FALSE(skippy::validate(contract()).has_value());
}

TEST(ExecutionContract, RejectsAnotherVersion) {
    ExecutionContract value = contract();
    value.version = 2;
    EXPECT_EQ(*skippy::validate(value), ContractError::UnsupportedVersion);
}

TEST(ExecutionContract, RejectsAnEmptyParameterName) {
    EXPECT_EQ(*skippy::validate(contract({"a", ""})), ContractError::EmptyParameter);
}

TEST(ExecutionContract, RejectsAnEmptyProfile) {
    EXPECT_EQ(*skippy::validate(contract({})), ContractError::EmptyParameter);
}

TEST(ExecutionContract, RejectsUnsortedParameters) {
    EXPECT_EQ(*skippy::validate(contract({"b", "a"})), ContractError::ParametersNotSorted);
}

TEST(ExecutionContract, RejectsDuplicateParameters) {
    EXPECT_EQ(*skippy::validate(contract({"a", "a"})), ContractError::DuplicateParameter);
}

TEST(ExecutionContract, RejectsAContractForAnotherStage) {
    EXPECT_EQ(*skippy::validate(contract(), LayerRange{6, 12}), ContractError::RangeMismatch);
    EXPECT_FALSE(skippy::validate(contract(), kStage).has_value());
}

TEST(ExecutionContract, RejectsNonContiguousAuxiliaryOffsets) {
    ExecutionContract value = contract();
    value.auxiliary = {{.offset = 1, .parameters = {"c"}}};
    EXPECT_EQ(*skippy::validate(value), ContractError::AuxiliaryOffsetNotContiguous);
}

TEST(ExecutionContract, AcceptsContiguousAuxiliaryOffsets) {
    ExecutionContract value = contract();
    value.auxiliary = {{.offset = 0, .parameters = {"c"}}, {.offset = 1, .parameters = {"d", "e"}}};
    EXPECT_FALSE(skippy::validate(value).has_value());
}

TEST(ExecutionContract, DecoderProfileIsTheDefault) {
    const ExecutionContract value = contract();
    const auto parameters = skippy::parameters_for(value, false, 0, false);
    ASSERT_TRUE(parameters.has_value());
    EXPECT_EQ(parameters->size(), 2U);
    EXPECT_EQ((*parameters)[0], "a");
}

TEST(ExecutionContract, EmbeddingInputNeedsAnEmbeddingProfile) {
    EXPECT_EQ(skippy::parameters_for(contract(), false, 0, true).error(),
              ContractError::EmbeddingProfileMissing);
}

TEST(ExecutionContract, EmbeddingInputUsesTheEmbeddingProfile) {
    ExecutionContract value = contract();
    value.decoder_embeddings = std::vector<std::string>{"raw"};
    const auto parameters = skippy::parameters_for(value, false, 0, true);
    ASSERT_TRUE(parameters.has_value());
    EXPECT_EQ((*parameters)[0], "raw");
}

TEST(ExecutionContract, AuxiliaryProfileIsSelectedByOffset) {
    ExecutionContract value = contract();
    value.auxiliary = {{.offset = 0, .parameters = {"c"}}, {.offset = 1, .parameters = {"d"}}};

    const auto first = skippy::parameters_for(value, true, 0, false);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ((*first)[0], "c");

    const auto second = skippy::parameters_for(value, true, 1, false);
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ((*second)[0], "d");

    EXPECT_EQ(skippy::parameters_for(value, true, 2, false).error(),
              ContractError::AuxiliaryProfileMissing);
    EXPECT_EQ(skippy::parameters_for(value, true, -1, false).error(),
              ContractError::AuxiliaryProfileMissing);
}

TEST(ExecutionContract, ResidentsAreTheSortedUnionOfEveryProfile) {
    ExecutionContract value = contract({"b", "d"});
    value.decoder_embeddings = std::vector<std::string>{"a"};
    value.auxiliary = {{.offset = 0, .parameters = {"c"}}, {.offset = 1, .parameters = {"e", "d"}}};

    const std::vector<TensorId> residents = skippy::resident_parameters(value);
    const std::vector<std::string> names = [&] {
        std::vector<std::string> out;
        for (const TensorId& resident : residents)
            out.push_back(resident.value);
        return out;
    }();
    EXPECT_EQ(names, (std::vector<std::string>{"a", "b", "c", "d", "e"}));
}

TEST(ExecutionContract, ResidentsMustMatchExactly) {
    const std::vector<TensorId> matching{TensorId{"a"}, TensorId{"b"}};
    EXPECT_FALSE(skippy::validate_residents(contract(), matching).has_value());

    const std::vector<TensorId> missing{TensorId{"a"}};
    EXPECT_EQ(*skippy::validate_residents(contract(), missing), ContractError::ResidentMismatch);

    const std::vector<TensorId> swapped{TensorId{"b"}, TensorId{"a"}};
    EXPECT_EQ(*skippy::validate_residents(contract(), swapped), ContractError::ResidentMismatch);
}

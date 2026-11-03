// Property tests: structure with a law (TYPE-RULES §6). The fuzzers drive
// bytes the facade did not produce; these drive generated structure against
// invariants — every split partitions, every clone equals its original and
// outlives it, every split-built chain validates and every broken one is
// rejected by name, normalization makes unit vectors, batches place every
// sequence. A property that fails is a finding: fix the code, keep the law.
#include "batch.h"
#include "normalize.h"
#include "stage_plan.h"
#include "types.h"

#include <gtest/gtest.h>

#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using skippy::LayerRange;
using skippy::OwnedProfile;
using skippy::OwnedValue;
using skippy::PackageId;
using skippy::PlanId;
using skippy::ProfileId;
using skippy::StagePlan;
using skippy::ValueBinding;
using skippy::ValueIdentity;

skippy::StagePlan plan_with(const std::vector<std::pair<std::int32_t, std::int32_t>>& specs,
                            std::int32_t layers, LayerRange range) {
    StagePlan plan;
    plan.package = PackageId{"pkg"};
    plan.id = PlanId{"plan"};
    plan.layer_count = layers;
    plan.range = range;
    for (std::size_t index = 0; index < specs.size(); ++index) {
        OwnedProfile profile;
        profile.id = ProfileId{"p" + std::to_string(index)};
        profile.slice_range = range;
        for (std::int32_t value = 0; value < specs[index].first; ++value) {
            profile.activation_imports.push_back(OwnedValue{
                ValueIdentity{"in" + std::to_string(index) + "-" + std::to_string(value)},
                ValueBinding{"bin" + std::to_string(value)}});
        }
        for (std::int32_t value = 0; value < specs[index].second; ++value) {
            profile.activation_exports.push_back(OwnedValue{
                ValueIdentity{"out" + std::to_string(index) + "-" + std::to_string(value)},
                ValueBinding{"bout" + std::to_string(value)}});
        }
        plan.append(std::move(profile));
    }
    return plan;
}

std::vector<StagePlan> clone_chain(const std::vector<StagePlan>& chain) {
    std::vector<StagePlan> clones;
    clones.reserve(chain.size());
    for (const StagePlan& plan : chain) {
        clones.push_back(plan.clone());
    }
    return clones;
}

std::vector<std::pair<std::int32_t, std::int32_t>>
clamped_specs(const std::vector<std::pair<std::int32_t, std::int32_t>>& raw) {
    std::vector<std::pair<std::int32_t, std::int32_t>> specs;
    for (const auto& [imports, exports] : raw) {
        if (specs.size() >= 6U) break;
        specs.emplace_back(std::abs(imports) % 12, std::abs(exports) % 12);
    }
    if (specs.empty()) specs.emplace_back(0, 0);
    return specs;
}

} // namespace

RC_GTEST_PROP(SplitLayers, PartitionsContiguouslyAndNearEqually,
              (std::int32_t raw_layers, std::uint32_t raw_stages)) {
    const auto layers = 1 + std::abs(raw_layers) % 512;
    const auto stages = 1 + raw_stages % static_cast<std::uint32_t>(layers);
    const auto ranges = skippy::split_layers(static_cast<std::int32_t>(layers), stages);
    RC_ASSERT(ranges.has_value());
    std::int32_t sum = 0;
    std::int32_t lowest = ranges->front().count();
    std::int32_t highest = 0;
    std::int32_t expected_start = 0;
    for (const LayerRange& range : *ranges) {
        RC_ASSERT(range.start == expected_start);
        RC_ASSERT(!range.empty());
        sum += range.count();
        lowest = std::min(lowest, range.count());
        highest = std::max(highest, range.count());
        expected_start = range.end;
    }
    RC_ASSERT(sum == static_cast<std::int32_t>(layers));
    RC_ASSERT(expected_start == static_cast<std::int32_t>(layers));
    RC_ASSERT(highest - lowest <= 1);
}

RC_GTEST_PROP(SplitLayers, RejectsImpossibleCounts,
              (std::int32_t raw_layers, std::uint32_t raw_stages)) {
    const auto layers = std::abs(raw_layers) % 512;
    RC_ASSERT(!skippy::split_layers(static_cast<std::int32_t>(layers), 0U).has_value());
    RC_ASSERT(
        !skippy::split_layers(-static_cast<std::int32_t>(layers) - 1, raw_stages + 1U).has_value());
    if (layers > 0) {
        const auto too_many = static_cast<std::uint32_t>(layers) + 1U + raw_stages;
        RC_ASSERT(!skippy::split_layers(static_cast<std::int32_t>(layers), too_many).has_value());
    }
}

RC_GTEST_PROP(StagePlanClone, PreservesViewsAndSpansAndNeverGrows,
              (const std::vector<std::pair<std::int32_t, std::int32_t>>& raw_specs)) {
    const auto specs = clamped_specs(raw_specs);
    StagePlan plan = plan_with(specs, 4, LayerRange{1, 3});

    std::vector<std::vector<std::string>> original_imports;
    for (const auto& profile : plan.profiles) {
        std::vector<std::string> views;
        for (const auto& imported : plan.activation_imports_of(profile)) {
            views.emplace_back(plan.view(imported.identity.slice));
            views.emplace_back(plan.view(imported.binding.slice));
        }
        original_imports.push_back(std::move(views));
    }

    const StagePlan clone = plan.clone();
    RC_ASSERT(clone.identity_arena.size() == plan.identity_arena.size());
    RC_ASSERT(clone.activation_imports.size() <= plan.activation_imports.size());
    RC_ASSERT(clone.profiles.size() == plan.profiles.size());
    for (std::size_t index = 0; index < clone.profiles.size(); ++index) {
        const auto& profile = clone.profiles[index];
        std::vector<std::string> views;
        for (const auto& imported : clone.activation_imports_of(profile)) {
            views.emplace_back(clone.view(imported.identity.slice));
            views.emplace_back(clone.view(imported.binding.slice));
        }
        RC_ASSERT(views == original_imports[index]);
    }
}

RC_GTEST_PROP(StagePlanClone, SurvivesOriginalDestruction,
              (const std::vector<std::pair<std::int32_t, std::int32_t>>& raw_specs)) {
    const auto specs = clamped_specs(raw_specs);
    std::vector<std::vector<std::string>> captured;
    StagePlan clone;
    {
        // A lone stage must partition [0, layer_count) on its own for the
        // closing validate_chain call to accept it.
        StagePlan plan = plan_with(specs, 4, LayerRange{0, 4});
        for (const auto& profile : plan.profiles) {
            std::vector<std::string> views;
            for (const auto& imported : plan.activation_imports_of(profile)) {
                views.emplace_back(plan.view(imported.identity.slice));
            }
            captured.push_back(std::move(views));
        }
        clone = plan.clone();
    }
    for (std::size_t index = 0; index < clone.profiles.size(); ++index) {
        std::vector<std::string> views;
        for (const auto& imported : clone.activation_imports_of(clone.profiles[index])) {
            views.emplace_back(clone.view(imported.identity.slice));
        }
        RC_ASSERT(views == captured[index]);
    }
    const skippy::StagePlan chain[] = {std::move(clone)};
    RC_ASSERT(skippy::validate_chain(chain).has_value());
}

RC_GTEST_PROP(ValidateChain, AcceptsEverySplitPartitionAndNamesEveryBreak,
              (std::int32_t raw_layers, std::uint32_t raw_stages)) {
    const auto layers = 1 + std::abs(raw_layers) % 64;
    const auto stages = 1 + raw_stages % static_cast<std::uint32_t>(layers);
    const auto ranges = skippy::split_layers(static_cast<std::int32_t>(layers), stages);
    RC_ASSERT(ranges.has_value());
    std::vector<StagePlan> chain;
    chain.reserve(ranges->size());
    for (const LayerRange& range : *ranges) {
        chain.push_back(plan_with({{0, 0}}, static_cast<std::int32_t>(layers), range));
    }
    RC_ASSERT(skippy::validate_chain(std::span<const StagePlan>{chain}).has_value());

    if (chain.size() < 2U) return;
    std::vector<StagePlan> gap = clone_chain(chain);
    --gap[1].range.end;
    gap[1].profiles.front().slice_range = gap[1].range;
    RC_ASSERT(!skippy::validate_chain(std::span<const StagePlan>{gap}).has_value());

    std::vector<StagePlan> overlap = clone_chain(chain);
    ++overlap[0].range.end;
    overlap[0].profiles.front().slice_range = overlap[0].range;
    RC_ASSERT(!skippy::validate_chain(std::span<const StagePlan>{overlap}).has_value());

    std::vector<StagePlan> reorder = clone_chain(chain);
    std::swap(reorder[0], reorder[1]);
    const auto rejected = skippy::validate_chain(std::span<const StagePlan>{reorder});
    RC_ASSERT(!rejected.has_value());
    RC_ASSERT(rejected.error() == skippy::PlanError::IncompletePartition);
}

RC_GTEST_PROP(NormalizeEmbedding, UnitNormIdempotentZeroUnchanged,
              (const std::vector<float>& raw)) {
    if (raw.empty()) return;
    std::vector<float> source = raw;
    for (float& value : source) {
        if (!std::isfinite(value)) value = 0.5F;
    }
    std::vector<float> normalized(source.size(), 0.0F);
    skippy::detail::normalize_embedding(source.data(), normalized.data(), source.size());

    bool all_zero = true;
    for (const float value : source) {
        all_zero = all_zero && value == 0.0F;
    }
    if (all_zero) {
        RC_ASSERT(normalized == source);
        return;
    }
    double squared = 0.0;
    for (const float value : normalized) {
        squared += static_cast<double>(value) * static_cast<double>(value);
    }
    RC_ASSERT(std::abs(std::sqrt(squared) - 1.0) < 1e-3);
    std::vector<float> again(normalized.size(), 0.0F);
    skippy::detail::normalize_embedding(normalized.data(), again.data(), normalized.size());
    for (std::size_t index = 0; index < again.size(); ++index) {
        RC_ASSERT(std::abs(again[index] - normalized[index]) < 1e-6F);
    }
}

namespace {

struct TestBatch {
    std::int32_t n_tokens = 0;
    std::int32_t token[128] = {};
    std::int32_t pos[128] = {};
    std::int32_t n_seq_id[128] = {};
    std::int32_t seq_id[128][4] = {};
    std::int8_t logits[128] = {};
};
static_assert(skippy::detail::TokenBatch<TestBatch>, "the test batch satisfies the concept");

} // namespace

RC_GTEST_PROP(FillEmbeddingBatches, PositionsPerSequenceAndSeqIdIsRow,
              (const std::vector<std::vector<std::int32_t>>& raw_sequences)) {
    TestBatch batch;
    std::vector<std::vector<skippy::Token>> tokens;
    std::vector<std::span<const skippy::Token>> inputs;
    std::int32_t expected_total = 0;
    for (std::size_t sequence = 0; sequence < std::min<std::size_t>(raw_sequences.size(), 6U);
         ++sequence) {
        std::vector<skippy::Token> one;
        for (std::size_t index = 0;
             index < std::min<std::size_t>(raw_sequences[sequence].size(), 8U); ++index) {
            one.push_back(static_cast<skippy::Token>((sequence + index) % 32000));
            ++expected_total;
        }
        tokens.push_back(std::move(one));
    }
    if (tokens.empty()) return;
    for (const auto& one : tokens) {
        inputs.emplace_back(one);
    }
    const auto total = skippy::detail::fill_embedding_batches(batch, inputs);
    RC_ASSERT(total == expected_total);
    RC_ASSERT(batch.n_tokens == expected_total);
    std::int32_t at = 0;
    for (std::size_t sequence = 0; sequence < tokens.size(); ++sequence) {
        for (std::size_t index = 0; index < tokens[sequence].size(); ++index) {
            RC_ASSERT(batch.pos[at] == static_cast<std::int32_t>(index));
            RC_ASSERT(batch.seq_id[at][0] == static_cast<std::int32_t>(sequence));
            RC_ASSERT(batch.n_seq_id[at] == 1);
            RC_ASSERT(batch.logits[at] == 1);
            RC_ASSERT(batch.token[at] == static_cast<std::int32_t>(tokens[sequence][index]));
            ++at;
        }
    }
    RC_ASSERT(at == total);
}

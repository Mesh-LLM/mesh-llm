#pragma once

// The rerank workload, generic over the runtime's backend.
//
// Layers, top down:
//   rerank()                   this file: workload checks, prompt assembly, sequencing
//   Backend::SessionResource   tokenises text and scores a token sequence natively
//
// Prompt assembly is pure: the backend supplies the model's template (if any)
// and the EOS/SEP piece text once, as strings, and never sees the query or

#include "rerank_template.h"
#include "runtime.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace skippy {

// How a model wants query and document composed. With a template, that is the
// whole story; without one, the original ABI's fallback applies:
//   query + [EOS piece] + [SEP piece] + document
// where each piece is present only if the vocabulary appends that token.
struct RerankPromptFormat {
    std::optional<std::string_view> prompt_template =
        std::nullopt;           // model-authored, uses {query} / {document}
    std::string_view eos_piece; // empty when the vocab does not add EOS
    std::string_view sep_piece; // empty when the vocab does not add SEP
};

[[nodiscard]] constexpr std::string build_rerank_prompt(const RerankPromptFormat& format,
                                                        std::string_view query,
                                                        std::string_view document) {
    if (format.prompt_template) {
        return detail::render_rerank_template(*format.prompt_template, query, document);
    }
    std::string prompt;
    prompt.reserve(query.size() + format.eos_piece.size() + format.sep_piece.size() +
                   document.size());
    prompt.append(query).append(format.eos_piece).append(format.sep_piece).append(document);
    return prompt;
}

// What a backend's session must provide for rerank() to compile against it.
template <typename S> using TokenizeResult = std::expected<std::span<const Token>, WorkloadError>;

template <typename S> using ScoreManyResult = std::expected<std::span<const float>, WorkloadError>;

template <typename S>
concept RerankSession =
    requires(S& session, const S& const_session, std::string_view text,
             std::span<const Token> tokens, std::span<const std::span<const Token>> inputs) {
        { const_session.workload() } -> std::convertible_to<const WorkloadInfo&>;
        { const_session.rerank_prompt_format() } -> std::convertible_to<RerankPromptFormat>;
        // Tokenises `text`; the span borrows session-owned scratch that lives
        // until the next tokenize() on the same session.
        { session.tokenize(text) } -> std::same_as<TokenizeResult<S>>;
        // Decodes `tokens` and returns the single classifier output.
        { session.score(tokens) } -> std::same_as<std::expected<float, WorkloadError>>;
        // Decodes several sequences in one native pass; one score per row.
        { session.score_many(inputs) } -> std::same_as<ScoreManyResult<S>>;
    };

struct RerankResult {
    float score;
    std::size_t token_count; // tokens the composed prompt produced
};

struct RerankBatchResult {
    std::size_t count;       // scores written, one per document
    std::size_t token_count; // tokens across every composed prompt
};

// Scores one query/document pair. Checks that need no native call run first,
// so a bad request never reaches the backend.
template <RuntimeBackend Backend>
    requires RerankSession<typename Backend::SessionResource>
[[nodiscard]] std::expected<RerankResult, WorkloadError>
rerank(NativeRuntime<Backend>& runtime, const Session& handle, std::string_view query,
       std::string_view document) {
    using Result = std::expected<RerankResult, WorkloadError>;

    const auto run = [&](Backend::SessionResource& session) -> Result {
        const WorkloadInfo& workload = std::as_const(session).workload();
        if (workload.kind != WorkloadKind::Rerank) {
            return std::unexpected(WorkloadError::UnsupportedWorkload);
        }
        if (workload.classifier_outputs != 1) {
            return std::unexpected(WorkloadError::UnsupportedClassifierCount);
        }

        const std::string prompt =
            build_rerank_prompt(std::as_const(session).rerank_prompt_format(), query, document);

        return session.tokenize(prompt).and_then([&](std::span<const Token> tokens) -> Result {
            if (tokens.empty()) {
                return std::unexpected(WorkloadError::TokenizationFailed);
            }
            return session.score(tokens).transform([&](float score) {
                return RerankResult{.score = score, .token_count = tokens.size()};
            });
        });
    };

    return runtime.with_session(handle, run)
        .transform_error([](RuntimeError) { return WorkloadError::InvalidSession; })
        .and_then(std::identity{});
}

// Scores one query against several documents in a single native pass.
// Prompts are composed and tokenised per document, then decoded together, so
// the per-decode fixed cost is paid once. `scores` must hold one slot per
// document; requires `SessionConfig::sequences >= documents.size()`.
template <RuntimeBackend Backend>
    requires RerankSession<typename Backend::SessionResource>
[[nodiscard]] std::expected<RerankBatchResult, WorkloadError>
rerank_batch(NativeRuntime<Backend>& runtime, const Session& handle, std::string_view query,
             std::span<const std::string_view> documents, std::span<float> scores) {
    using Result = std::expected<RerankBatchResult, WorkloadError>;

    if (documents.empty()) {
        return std::unexpected(WorkloadError::EmptyInput);
    }
    if (scores.size() < documents.size()) {
        return std::unexpected(WorkloadError::OutputTooSmall);
    }

    const auto run = [&](Backend::SessionResource& session) -> Result {
        const WorkloadInfo& workload = std::as_const(session).workload();
        if (workload.kind != WorkloadKind::Rerank) {
            return std::unexpected(WorkloadError::UnsupportedWorkload);
        }
        if (workload.classifier_outputs != 1) {
            return std::unexpected(WorkloadError::UnsupportedClassifierCount);
        }

        const RerankPromptFormat format = std::as_const(session).rerank_prompt_format();
        // tokenize() borrows session scratch, so each document's tokens are
        // copied out before the next one overwrites them.
        std::vector<std::vector<Token>> owned;
        owned.reserve(documents.size());
        std::size_t token_count = 0;
        for (const std::string_view document : documents) {
            const std::string prompt = build_rerank_prompt(format, query, document);
            const auto tokens = session.tokenize(prompt);
            if (!tokens) {
                return std::unexpected(tokens.error());
            }
            if (tokens->empty()) {
                return std::unexpected(WorkloadError::TokenizationFailed);
            }
            token_count += tokens->size();
            owned.emplace_back(tokens->begin(), tokens->end());
        }

        std::vector<std::span<const Token>> views;
        views.reserve(owned.size());
        for (const auto& row : owned) {
            views.emplace_back(row);
        }

        return session.score_many(views).and_then([&](std::span<const float> native) -> Result {
            if (native.size() != documents.size()) {
                return std::unexpected(WorkloadError::DimensionMismatch);
            }
            std::ranges::copy(native, scores.begin());
            return RerankBatchResult{.count = documents.size(), .token_count = token_count};
        });
    };

    return runtime.with_session(handle, run)
        .transform_error([](RuntimeError) { return WorkloadError::InvalidSession; })
        .and_then(std::identity{});
}

} // namespace skippy

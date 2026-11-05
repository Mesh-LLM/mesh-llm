#pragma once

#include "runtime.h"
#include "types.h"
#include "workload.h"
#include "workload_error.h"

#include <concepts>
#include <expected>
#include <functional>
#include <span>
#include <utility>

namespace skippy {

template <typename S>
concept EncoderSession =
    requires(S& session, const S& const_session, std::span<const Token> tokens) {
        { const_session.workload() } -> std::convertible_to<const WorkloadInfo&>;
        // Runs the encoder over `tokens`, leaving its output in the session's
        // context for a later decode; returns the token the decoder starts from.
        { session.encode(tokens) } -> std::same_as<std::expected<Token, WorkloadFailure>>;
    };

template <RuntimeBackend Backend>
    requires EncoderSession<typename Backend::SessionResource>
[[nodiscard]] std::expected<Token, WorkloadFailure> encode_prompt(NativeRuntime<Backend>& runtime,
                                                                  const Session& handle,
                                                                  std::span<const Token> tokens) {
    using Result = std::expected<Token, WorkloadFailure>;

    if (tokens.empty()) {
        return std::unexpected(WorkloadFailure::code_only(WorkloadError::EmptyInput));
    }

    const auto run = [&](Backend::SessionResource& session) -> Result {
        if (std::as_const(session).workload().kind != WorkloadKind::EncoderDecoder) {
            return std::unexpected(WorkloadFailure::code_only(WorkloadError::UnsupportedWorkload));
        }
        return session.encode(tokens);
    };

    return runtime.with_session(handle, run)
        .transform_error(
            [](RuntimeError) { return WorkloadFailure::code_only(WorkloadError::InvalidSession); })
        .and_then(std::identity{});
}

} // namespace skippy
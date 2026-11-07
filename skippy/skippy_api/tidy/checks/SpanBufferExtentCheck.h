#pragma once

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::skippy {

/// A std::span over a fixed-extent buffer takes its extent from the buffer,
/// not from a runtime count. `std::span<float>(scores, ids.size())` over
/// `float scores[256]` claims whatever the count says: when the count can
/// exceed the buffer, the span lies its way past every downstream size check
/// and steers a write past the buffer (PR #2907 finding #8 — the bridge's
/// 256-float score buffer spanned by the selection size). The honest forms
/// are the single-argument construction (CTAD takes the buffer's extent) and
/// a count that is an integer constant within the extent; a runtime count
/// must be bounded before the span exists.
///
/// Options: none.
class SpanBufferExtentCheck : public ClangTidyCheck {
public:
    SpanBufferExtentCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context) {}
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

} // namespace clang::tidy::skippy

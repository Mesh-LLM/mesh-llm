#pragma once

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::skippy {

/// A `(void)x` or `static_cast<void>(x)` on a local whose only other use is
/// its own definition hides a dead computation from -Wunused and the
/// analyzer. The counts in materialize_profile and the old clone() were this.
///
/// Would have flagged: stage_planner.cpp:360-363, stage_plan.cpp:332-335.
class NoVoidSilencingCheck : public ClangTidyCheck {
public:
    NoVoidSilencingCheck(StringRef Name, ClangTidyContext* Context) : ClangTidyCheck(Name, Context) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
};

} // namespace clang::tidy::skippy

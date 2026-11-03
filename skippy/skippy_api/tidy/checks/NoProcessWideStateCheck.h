#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include <string>
#include <vector>

namespace clang::tidy::skippy {

/// AGENTS "State": avoid process-wide state. Flags mutable namespace-scope
/// variables, static data members and function-local statics, except
/// constexpr/const ones, the listed types, and statics inside the listed
/// functions.
///
/// Would have flagged: memoised_file_result's static index (P3-5).
///
/// Options:
///   AllowedTypes       (default "::std::once_flag")
///   AllowedInFunctions (default "skippy::detail::next_generation")
class NoProcessWideStateCheck : public ClangTidyCheck {
public:
    NoProcessWideStateCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::vector<StringRef> AllowedTypes_;
    std::vector<StringRef> AllowedInFunctions_;
};

} // namespace clang::tidy::skippy

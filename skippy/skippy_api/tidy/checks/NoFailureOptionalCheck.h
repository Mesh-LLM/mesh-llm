#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Regex.h"

#include <string>

namespace clang::tidy::skippy {

/// ENFORCEMENT.md §4 "Failure-only results have one spelling, and it chains":
/// std::expected<void, E> composes with and_then; std::optional<E>-as-failure
/// cannot (optional::and_then runs on the engaged value, i.e. on the
/// failure). This check is skippy-no-expected-void flipped to the new
/// convention; skippy-optional-error-polarity stays for the transition.
///
/// Sunsets when one failure spelling exists tree-wide — then keep the
/// survivor, it is free.
///
/// Options:
///   ErrorTypeRegex (default "(Error|Failure)$") which payload types are failures.
///
/// Would have caught: the ~25 optional<E>-as-failure declarations flipped in
/// the one-commit migration (P2-1's list, plus stage_plan/execution_contract/
/// stage_chain/stage_planner/runtime.h).
class NoFailureOptionalCheck : public ClangTidyCheck {
public:
    NoFailureOptionalCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context),
          ErrorTypeRegex_(Options.get("ErrorTypeRegex", "(Error|Failure)$")),
          ErrorType_(ErrorTypeRegex_) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "ErrorTypeRegex", ErrorTypeRegex_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string ErrorTypeRegex_;
    llvm::Regex ErrorType_;
    llvm::DenseSet<unsigned> Reported_;
};

} // namespace clang::tidy::skippy

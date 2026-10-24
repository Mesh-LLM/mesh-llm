#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/ADT/DenseSet.h"

namespace clang::tidy::skippy {

/// ENFORCEMENT.md §4 "Results compose; they are not inspected": `.value()` is
/// the only throwing accessor on std::expected and std::optional, and it has
/// no place in this tree. Consume results with and_then / transform /
/// or_else / transform_error / value_or, or guard with `if (!result)` and
/// read `*result`.
///
/// Never sunsets.
///
/// Would have caught: the six `.value()` call sites removed in the
/// compose-rule migration (model_inventory.h, ffi_vs_api_benchmark.cpp,
/// three test files).
class NoThrowingAccessorCheck : public ClangTidyCheck {
public:
    NoThrowingAccessorCheck(StringRef Name, ClangTidyContext* Context) : ClangTidyCheck(Name, Context) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    llvm::DenseSet<unsigned> Reported_;
};

} // namespace clang::tidy::skippy

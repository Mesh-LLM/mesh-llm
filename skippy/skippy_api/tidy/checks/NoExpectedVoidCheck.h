#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/ADT/DenseSet.h"

namespace clang::tidy::skippy {

/// AGENTS "Types and interfaces": never std::expected<void, E>. A failure-only
/// result is Fallible<E> (or optional<E> while that migration runs).
///
/// Would have flagged: the 15 declarations listed under P2-1.
class NoExpectedVoidCheck : public ClangTidyCheck {
public:
    NoExpectedVoidCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    llvm::DenseSet<unsigned> Reported_;
};

} // namespace clang::tidy::skippy

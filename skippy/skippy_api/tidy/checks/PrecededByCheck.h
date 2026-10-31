#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include <string>

namespace clang::tidy::skippy {

/// A call protocol: every call to `Call` must be immediately preceded, in the
/// same block, by a statement that calls `Before`.
///
/// Configured here as llama_memory_clear / llama_synchronize (AGENTS "llama
/// returns before the backend finishes"). Would have flagged P1-1 at
/// backend.cpp:224 and :316.
///
/// Options:
///   Call   (default "llama_memory_clear")
///   Before (default "llama_synchronize")
class PrecededByCheck : public ClangTidyCheck {
public:
    PrecededByCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context), Call_(Options.get("Call", "llama_memory_clear")),
          Before_(Options.get("Before", "llama_synchronize")) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "Call", Call_);
        Options.store(Opts, "Before", Before_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string Call_;
    std::string Before_;
};

} // namespace clang::tidy::skippy

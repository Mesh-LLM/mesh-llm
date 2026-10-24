#pragma once

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::skippy {

/// House rule: no for(;;)/while/do loops; ranges algorithms and adapters
/// instead. An index loop at an index-based ABI is allowed with a
/// `// NOLINT(skippy-no-raw-loops): <reason>` on the line.
///
/// Options:
///   FlagRangeFor (bool, default false) also flag range-for statements.
class NoRawLoopsCheck : public ClangTidyCheck {
public:
    NoRawLoopsCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context), FlagRangeFor_(Options.get("FlagRangeFor", false)) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override { Options.store(Opts, "FlagRangeFor", FlagRangeFor_); }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    bool FlagRangeFor_;
};

} // namespace clang::tidy::skippy

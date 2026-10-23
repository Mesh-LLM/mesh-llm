#pragma once

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::skippy {

/// AGENTS trap: "an allocation that must happen inside a failure handler gets
/// its own catch (...)". Flags allocating expressions inside a bad_alloc or
/// catch-all handler of a noexcept function (run_pass, guarded_message).
///
/// Would have flagged: P2-11 (`std::string(where) + ": out of memory"` and
/// rust::String::lossy inside guarded_message's bad_alloc arm).
///
/// Options:
///   OnlyInNoexcept (bool, default true)
class NoAllocInOomHandlerCheck : public ClangTidyCheck {
public:
    NoAllocInOomHandlerCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context), OnlyInNoexcept_(Options.get("OnlyInNoexcept", true)) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override { Options.store(Opts, "OnlyInNoexcept", OnlyInNoexcept_); }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    bool OnlyInNoexcept_;
};

} // namespace clang::tidy::skippy

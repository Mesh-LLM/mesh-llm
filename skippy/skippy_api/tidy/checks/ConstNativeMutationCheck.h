#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include <string>

namespace clang::tidy::skippy {

/// AGENTS "const means what it says": a const method that hands a member to a
/// native function through a non-const pointer mutates state behind const
/// (llama's getters take a mutable context because they synchronise).
///
/// Options:
///   NativeRegex (default "^::(llama_|ggml_|skippy_)")
class ConstNativeMutationCheck : public ClangTidyCheck {
public:
    ConstNativeMutationCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context), NativeRegex_(Options.get("NativeRegex", "^::(llama_|ggml_|skippy_)")) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override { Options.store(Opts, "NativeRegex", NativeRegex_); }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string NativeRegex_;
};

} // namespace clang::tidy::skippy

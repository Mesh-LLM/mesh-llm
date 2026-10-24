#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>
#include <vector>

namespace clang::tidy::skippy {

/// AGENTS "State": a mutex is never held across a call into llama. Within the
/// block where a lock guard lives, every later statement is scanned for a
/// native call until the guard is explicitly unlocked.
///
/// Options:
///   LockTypes   (default scoped_lock;lock_guard;unique_lock;shared_lock)
///   NativeRegex (default "^::(llama_|ggml_|skippy_)")
class NoLockAcrossNativeCallCheck : public ClangTidyCheck {
public:
    NoLockAcrossNativeCallCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::vector<StringRef> LockTypes_;
    std::string NativeRegex_;
};

} // namespace clang::tidy::skippy

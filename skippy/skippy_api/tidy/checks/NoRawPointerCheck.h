#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>

namespace clang::tidy::skippy {

/// ENFORCEMENT.md §4 "Views over raw pointers, everywhere we can": a raw
/// pointer may appear in exactly two places — inside an owning cell
/// (unique_ptr, AtomicOwned) and in the C struct the ABI defines, produced by
/// one native() function. Fields, locals, parameters and returns are spans,
/// string_views, references or owners. Function pointers are not data
/// pointers and pass. Never sunsets.
///
/// Options:
///   AllowedNames (default "(^|::)(native|argv|optarg)$") declarations
///     exempt by (un)qualified name — the ABI's native() and the C runtime's
///     argv.
///   AllowedFiles (default "^$") files exempt wholesale (ABI POD headers).
///
/// Would have caught: Reader's three cursor pointers (base_/pos_/end_, now a
/// span + offset) and the nullable RerankPromptSource* borrow.
class NoRawPointerCheck : public ClangTidyCheck {
public:
    NoRawPointerCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context),
          AllowedNames_(Options.get("AllowedNames", "(^|::)(native|argv|optarg)$")),
          AllowedFiles_(Options.get("AllowedFiles", "^$")),
          Names_(AllowedNames_), Files_(AllowedFiles_) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "AllowedNames", AllowedNames_);
        Options.store(Opts, "AllowedFiles", AllowedFiles_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string AllowedNames_;
    std::string AllowedFiles_;
    llvm::Regex Names_;
    llvm::Regex Files_;
};

} // namespace clang::tidy::skippy

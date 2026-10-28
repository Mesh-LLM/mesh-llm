#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>

namespace clang::tidy::skippy {

/// AGENTS trap "Do not trust the compiler to catch a polarity flip": an
/// std::optional<Error> converted to bool reads backwards from
/// expected<void, Error>. Require .has_value() (or Fallible<E>).
///
/// Options:
///   ErrorTypeRegex (default "(Error|Failure)$") which payload types are failures.
class OptionalErrorPolarityCheck : public ClangTidyCheck {
public:
    OptionalErrorPolarityCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context),
          ErrorTypeRegex_(Options.get("ErrorTypeRegex", "(Error|Failure)$")),
          ErrorType_(ErrorTypeRegex_) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "ErrorTypeRegex", ErrorTypeRegex_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string ErrorTypeRegex_;
    llvm::Regex ErrorType_;
};

} // namespace clang::tidy::skippy

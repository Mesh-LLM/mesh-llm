#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include <string>

namespace clang::tidy::skippy {

/// AGENTS trap: rust::String(const std::string&) throws std::invalid_argument
/// on non-UTF-8. Native text crosses the bridge through rust::String::lossy;
/// only string literals and the listed ASCII producers may use the throwing
/// constructor.
///
/// Would have flagged: every rust::String(message) in bridge.cpp before 136020b3.
///
/// Options:
///   AsciiProducers (regex, default "::hex_of$")
class RustStringLossyCheck : public ClangTidyCheck {
public:
    RustStringLossyCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context),
          AsciiProducers_(Options.get("AsciiProducers", "::hex_of$")) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "AsciiProducers", AsciiProducers_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string AsciiProducers_;
};

} // namespace clang::tidy::skippy

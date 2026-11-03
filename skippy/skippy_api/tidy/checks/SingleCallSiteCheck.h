#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include <string>
#include <vector>

namespace clang::tidy::skippy {

/// AGENTS trap "A native getter with a non-zero default is not a presence
/// test": functions whose raw value needs interpretation (n_cls_out defaults
/// to 1; n_ctx is padded) may be called from exactly the listed callers, which
/// are the constructors of the types that record what the value means.
///
/// Would have flagged: P0-1/P0-B (`llama_model_n_cls_out` in load_model) and
/// P0-A (`llama_n_ctx` in ~LlamaSession).
///
/// Options:
///   Functions      ';'-separated unqualified names
///   AllowedCallers ';'-separated qualified caller names
///   Reason         appended to the diagnostic
class SingleCallSiteCheck : public ClangTidyCheck {
public:
    SingleCallSiteCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::vector<StringRef> Functions_;
    std::vector<StringRef> AllowedCallers_;
    std::string Reason_;
};

} // namespace clang::tidy::skippy

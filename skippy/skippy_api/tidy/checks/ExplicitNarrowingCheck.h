#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>
#include <vector>

namespace clang::tidy::skippy {

/// AGENTS "No silent widening or narrowing": an explicit integer cast that can
/// lose value (narrower width, or a signedness change the width cannot
/// absorb) is allowed only inside the listed functions/files, when the source
/// is a constant that fits, or when the line states its bound
/// ("// bounded by ..."). Everywhere else: narrow<To>() and handle nullopt.
///
/// Would have flagged: arena_put (size_t→uint32_t ×2), the three
/// int64→int32 layer casts in the bridge (P2-2).
///
/// Options:
///   AllowedFunctions (default "skippy::narrow;skippy::checked_mul")
///   AllowedFiles     (regex, default "narrow\\.h$")
///   BoundMarker      (default "bounded by")
///   FlagSignChangeOnWidening (bool, default true) also flag signed→wider unsigned
///                    (a negative value becomes huge); false keeps only width loss
///                    and same-width sign changes
class ExplicitNarrowingCheck : public ClangTidyCheck {
public:
    ExplicitNarrowingCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::vector<StringRef> AllowedFunctions_;
    std::string AllowedFilesRegex_;
    llvm::Regex AllowedFiles_;
    std::string BoundMarker_;
    bool FlagSignChangeOnWidening_;
};

} // namespace clang::tidy::skippy

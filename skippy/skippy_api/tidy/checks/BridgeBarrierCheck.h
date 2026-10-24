#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>
#include <vector>

namespace clang::tidy::skippy {

/// AGENTS trap "An FFI result shape guarded<T> cannot fold is an unguarded
/// method": in the bridge, every noexcept member must be exactly
/// `return <barrier>(...)`. Anything else can unwind into Rust.
///
/// Would have flagged: the eight methods of P0-2 before 64766af1.
///
/// Options:
///   FileRegex (default "bridge\\.cpp$")  which files are the bridge
///   Barriers  (default "guarded;guarded_message")
///   Exempt    (default "")  method names that may stay bare (getters)
class BridgeBarrierCheck : public ClangTidyCheck {
public:
    BridgeBarrierCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string FileRegex_;
    llvm::Regex File_;
    std::vector<StringRef> Barriers_;
    std::vector<StringRef> Exempt_;
};

} // namespace clang::tidy::skippy

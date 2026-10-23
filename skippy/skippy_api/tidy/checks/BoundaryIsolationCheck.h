#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Regex.h"

#include <string>

namespace clang::tidy::skippy {

/// ENFORCEMENT.md §0: the current phase keeps the llama and Rust boundaries
/// where they are. Every native include, native call and native type lives in
/// src/llama, include/llama, tools, benchmarks and skippy-sys/cpp, and nowhere
/// else — so the core compiles without llama by construction, not by the test
/// link happening to fail.
///
/// Options:
///   AllowedDirectories (default "src/llama/;include/llama/;/tools/;/benchmarks/;skippy-sys/cpp/")
///     path fragments; a file whose presumed path contains one is boundary code.
///   NativeRegex (default "(^|::)(llama_|ggml_|skippy_)") qualified names of
///     native free functions and native record types.
///   NativeHeaderRegex (default "^(llama|llama-cpp|ggml|ggml-alloc|ggml-backend|gguf|skippy)\\.h$")
///     basenames of native headers.
///   IgnoredFiles (default "(^|/)\\.deps/") files never reported (vendored
///     trees reached through -I).
///
/// Would have caught: a core header reaching for llama.h directly instead of
/// going through include/llama; the probe code shape (tools/) drifting into
/// src/.
class BoundaryIsolationCheck : public ClangTidyCheck {
public:
    BoundaryIsolationCheck(StringRef Name, ClangTidyContext* Context);
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override { return LangOpts.CPlusPlus; }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override;
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void registerPPCallbacks(const SourceManager& SM, Preprocessor* PP, Preprocessor* ModuleExpanderPP) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

    /// True when the file holding `Path` is boundary code or ignored.
    bool fileIsExempt(StringRef Path) const;

private:
    std::string AllowedDirectories_;
    std::string NativeRegex_;
    std::string NativeHeaderRegex_;
    std::string IgnoredFiles_;
    llvm::SmallVector<std::string, 8> Directories_;
    llvm::Regex Native_;
    llvm::Regex NativeHeader_;
    llvm::Regex Ignored_;
};

} // namespace clang::tidy::skippy

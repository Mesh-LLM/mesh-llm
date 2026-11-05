#include "BoundaryIsolationCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

BoundaryIsolationCheck::BoundaryIsolationCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      AllowedDirectories_(Options
                              .get("AllowedDirectories",
                                   "src/llama/;include/llama/;/tools/;/benchmarks/;skippy-sys/cpp/")
                              .str()),
      NativeRegex_(Options.get("NativeRegex", "(^|::)(llama_|ggml_|skippy_)").str()),
      NativeHeaderRegex_(
          Options
              .get("NativeHeaderRegex",
                   "^(llama|llama-cpp|ggml|ggml-alloc|ggml-backend|gguf|skippy)\\.h$")
              .str()),
      IgnoredFiles_(Options.get("IgnoredFiles", "(^|/)\\.deps/").str()), Native_(NativeRegex_),
      NativeHeader_(NativeHeaderRegex_), Ignored_(IgnoredFiles_) {
    StringRef Rest{AllowedDirectories_};
    while (!Rest.empty()) {
        const auto [Head, Tail] = Rest.split(';');
        if (!Head.empty()) Directories_.push_back(Head.str());
        Rest = Tail;
    }
}

void BoundaryIsolationCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "AllowedDirectories", AllowedDirectories_);
    Options.store(Opts, "NativeRegex", NativeRegex_);
    Options.store(Opts, "NativeHeaderRegex", NativeHeaderRegex_);
    Options.store(Opts, "IgnoredFiles", IgnoredFiles_);
}

bool BoundaryIsolationCheck::fileIsExempt(StringRef Path) const {
    if (Ignored_.match(Path)) return true;
    for (const std::string& Fragment : Directories_)
        if (Path.contains(Fragment)) return true;
    return false;
}

namespace {

/// Splits "a/b/c.h" or "<a/b/c.h>" to the basename the rule names.
StringRef basenameOf(StringRef FileName) {
    StringRef Name = FileName;
    if (Name.ends_with(">") || Name.ends_with("\"")) Name = Name.drop_back();
    const size_t Slash = Name.rfind('/');
    return Slash == StringRef::npos ? Name : Name.substr(Slash + 1);
}

} // namespace

void BoundaryIsolationCheck::registerPPCallbacks(const SourceManager& SM, Preprocessor* PP,
                                                 Preprocessor* ModuleExpanderPP) {
    class IncludeCallback : public PPCallbacks {
    public:
        IncludeCallback(BoundaryIsolationCheck& Check, const SourceManager& SM)
            : Check_(Check), SM_(SM) {}

        void InclusionDirective(SourceLocation HashLoc, const Token& /*IncludeTok*/,
                                StringRef FileName, bool /*IsAngled*/,
                                CharSourceRange /*FilenameRange*/, OptionalFileEntryRef /*File*/,
                                StringRef /*SearchPath*/, StringRef /*RelativePath*/,
                                const Module* /*SuggestedModule*/, bool /*ModuleImported*/,
                                SrcMgr::CharacteristicKind /*FileType*/) override {
            const PresumedLoc Presumed = SM_.getPresumedLoc(SM_.getExpansionLoc(HashLoc));
            if (!Presumed.isValid()) return;
            if (!SM_.isWrittenInMainFile(SM_.getExpansionLoc(HashLoc))) return;
            const StringRef File = Presumed.getFilename();
            if (Check_.fileIsExempt(File)) return;
            if (!Check_.NativeHeader_.match(basenameOf(FileName))) return;
            Check_.diag(HashLoc, "native include '%0' outside the boundary directories "
                                 "(src/llama, include/llama, tools, benchmarks, skippy-sys/cpp); "
                                 "the core must compile without llama by construction "
                                 "(TYPE-RULES.md §0)")
                << FileName;
        }

    private:
        BoundaryIsolationCheck& Check_;
        const SourceManager& SM_;
    };
    PP->addPPCallbacks(std::make_unique<IncludeCallback>(*this, SM));
}

void BoundaryIsolationCheck::registerMatchers(MatchFinder* Finder) {
    const auto NativeFunction = functionDecl(matchesName(NativeRegex_));
    const auto NativeRecord =
        hasUnqualifiedDesugaredType(tagType(hasDeclaration(namedDecl(matchesName(NativeRegex_)))));
    const auto NativeType =
        anyOf(hasType(NativeRecord), hasType(pointerType(pointee(NativeRecord))),
              hasType(referenceType(pointee(NativeRecord))));

    // A native free call outside the boundary directories.
    Finder->addMatcher(callExpr(callee(NativeFunction)).bind("native-call"), this);
    // A native record used in any declared type (variable, field, parameter)
    // outside the boundary directories — directly, through a pointer, or
    // through a reference.
    Finder->addMatcher(
        declaratorDecl(NativeType, unless(isExpansionInSystemHeader())).bind("native-type"), this);
    // A native record as a function's return type.
    Finder->addMatcher(functionDecl(returns(anyOf(NativeRecord, pointerType(pointee(NativeRecord)),
                                                  referenceType(pointee(NativeRecord)))),
                                    unless(isExpansionInSystemHeader()))
                           .bind("native-type"),
                       this);
}

void BoundaryIsolationCheck::check(const MatchFinder::MatchResult& Result) {
    const SourceManager& SM = *Result.SourceManager;
    if (Result.Nodes.getNodeAs<CallExpr>("native-call") != nullptr) {
        const auto* Call = Result.Nodes.getNodeAs<CallExpr>("native-call");
        const SourceLocation Loc = SM.getExpansionLoc(Call->getBeginLoc());
        if (!SM.isWrittenInMainFile(Loc)) return;
        const PresumedLoc Presumed = SM.getPresumedLoc(Loc);
        if (!Presumed.isValid() || fileIsExempt(Presumed.getFilename())) return;
        diag(Call->getBeginLoc(), "native call '%0' outside the boundary directories; use the "
                                  "façade (include/skippy, include/skippy_api core) instead "
                                  "(TYPE-RULES.md §0)")
            << Call->getDirectCallee()->getQualifiedNameAsString();
        return;
    }
    const auto* Decl = Result.Nodes.getNodeAs<DeclaratorDecl>("native-type");
    if (Decl == nullptr) return;
    const SourceLocation Loc = SM.getExpansionLoc(Decl->getLocation());
    if (!SM.isWrittenInMainFile(Loc)) return;
    const PresumedLoc Presumed = SM.getPresumedLoc(Loc);
    if (!Presumed.isValid() || fileIsExempt(Presumed.getFilename())) return;
    PrintingPolicy Policy(Result.Context->getLangOpts());
    diag(Decl->getLocation(), "native type '%0' outside the boundary directories "
                              "(TYPE-RULES.md §0)")
        << QualType{Decl->getType().getTypePtr()->getUnqualifiedDesugaredType(), 0}.getAsString(
               Policy);
}

} // namespace clang::tidy::skippy

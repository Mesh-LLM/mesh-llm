#include "NoRawPointerCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoRawPointerCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(
        declaratorDecl(hasType(pointerType()), unless(isExpansionInSystemHeader())).bind("decl"),
        this);
    // A raw pointer return type is the same violation at the signature level.
    Finder->addMatcher(
        functionDecl(returns(pointerType()), unless(isExpansionInSystemHeader())).bind("decl"),
        this);
}

void NoRawPointerCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Decl = Result.Nodes.getNodeAs<DeclaratorDecl>("decl");
    if (Decl == nullptr) return;
    // FunctionDecl::getType() is the function prototype; the pointer rule is
    // about what it returns.
    const QualType Type = isa<FunctionDecl>(Decl)
                              ? cast<FunctionDecl>(Decl)->getReturnType()
                              : Decl->getType();
    const auto* Pointer = Type->getAs<PointerType>();
    if (Pointer == nullptr) return;
    // Function pointers are not data pointers.
    if (Pointer->getPointeeType()->isFunctionType()) return;

    const std::string Spelling = Decl->getNameAsString();
    if (Names_.match(Spelling)) return;
    const SourceManager& SM = *Result.SourceManager;
    const PresumedLoc Presumed = SM.getPresumedLoc(SM.getExpansionLoc(Decl->getLocation()));
    if (Presumed.isValid() && Files_.match(Presumed.getFilename())) return;

    diag(Decl->getLocation(),
         "raw pointer '%0' outside an owning cell: use a span/string_view/reference (a view), "
         "or an owner (unique_ptr with a named deleter); the only raw pointers are inside "
         "owning cells and in the ABI struct built by native() (TYPE-RULES.md §4)")
        << Spelling;
}

} // namespace clang::tidy::skippy

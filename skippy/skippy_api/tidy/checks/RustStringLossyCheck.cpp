#include "RustStringLossyCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void RustStringLossyCheck::registerMatchers(MatchFinder* Finder) {
    const auto RustString = cxxRecordDecl(hasName("::rust::String"));
    const auto Ascii = ignoringParenImpCasts(
        anyOf(stringLiteral(), callExpr(callee(functionDecl(matchesName(AsciiProducers_)))),
              hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(RustString))))));
    Finder->addMatcher(
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(
                ofClass(RustString),
                unless(anyOf(isCopyConstructor(), isMoveConstructor(), isDefaultConstructor())))),
            unless(hasArgument(0, Ascii)), unless(isExpansionInSystemHeader()))
            .bind("ctor"),
        this);
}

void RustStringLossyCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
    diag(Ctor->getBeginLoc(),
         "rust::String(...) validates UTF-8 and throws on native bytes; use "
         "rust::String::lossy(...) for any text the backend, planner or a file produced");
}

} // namespace clang::tidy::skippy

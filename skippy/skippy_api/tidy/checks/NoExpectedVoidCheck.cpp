#include "NoExpectedVoidCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoExpectedVoidCheck::registerMatchers(MatchFinder* Finder) {
    // The specialization loc, not the elaborated wrapper around it: one report per spelling.
    Finder->addMatcher(
        templateSpecializationTypeLoc(
            loc(qualType(hasDeclaration(classTemplateSpecializationDecl(
                hasName("::std::expected"), hasTemplateArgument(0, refersToType(voidType())))))),
            unless(isExpansionInSystemHeader()))
            .bind("loc"),
        this);
}

void NoExpectedVoidCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Loc = Result.Nodes.getNodeAs<TypeLoc>("loc");
    const SourceLocation Begin = Loc->getBeginLoc();
    if (Begin.isInvalid() || !Reported_.insert(Begin.getRawEncoding()).second) return;
    diag(Begin, "std::expected<void, E> is banned: a failure-only result is Fallible<E> "
                "(nullopt-style: an engaged value is the reason it failed)");
}

} // namespace clang::tidy::skippy

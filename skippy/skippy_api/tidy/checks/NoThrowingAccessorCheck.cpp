#include "NoThrowingAccessorCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoThrowingAccessorCheck::registerMatchers(MatchFinder* Finder) {
    // Matched on the callee name, not on the class: libc++'s optional keeps
    // `value` behind inline namespaces and base classes, and the rule is
    // about the accessor, not the type that carries it.
    Finder->addMatcher(cxxMemberCallExpr(callee(cxxMethodDecl(hasName("value"))),
                                         unless(isExpansionInSystemHeader()))
                           .bind("value"),
                       this);
}

void NoThrowingAccessorCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("value");
    if (Call == nullptr) return;
    const SourceLocation Begin = Call->getBeginLoc();
    if (Begin.isInvalid() || !Reported_.insert(Begin.getRawEncoding()).second) return;
    diag(Begin, "'.value()' is the throwing accessor; results compose — use and_then/transform/"
                "value_or, or guard with `if (!result)` and read `*result` "
                "(TYPE-RULES.md §4)");
}

} // namespace clang::tidy::skippy

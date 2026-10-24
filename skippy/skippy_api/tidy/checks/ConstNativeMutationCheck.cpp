#include "ConstNativeMutationCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void ConstNativeMutationCheck::registerMatchers(MatchFinder* Finder) {
    const auto FromThis = expr(anyOf(cxxThisExpr(), hasDescendant(cxxThisExpr())));
    Finder->addMatcher(
        callExpr(callee(functionDecl(matchesName(NativeRegex_)).bind("native")),
                 hasAncestor(cxxMethodDecl(isConst(), unless(ofClass(isLambda()))).bind("method")),
                 forEachArgumentWithParam(FromThis.bind("arg"),
                                          parmVarDecl(hasType(pointerType(pointee(unless(isConstQualified())))))))
            .bind("call"),
        this);
}

void ConstNativeMutationCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Arg = Result.Nodes.getNodeAs<Expr>("arg");
    diag(Arg->getBeginLoc(),
         "const method %0 passes a member to %1 through a non-const pointer: that mutates state behind "
         "const; drop const on the method or take a const pointer")
        << Result.Nodes.getNodeAs<CXXMethodDecl>("method") << Result.Nodes.getNodeAs<FunctionDecl>("native");
}

} // namespace clang::tidy::skippy

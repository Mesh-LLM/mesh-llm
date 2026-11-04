#include "NoVoidSilencingCheck.h"

#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoVoidSilencingCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(
        explicitCastExpr(hasDestinationType(voidType()),
                         hasSourceExpression(ignoringParenImpCasts(declRefExpr(
                             to(varDecl(hasLocalStorage(), unless(parmVarDecl())).bind("var"))))),
                         hasAncestor(functionDecl(hasBody(stmt().bind("body")))))
            .bind("cast"),
        this);
}

namespace {

/// A reference that only stores into the variable (`x = …`, `x += …`, `++x`)
/// is not a read; a variable whose only read is its (void) cast is dead.
bool isWrite(ASTContext& Ctx, const DeclRefExpr& Ref) {
    const Stmt* Child = &Ref;
    for (;;) {
        const auto Parents = Ctx.getParents(*Child);
        if (Parents.empty()) return false;
        const Stmt* Parent = Parents[0].get<Stmt>();
        if (Parent == nullptr) return false;
        if (isa<ImplicitCastExpr>(Parent) || isa<ParenExpr>(Parent)) {
            Child = Parent;
            continue;
        }
        if (const auto* Binary = dyn_cast<BinaryOperator>(Parent))
            return Binary->isAssignmentOp() && Binary->getLHS() == Child;
        if (const auto* Unary = dyn_cast<UnaryOperator>(Parent))
            return Unary->isIncrementDecrementOp();
        return false;
    }
}

} // namespace

void NoVoidSilencingCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Cast = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");
    const auto* Var = Result.Nodes.getNodeAs<VarDecl>("var");
    const auto* Body = Result.Nodes.getNodeAs<Stmt>("body");
    ASTContext& Ctx = *Result.Context;

    const auto Uses = match(findAll(declRefExpr(to(equalsNode(Var))).bind("use")), *Body, Ctx);
    const auto Reads = llvm::count_if(Uses, [&](const BoundNodes& Bound) {
        return !isWrite(Ctx, *Bound.getNodeAs<DeclRefExpr>("use"));
    });
    if (Reads != 1) return;

    diag(Cast->getBeginLoc(),
         "%0 is never read except to be discarded; delete it or use it (the cast hides a "
         "dead store from -Wunused-but-set-variable and the analyzer)")
        << Var;
    diag(Var->getLocation(), "defined here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::skippy

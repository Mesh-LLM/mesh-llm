#include "PrecededByCheck.h"
#include "Support.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void PrecededByCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(callExpr(callee(functionDecl(hasName(Call_)))).bind("call"), this);
}

void PrecededByCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Call = Result.Nodes.getNodeAs<CallExpr>("call");
    ASTContext& Ctx = *Result.Context;

    const auto Block = support::enclosingBlock(Ctx, *Call);
    const auto Missing = [&] {
        diag(Call->getBeginLoc(),
             "%0 must be immediately preceded by %1 in the same block: the previous "
             "pass may still be running on the backend")
            << Call_ << Before_;
    };
    if (!Block) return Missing();

    const auto& [Compound, Statement] = *Block;
    const auto Position = llvm::find(Compound->body(), Statement);
    if (Position == Compound->body_begin()) return Missing();

    const Stmt& Previous = **std::prev(Position);
    const bool Quiesced =
        !match(findAll(callExpr(callee(functionDecl(hasName(Before_)))).bind("before")), Previous,
               Ctx)
             .empty();
    if (!Quiesced) Missing();
}

} // namespace clang::tidy::skippy

#include "NoRawLoopsCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoRawLoopsCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(
        stmt(anyOf(forStmt(), whileStmt(), doStmt()), unless(isExpansionInSystemHeader()))
            .bind("loop"),
        this);
    if (FlagRangeFor_) {
        Finder->addMatcher(cxxForRangeStmt(unless(isExpansionInSystemHeader())).bind("range"),
                           this);
    }
}

void NoRawLoopsCheck::check(const MatchFinder::MatchResult& Result) {
    if (const auto* Loop = Result.Nodes.getNodeAs<Stmt>("loop")) {
        if (Loop->getBeginLoc().isMacroID()) return;
        diag(Loop->getBeginLoc(), "raw loop; write it as a ranges algorithm or adapter "
                                  "(an index loop is allowed only at an index-based ABI: NOLINT "
                                  "this line with the reason)");
        return;
    }
    if (const auto* Range = Result.Nodes.getNodeAs<Stmt>("range")) {
        if (Range->getBeginLoc().isMacroID()) return;
        diag(Range->getBeginLoc(),
             "range-for; prefer a named algorithm (for_each, transform, fold_left)");
    }
}

} // namespace clang::tidy::skippy

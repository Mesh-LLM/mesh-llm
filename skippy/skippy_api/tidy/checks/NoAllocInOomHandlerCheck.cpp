#include "NoAllocInOomHandlerCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoAllocInOomHandlerCheck::registerMatchers(MatchFinder* Finder) {
    const auto Growable = cxxRecordDecl(hasAnyName("::std::basic_string", "::std::vector",
                                                   "::std::deque", "::std::unordered_map",
                                                   "::std::map", "::rust::String", "::rust::Vec"));
    const auto OwnsHeap =
        hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(Growable))));

    const auto Allocating = expr(anyOf(
        cxxNewExpr(),
        callExpr(callee(functionDecl(matchesName("^::std::(make_unique|make_shared|to_string|"
                                                 "format|vformat)$|^::rust::String::lossy$")))),
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(
                ofClass(Growable), unless(anyOf(isDefaultConstructor(), isMoveConstructor())))),
            unless(argumentCountIs(0))),
        cxxMemberCallExpr(on(OwnsHeap), callee(cxxMethodDecl(hasAnyName(
                                            "push_back", "emplace_back", "append", "assign",
                                            "insert", "reserve", "resize", "emplace")))),
        cxxOperatorCallExpr(hasAnyOverloadedOperatorName("+", "+="), hasArgument(0, OwnsHeap))));

    const auto OutOfMemoryHandler =
        anyOf(isCatchAll(),
              has(varDecl(hasType(references(cxxRecordDecl(hasName("::std::bad_alloc")))))));

    if (OnlyInNoexcept_) {
        Finder->addMatcher(cxxCatchStmt(OutOfMemoryHandler, hasAncestor(functionDecl(isNoThrow())),
                                        unless(isExpansionInSystemHeader()),
                                        forEachDescendant(Allocating.bind("alloc"))),
                           this);
    } else {
        Finder->addMatcher(cxxCatchStmt(OutOfMemoryHandler, unless(isExpansionInSystemHeader()),
                                        forEachDescendant(Allocating.bind("alloc"))),
                           this);
    }
}

void NoAllocInOomHandlerCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Alloc = Result.Nodes.getNodeAs<Expr>("alloc");
    diag(Alloc->getBeginLoc(),
         "allocation inside an out-of-memory handler%0: if this throws, the handler terminates. "
         "Report a code "
         "with no text (Failure<Code>::out_of_memory()) or give the allocation its own catch (...)")
        << (OnlyInNoexcept_ ? " of a noexcept function" : "");
}

} // namespace clang::tidy::skippy

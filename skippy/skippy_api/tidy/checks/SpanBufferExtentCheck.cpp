#include "SpanBufferExtentCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include "llvm/ADT/APSInt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void SpanBufferExtentCheck::registerMatchers(MatchFinder* Finder) {
    // The class is matched as any record and confirmed by unqualified name in
    // check(): a matcher spelled `std::span` never matches libc++'s
    // `std::__1::span` (the once_flag trap).
    Finder->addMatcher(
        cxxConstructExpr(
            hasDeclaration(cxxConstructorDecl(ofClass(cxxRecordDecl().bind("record")))),
            argumentCountIs(2),
            hasArgument(0, ignoringImpCasts(declRefExpr(to(varDecl().bind("buffer"))))),
            hasArgument(1, expr().bind("count")))
            .bind("span"),
        this);
}

void SpanBufferExtentCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Record = Result.Nodes.getNodeAs<CXXRecordDecl>("record");
    const auto* Buffer = Result.Nodes.getNodeAs<VarDecl>("buffer");
    const auto* Count = Result.Nodes.getNodeAs<Expr>("count");
    const auto* Span = Result.Nodes.getNodeAs<CXXConstructExpr>("span");
    if (Record == nullptr || Record->getName() != "span") return;
    const auto* Array = Result.Context->getAsConstantArrayType(Buffer->getType());
    if (Array == nullptr) return;
    if (const auto Value = Count->getIntegerConstantExpr(*Result.Context);
        Value && Value->isNonNegative() &&
        Value->getSExtValue() <= Array->getSize().getSExtValue()) {
        return; // a constant prefix of the buffer: honest about what it names.
    }
    diag(Span->getBeginLoc(),
         "std::span over fixed buffer '%0' with a runtime count; take the extent from the buffer "
         "(single-argument form) and let the consuming boundary reject the overflow")
        << Buffer->getName();
}

} // namespace clang::tidy::skippy

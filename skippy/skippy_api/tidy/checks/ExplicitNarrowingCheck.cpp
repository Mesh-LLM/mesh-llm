#include "ExplicitNarrowingCheck.h"
#include "Support.h"

#include "clang-tidy/utils/OptionsUtils.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/APSInt.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

ExplicitNarrowingCheck::ExplicitNarrowingCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      AllowedFunctions_(utils::options::parseStringList(
          Options.get("AllowedFunctions", "skippy::narrow;skippy::checked_mul"))),
      AllowedFilesRegex_(Options.get("AllowedFiles", "narrow\\.h$")),
      AllowedFiles_(AllowedFilesRegex_), BoundMarker_(Options.get("BoundMarker", "bounded by")),
      FlagSignChangeOnWidening_(Options.get("FlagSignChangeOnWidening", true)) {}

void ExplicitNarrowingCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "AllowedFunctions", utils::options::serializeStringList(AllowedFunctions_));
    Options.store(Opts, "AllowedFiles", AllowedFilesRegex_);
    Options.store(Opts, "BoundMarker", BoundMarker_);
    Options.store(Opts, "FlagSignChangeOnWidening", FlagSignChangeOnWidening_);
}

void ExplicitNarrowingCheck::registerMatchers(MatchFinder* Finder) {
    const auto IntegerNotBool = qualType(isInteger(), unless(booleanType()));
    Finder->addMatcher(explicitCastExpr(hasDestinationType(IntegerNotBool),
                                        hasSourceExpression(hasType(IntegerNotBool)),
                                        unless(isExpansionInSystemHeader()))
                           .bind("cast"),
                       this);
}

namespace {

bool fits(const llvm::APSInt& Value, unsigned ToWidth, bool ToSigned) {
    llvm::APSInt Converted = Value.extOrTrunc(ToWidth);
    Converted.setIsSigned(ToSigned);
    return llvm::APSInt::isSameValue(Value, Converted);
}

} // namespace

void ExplicitNarrowingCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Cast = Result.Nodes.getNodeAs<ExplicitCastExpr>("cast");
    ASTContext& Ctx = *Result.Context;
    const Expr* Source = Cast->getSubExprAsWritten()->IgnoreParenImpCasts();

    const QualType To = Cast->getTypeAsWritten();
    const QualType From = Source->getType();
    const unsigned ToWidth = Ctx.getIntWidth(To);
    const unsigned FromWidth = Ctx.getIntWidth(From);
    const bool ToSigned = To->isSignedIntegerType();
    const bool FromSigned = From->isSignedIntegerType();

    const bool LosesWidth = ToWidth < FromWidth;
    const bool LosesSign =
        (FromSigned && !ToSigned && (ToWidth <= FromWidth || FlagSignChangeOnWidening_)) ||
        (!FromSigned && ToSigned && ToWidth <= FromWidth);
    if (!LosesWidth && !LosesSign) return;

    if (Expr::EvalResult Constant;
        Source->EvaluateAsInt(Constant, Ctx) && fits(Constant.Val.getInt(), ToWidth, ToSigned))
        return;
    if (support::fileMatches(*Result.SourceManager, Cast->getBeginLoc(), AllowedFiles_)) return;
    if (const FunctionDecl* Owner =
            support::enclosingNamedFunction(Ctx, DynTypedNode::create(*Cast));
        Owner != nullptr &&
        llvm::is_contained(AllowedFunctions_, StringRef{Owner->getQualifiedNameAsString()}))
        return;
    if (!BoundMarker_.empty() &&
        StringRef{support::lineText(*Result.SourceManager, Result.Context->getLangOpts(),
                                    Cast->getBeginLoc())}
            .contains_insensitive(BoundMarker_))
        return;

    diag(Cast->getBeginLoc(),
         "cast to %0 from %1 can lose value (%2 → %3 bits%4); use narrow<%5>() and handle nullopt, "
         "or state the proven bound on this line (\"%6 ...\")")
        << To << From << FromWidth << ToWidth << (LosesSign ? ", signedness changes" : "")
        << To.getAsString() << BoundMarker_;
}

} // namespace clang::tidy::skippy

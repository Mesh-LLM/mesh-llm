#include "NoFailureOptionalCheck.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoFailureOptionalCheck::registerMatchers(MatchFinder* Finder) {
    // The specialization loc, not the elaborated wrapper around it: one
    // report per spelling.
    Finder->addMatcher(
        templateSpecializationTypeLoc(loc(qualType(hasDeclaration(classTemplateSpecializationDecl(
                                          hasName("::std::optional"))))),
                                      unless(isExpansionInSystemHeader()))
            .bind("loc"),
        this);
}

void NoFailureOptionalCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Loc = Result.Nodes.getNodeAs<TemplateSpecializationTypeLoc>("loc");
    if (Loc == nullptr) return;
    const SourceLocation Begin = Loc->getBeginLoc();
    if (Begin.isInvalid() || !Reported_.insert(Begin.getRawEncoding()).second) return;
    const auto* Spec =
        Loc->getTypePtr()->getAsTagDecl()->getDescribedTemplate() != nullptr
            ? nullptr
            : dyn_cast_or_null<ClassTemplateSpecializationDecl>(Loc->getTypePtr()->getAsTagDecl());
    if (Spec == nullptr || Spec->getTemplateArgs().size() == 0) return;
    const QualType Payload = Spec->getTemplateArgs()[0].getAsType();
    if (Payload.isNull()) return;
    PrintingPolicy Policy(Result.Context->getLangOpts());
    Policy.SuppressTagKeyword = true;
    const std::string Name = Payload.getUnqualifiedType().getAsString(Policy);
    if (!ErrorType_.match(Name)) return;
    diag(Begin, "std::optional<%0> is a failure channel; the failure-only spelling is "
                "std::expected<void, %0>, which chains with and_then "
                "(TYPE-RULES.md §4)")
        << Name;
}

} // namespace clang::tidy::skippy

#include "OptionalErrorPolarityCheck.h"

#include "clang/AST/DeclTemplate.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void OptionalErrorPolarityCheck::registerMatchers(MatchFinder* Finder) {
    // `!x`, `if (x)`, `x ? :` all go through optional's explicit operator bool.
    Finder->addMatcher(
        cxxMemberCallExpr(callee(cxxConversionDecl()),
                          on(expr(hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                              classTemplateSpecializationDecl(hasName("::std::optional")).bind("spec"))))))),
                          unless(isExpansionInSystemHeader()))
            .bind("conv"),
        this);
}

void OptionalErrorPolarityCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Spec = Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("spec");
    const auto* Conv = Result.Nodes.getNodeAs<CXXMemberCallExpr>("conv");
    if (Spec->getTemplateArgs().size() == 0) return;
    const QualType Payload = Spec->getTemplateArgs()[0].getAsType();
    PrintingPolicy Policy(Result.Context->getLangOpts());
    Policy.SuppressTagKeyword = true;
    const std::string Name = Payload.getUnqualifiedType().getAsString(Policy);
    if (!ErrorType_.match(Name)) return;
    diag(Conv->getBeginLoc(),
         "std::optional<%0> is a failure channel (engaged means it failed); converting it to bool "
         "inverts. Spell .has_value(), or return std::expected<void, %0> (TYPE-RULES.md §4)")
        << Name;
}

} // namespace clang::tidy::skippy

#include "NoProcessWideStateCheck.h"
#include "Support.h"

#include "clang-tidy/utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

NoProcessWideStateCheck::NoProcessWideStateCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context), AllowedTypes_(utils::options::parseStringList(
                                         Options.get("AllowedTypes", "::std::once_flag"))),
      AllowedInFunctions_(utils::options::parseStringList(
          Options.get("AllowedInFunctions", "skippy::detail::next_generation"))) {}

void NoProcessWideStateCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "AllowedTypes", utils::options::serializeStringList(AllowedTypes_));
    Options.store(Opts, "AllowedInFunctions",
                  utils::options::serializeStringList(AllowedInFunctions_));
}

void NoProcessWideStateCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(
        varDecl(
            hasGlobalStorage(), isDefinition(),
            unless(anyOf(isConstexpr(), hasType(isConstQualified()), isExpansionInSystemHeader(),
                         hasType(hasUnqualifiedDesugaredType(recordType(
                             hasDeclaration(cxxRecordDecl(hasAnyName(AllowedTypes_)))))))))
            .bind("var"),
        this);
}

void NoProcessWideStateCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Var = Result.Nodes.getNodeAs<VarDecl>("var");
    if (Var->isInvalidDecl() || Var->getType()->containsErrors()) return;
    if (Var->isStaticLocal()) {
        const FunctionDecl* Owner =
            support::enclosingNamedFunction(*Result.Context, DynTypedNode::create(*Var));
        if (Owner != nullptr &&
            llvm::is_contained(AllowedInFunctions_, StringRef{Owner->getQualifiedNameAsString()}))
            return;
    }
    diag(Var->getLocation(), "%0 is mutable process-wide state: two runtimes in one process share "
                             "it and tests cannot reset it. "
                             "Own it in the object that uses it (PlanStore in LlamaBackend is the "
                             "pattern) or allow it by name "
                             "with the reason in .clang-tidy")
        << Var;
}

} // namespace clang::tidy::skippy

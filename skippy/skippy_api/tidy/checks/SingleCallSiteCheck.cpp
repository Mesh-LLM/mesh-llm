#include "SingleCallSiteCheck.h"
#include "Support.h"

#include "clang-tidy/utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

SingleCallSiteCheck::SingleCallSiteCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      Functions_(utils::options::parseStringList(Options.get("Functions", ""))),
      AllowedCallers_(utils::options::parseStringList(Options.get("AllowedCallers", ""))),
      Reason_(Options.get("Reason", "")) {}

void SingleCallSiteCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "Functions", utils::options::serializeStringList(Functions_));
    Options.store(Opts, "AllowedCallers", utils::options::serializeStringList(AllowedCallers_));
    Options.store(Opts, "Reason", Reason_);
}

void SingleCallSiteCheck::registerMatchers(MatchFinder* Finder) {
    if (Functions_.empty()) return;
    Finder->addMatcher(
        callExpr(callee(functionDecl(hasAnyName(Functions_)).bind("callee"))).bind("call"), this);
}

void SingleCallSiteCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Call = Result.Nodes.getNodeAs<CallExpr>("call");
    const auto* Callee = Result.Nodes.getNodeAs<FunctionDecl>("callee");
    const FunctionDecl* Caller =
        support::enclosingNamedFunction(*Result.Context, DynTypedNode::create(*Call));
    const std::string CallerName =
        Caller != nullptr ? Caller->getQualifiedNameAsString() : "<global>";
    if (llvm::is_contained(AllowedCallers_, StringRef{CallerName})) return;

    diag(Call->getBeginLoc(), "%0 may only be read from %1 (here: %2)%3%4")
        << Callee << utils::options::serializeStringList(AllowedCallers_) << CallerName
        << (Reason_.empty() ? "" : ": ") << Reason_;
}

} // namespace clang::tidy::skippy

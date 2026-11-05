#include "BridgeBarrierCheck.h"
#include "Support.h"

#include "clang-tidy/utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

BridgeBarrierCheck::BridgeBarrierCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context), FileRegex_(Options.get("FileRegex", "bridge\\.cpp$")),
      File_(FileRegex_), Barriers_(utils::options::parseStringList(
                             Options.get("Barriers", "guarded;guarded_message"))),
      Exempt_(utils::options::parseStringList(Options.get("Exempt", ""))) {}

void BridgeBarrierCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "FileRegex", FileRegex_);
    Options.store(Opts, "Barriers", utils::options::serializeStringList(Barriers_));
    Options.store(Opts, "Exempt", utils::options::serializeStringList(Exempt_));
}

void BridgeBarrierCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(cxxMethodDecl(isDefinition(), isNoThrow(),
                                     unless(anyOf(cxxConstructorDecl(), cxxDestructorDecl(),
                                                  isImplicit(), isDeleted(), isDefaulted())))
                           .bind("method"),
                       this);
}

void BridgeBarrierCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
    if (!support::fileMatches(*Result.SourceManager, Method->getLocation(), File_)) return;
    if (llvm::is_contained(Exempt_, Method->getName())) return;

    const auto* Body = dyn_cast_or_null<CompoundStmt>(Method->getBody());
    // A call returning a non-trivial type (a cxx record with a rust::String
    // member) is wrapped in CXXBindTemporaryExpr/ExprWithCleanups; strip those
    // or every correctly-barred method with a rich result type fires.
    const auto StripWrappers = [](const Expr* E) -> const Expr* {
        while (E != nullptr) {
            if (const auto* Paren = dyn_cast<ParenExpr>(E)) {
                E = Paren->getSubExpr();
                continue;
            }
            if (const auto* Cast = dyn_cast<ImplicitCastExpr>(E)) {
                E = Cast->getSubExpr();
                continue;
            }
            if (const auto* Bind = dyn_cast<CXXBindTemporaryExpr>(E)) {
                E = Bind->getSubExpr();
                continue;
            }
            if (const auto* Clean = dyn_cast<ExprWithCleanups>(E)) {
                E = Clean->getSubExpr();
                continue;
            }
            break;
        }
        return E;
    };
    const auto RoutesThroughBarrier = [&] {
        if (Body == nullptr || Body->size() != 1) return false;
        const auto* Return = dyn_cast<ReturnStmt>(Body->body_front());
        if (Return == nullptr || Return->getRetValue() == nullptr) return false;
        const auto* Call = dyn_cast<CallExpr>(StripWrappers(Return->getRetValue()));
        const FunctionDecl* Callee = Call != nullptr ? Call->getDirectCallee() : nullptr;
        return Callee != nullptr && llvm::is_contained(Barriers_, Callee->getName());
    };
    if (RoutesThroughBarrier()) return;

    diag(Method->getLocation(),
         "noexcept bridge method %0 is not `return <barrier>(...)` (%1): any allocation or string "
         "conversion in its body is std::terminate into Rust")
        << Method << utils::options::serializeStringList(Barriers_);
}

} // namespace clang::tidy::skippy

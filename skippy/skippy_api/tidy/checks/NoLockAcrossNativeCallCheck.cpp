#include "NoLockAcrossNativeCallCheck.h"

#include "clang-tidy/utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

NoLockAcrossNativeCallCheck::NoLockAcrossNativeCallCheck(StringRef Name, ClangTidyContext* Context)
    : ClangTidyCheck(Name, Context),
      LockTypes_(utils::options::parseStringList(Options.get(
          "LockTypes", "::std::scoped_lock;::std::lock_guard;::std::unique_lock;::std::shared_lock"))),
      NativeRegex_(Options.get("NativeRegex", "^::(llama_|ggml_|skippy_)")) {}

void NoLockAcrossNativeCallCheck::storeOptions(ClangTidyOptions::OptionMap& Opts) {
    Options.store(Opts, "LockTypes", utils::options::serializeStringList(LockTypes_));
    Options.store(Opts, "NativeRegex", NativeRegex_);
}

void NoLockAcrossNativeCallCheck::registerMatchers(MatchFinder* Finder) {
    Finder->addMatcher(
        varDecl(hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(cxxRecordDecl(hasAnyName(LockTypes_)))))),
                hasParent(declStmt(hasParent(compoundStmt().bind("block"))).bind("decl")))
            .bind("lock"),
        this);
}

void NoLockAcrossNativeCallCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Lock = Result.Nodes.getNodeAs<VarDecl>("lock");
    const auto* Decl = Result.Nodes.getNodeAs<DeclStmt>("decl");
    const auto* Block = Result.Nodes.getNodeAs<CompoundStmt>("block");
    ASTContext& Ctx = *Result.Context;

    const auto NativeCall = callExpr(callee(functionDecl(matchesName(NativeRegex_)).bind("native"))).bind("call");
    const auto Unlock =
        cxxMemberCallExpr(callee(cxxMethodDecl(hasName("unlock"))), on(declRefExpr(to(equalsNode(Lock)))));

    auto After = std::next(llvm::find(Block->body(), Decl));
    for (; After != Block->body_end(); ++After) {
        const Stmt& Statement = **After;
        if (!match(findAll(Unlock), Statement, Ctx).empty()) return;
        for (const auto& Bound : match(findAll(NativeCall), Statement, Ctx)) {
            diag(Bound.getNodeAs<CallExpr>("call")->getBeginLoc(),
                 "native call %0 while %1 is held: a mutex must not be held across a call into the backend "
                 "(release it, or make the shared state an RCU snapshot)")
                << Bound.getNodeAs<FunctionDecl>("native") << Lock;
        }
    }
}

} // namespace clang::tidy::skippy

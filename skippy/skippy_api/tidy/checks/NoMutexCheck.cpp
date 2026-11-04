#include "NoMutexCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::skippy {

void NoMutexCheck::registerMatchers(MatchFinder* Finder) {
    // Every declared lockable (a member, a local, a parameter) is one lock
    // too many; the types come from the system header, the declarations do not.
    Finder->addMatcher(valueDecl(hasType(hasUnqualifiedDesugaredType(recordType(
                                     hasDeclaration(namedDecl(matchesName(LockTypeRegex_)))))),
                                 unless(isExpansionInSystemHeader()))
                           .bind("decl"),
                       this);
}

void NoMutexCheck::check(const MatchFinder::MatchResult& Result) {
    const auto* Decl = Result.Nodes.getNodeAs<ValueDecl>("decl");
    if (Decl == nullptr) return;
    const Type* Desugared = Decl->getType().getTypePtr()->getUnqualifiedDesugaredType();
    const auto* Record = Desugared != nullptr ? Desugared->getAsTagDecl() : nullptr;
    if (Record == nullptr) return;
    if (!LockType_.match(Record->getQualifiedNameAsString())) return;
    diag(Decl->getLocation(), "'%0' is a lock; no mutex anywhere — shared state is an RCU "
                              "snapshot behind atomic_shared_ptr or a CAS over atomics "
                              "(TYPE-RULES.md §3.2)")
        << Record->getQualifiedNameAsString();
}

} // namespace clang::tidy::skippy

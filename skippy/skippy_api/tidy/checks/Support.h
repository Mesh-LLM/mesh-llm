#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"

#include <optional>
#include <string>
#include <utility>

namespace clang::tidy::skippy::support {

/// The innermost function that is not a lambda's call operator.
inline const FunctionDecl* enclosingNamedFunction(ASTContext& Ctx, const DynTypedNode& Start) {
    DynTypedNode Node = Start;
    for (;;) {
        const auto Parents = Ctx.getParents(Node);
        if (Parents.empty()) return nullptr;
        Node = Parents[0];
        if (const auto* Function = Node.get<FunctionDecl>()) {
            const auto* Method = dyn_cast<CXXMethodDecl>(Function);
            const bool LambdaOperator = Method != nullptr && Method->getParent()->isLambda();
            if (!LambdaOperator) return Function;
        }
    }
}

/// The compound statement that directly holds the statement containing
/// `Start`, and that statement.
inline std::optional<std::pair<const CompoundStmt*, const Stmt*>>
enclosingBlock(ASTContext& Ctx, const Stmt& Start) {
    const Stmt* Child = &Start;
    for (;;) {
        const auto Parents = Ctx.getParents(*Child);
        if (Parents.empty()) return std::nullopt;
        const Stmt* Parent = Parents[0].get<Stmt>();
        if (Parent == nullptr) return std::nullopt;
        if (const auto* Block = dyn_cast<CompoundStmt>(Parent)) return std::pair{Block, Child};
        Child = Parent;
    }
}

inline bool fileMatches(const SourceManager& SM, SourceLocation Loc, const llvm::Regex& Pattern) {
    const PresumedLoc Presumed = SM.getPresumedLoc(SM.getSpellingLoc(Loc));
    return Presumed.isValid() && Pattern.match(Presumed.getFilename());
}

/// The full text of the source line holding `Loc`.
inline std::string lineText(const SourceManager& SM, const LangOptions& LangOpts, SourceLocation Loc) {
    const SourceLocation Spelling = SM.getSpellingLoc(Loc);
    const FileID File = SM.getFileID(Spelling);
    const unsigned Line = SM.getSpellingLineNumber(Spelling);
    const SourceLocation Begin = SM.translateLineCol(File, Line, 1);
    const SourceLocation End = SM.translateLineCol(File, Line + 1, 1);
    if (Begin.isInvalid() || End.isInvalid()) return {};
    return Lexer::getSourceText(CharSourceRange::getCharRange(Begin, End), SM, LangOpts).str();
}

} // namespace clang::tidy::skippy::support

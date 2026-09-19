#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using clang::ASTContext;
using clang::BinaryOperator;
using clang::CallExpr;
using clang::CompoundStmt;
using clang::CXXConstructorDecl;
using clang::DeclRefExpr;
using clang::Expr;
using clang::ForStmt;
using clang::FunctionDecl;
using clang::IfStmt;
using clang::RecursiveASTVisitor;
using clang::SourceLocation;
using clang::SourceManager;
using clang::Stmt;
using clang::StringLiteral;
using clang::VarDecl;
using clang::ast_matchers::cxxConstructorDecl;
using clang::ast_matchers::isDefinition;
using clang::ast_matchers::isExpansionInMainFile;
using clang::ast_matchers::MatchFinder;
using clang::ast_matchers::unless;

llvm::cl::OptionCategory RewriterCategory("skippy-stage-rewriter options");

llvm::cl::opt<std::string>
    SourceRoot("source-root",
               llvm::cl::desc("Root of the prepared llama.cpp source tree"),
               llvm::cl::value_desc("path"), llvm::cl::Required,
               llvm::cl::cat(RewriterCategory));

llvm::cl::opt<std::string> ReportPath(
    "report",
    llvm::cl::desc("Write the deterministic JSON report to this path"),
    llvm::cl::value_desc("path"), llvm::cl::Required,
    llvm::cl::cat(RewriterCategory));

llvm::cl::opt<std::string> LlamaCommit(
    "llama-commit",
    llvm::cl::desc("Exact llama.cpp commit represented by the source tree"),
    llvm::cl::value_desc("sha"), llvm::cl::Required,
    llvm::cl::cat(RewriterCategory));

llvm::cl::opt<bool>
    Apply("apply", llvm::cl::desc("Apply proven edits to the source tree"),
          llvm::cl::init(false), llvm::cl::cat(RewriterCategory));

struct Edit {
  std::string kind;
  std::string file;
  uint64_t offset = 0;
  uint64_t length = 0;
  std::string text;
};

struct Proof {
  std::string loop_var;
  std::string loop_start;
  std::string loop_end;
  std::string activation_in;
  std::string activation_out;
  bool embedding_owner = false;
  bool output_owner = false;
  std::vector<std::string> nonlocal_exits;
  std::string execution_scope = "partitioned_decoder";
  std::vector<std::string> scope_evidence;
};

struct BuilderReport {
  std::string file;
  std::string constructor;
  unsigned line = 0;
  std::string verdict;
  std::string unsupported_reason;
  Proof proof;
  std::vector<Edit> edits;
};

std::string sourceText(clang::SourceRange range, const SourceManager &sm,
                       const clang::LangOptions &lang) {
  if (range.isInvalid() || range.getBegin().isMacroID() ||
      range.getEnd().isMacroID()) {
    return {};
  }
  return clang::Lexer::getSourceText(
             clang::CharSourceRange::getTokenRange(range), sm, lang)
      .str();
}

std::optional<std::string> referencedName(const Expr *expr) {
  if (expr == nullptr) {
    return std::nullopt;
  }
  expr = expr->IgnoreParenImpCasts();
  if (const auto *ref = llvm::dyn_cast<DeclRefExpr>(expr)) {
    return ref->getDecl()->getNameAsString();
  }
  if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
    return member->getMemberDecl()->getNameAsString();
  }
  return std::nullopt;
}

const FunctionDecl *directCallee(const CallExpr *call) {
  return call == nullptr ? nullptr : call->getDirectCallee();
}

std::optional<std::pair<uint64_t, uint64_t>>
tokenRange(clang::SourceRange range, const SourceManager &sm,
           const clang::LangOptions &lang);

std::vector<const BinaryOperator *> assignmentsTo(const Stmt *root,
                                                  llvm::StringRef variable);

struct HyperconnectionPrelude {
  const VarDecl *carried_decl = nullptr;
  const BinaryOperator *repeat_assignment = nullptr;
  const Stmt *repeat_statement = nullptr;
  const CallExpr *repeat_call = nullptr;
  bool repeat_is_initializer = false;
  std::vector<const Stmt *> embedding_prelude_statements;
  std::string width;
  std::string multiplicity;
  std::string tokens;
};

struct AltupPrelude {
  const CompoundStmt *statement = nullptr;
  const CallExpr *repeat_call = nullptr;
  std::string width;
  std::string tokens;
  std::string count;
  // Name of the file-local static view-slice helper the builder uses for its
  // altup slices. Discovered from the AST by shape, never by upstream naming.
  std::string slice_helper;
};

struct PerLayerTokenProjection {
  const CallExpr *build_call = nullptr;
  const CallExpr *project_call = nullptr;
  const Stmt *build_statement = nullptr;
  const IfStmt *owner = nullptr;
  std::string variable;
};

struct RangeAwareInput {
  const CallExpr *build_call = nullptr;
  const Stmt *build_statement = nullptr;
  std::string variable;
};

struct RwkvFirstValue {
  const CallExpr *time_mix_call = nullptr;
  const Stmt *time_mix_statement = nullptr;
  const Stmt *next_statement = nullptr;
  std::string variable;
};

bool containsName(const Stmt *statement, llvm::StringRef target) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(llvm::StringRef target) : target_(target) {}

    bool VisitDeclRefExpr(DeclRefExpr *ref) {
      found_ |= ref->getDecl()->getNameAsString() == target_;
      return !found_;
    }

    bool VisitMemberExpr(clang::MemberExpr *member) {
      found_ |= member->getMemberDecl()->getNameAsString() == target_;
      return !found_;
    }

    bool found() const { return found_; }

  private:
    llvm::StringRef target_;
    bool found_ = false;
  } visitor(target);
  visitor.TraverseStmt(const_cast<Stmt *>(statement));
  return visitor.found();
}

bool containsLayerBound(const Expr *expression) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool VisitDeclRefExpr(DeclRefExpr *ref) {
      found_ |= llvm::StringRef(ref->getDecl()->getNameAsString())
                    .contains("n_layer");
      return !found_;
    }

    bool VisitMemberExpr(clang::MemberExpr *member) {
      found_ |= llvm::StringRef(member->getMemberDecl()->getNameAsString())
                    .contains("n_layer");
      return !found_;
    }

    bool found() const { return found_; }

  private:
    bool found_ = false;
  } visitor;
  visitor.TraverseStmt(const_cast<Expr *>(expression));
  return visitor.found();
}

std::string stableLoopEnd(const Expr *expression, const SourceManager &sm,
                          const clang::LangOptions &lang) {
  const Expr *normalized = expression->IgnoreParenImpCasts();
  if (const auto *reference = llvm::dyn_cast<DeclRefExpr>(normalized)) {
    if (const auto *variable = llvm::dyn_cast<VarDecl>(reference->getDecl())) {
      if (variable->getType().isConstQualified() && variable->hasInit()) {
        const std::string initializer =
            sourceText(variable->getInit()->getSourceRange(), sm, lang);
        if (!initializer.empty()) {
          return initializer;
        }
      }
    }
  }
  return sourceText(expression->getSourceRange(), sm, lang);
}

const Stmt *directChildContaining(const CompoundStmt *body, const Stmt *needle,
                                  const SourceManager &sm,
                                  const clang::LangOptions &lang) {
  const auto needle_range = tokenRange(needle->getSourceRange(), sm, lang);
  if (!needle_range) {
    return nullptr;
  }
  for (const Stmt *statement : body->body()) {
    const auto statement_range =
        tokenRange(statement->getSourceRange(), sm, lang);
    if (!statement_range) {
      continue;
    }
    if (statement_range->first <= needle_range->first &&
        statement_range->first + statement_range->second >=
            needle_range->first + needle_range->second) {
      return statement;
    }
  }
  return nullptr;
}

std::optional<std::string> assignedName(const CallExpr *call,
                                        ASTContext &context) {
  clang::DynTypedNode current = clang::DynTypedNode::create(*call);
  for (unsigned depth = 0; depth < 24; ++depth) {
    const auto parents = context.getParents(current);
    if (parents.size() != 1) {
      return std::nullopt;
    }
    const auto &parent = parents[0];
    if (const auto *binary = parent.get<BinaryOperator>()) {
      if (binary->isAssignmentOp()) {
        return referencedName(binary->getLHS());
      }
    }
    if (const auto *variable = parent.get<VarDecl>()) {
      return variable->getNameAsString();
    }
    if (parent.get<CompoundStmt>() != nullptr) {
      return std::nullopt;
    }
    current = parent;
  }
  return std::nullopt;
}

std::optional<uint64_t> fileOffset(SourceLocation location,
                                   const SourceManager &sm) {
  if (location.isInvalid() || location.isMacroID()) {
    return std::nullopt;
  }
  location = sm.getSpellingLoc(location);
  if (!location.isValid() || !sm.isWrittenInMainFile(location)) {
    return std::nullopt;
  }
  return sm.getFileOffset(location);
}

std::optional<std::pair<uint64_t, uint64_t>>
tokenRange(clang::SourceRange range, const SourceManager &sm,
           const clang::LangOptions &lang) {
  const auto begin = fileOffset(range.getBegin(), sm);
  const SourceLocation end_location = clang::Lexer::getLocForEndOfToken(
      sm.getSpellingLoc(range.getEnd()), 0, sm, lang);
  const auto end = fileOffset(end_location, sm);
  if (!begin || !end || *end < *begin) {
    return std::nullopt;
  }
  return std::pair<uint64_t, uint64_t>{*begin, *end - *begin};
}

std::string indentationAt(SourceLocation location, const SourceManager &sm) {
  location = sm.getSpellingLoc(location);
  if (!location.isValid() || !sm.isWrittenInMainFile(location)) {
    return {};
  }
  const char *data = sm.getCharacterData(location);
  const unsigned column = sm.getSpellingColumnNumber(location);
  std::string indent;
  for (unsigned i = 1; i < column; ++i) {
    const char ch = data[-static_cast<ptrdiff_t>(column - i)];
    if (ch != ' ' && ch != '\t') {
      return {};
    }
    indent.push_back(ch);
  }
  return indent;
}

class ExitVisitor final : public RecursiveASTVisitor<ExitVisitor> {
public:
  ExitVisitor(ASTContext &context, const ForStmt *target)
      : context_(context), target_(target) {}

  bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

  bool VisitReturnStmt(clang::ReturnStmt *) {
    exits.emplace_back("return");
    return true;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    exits.emplace_back("goto");
    return true;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    exits.emplace_back("throw");
    return true;
  }

  bool VisitBreakStmt(clang::BreakStmt *statement) {
    if (targetsLayerLoop(statement, true)) {
      exits.emplace_back("break");
    }
    return true;
  }

  bool VisitContinueStmt(clang::ContinueStmt *statement) {
    if (targetsLayerLoop(statement, false)) {
      continues.push_back(statement);
    }
    return true;
  }

  std::vector<std::string> exits;
  std::vector<const clang::ContinueStmt *> continues;

private:
  bool targetsLayerLoop(const Stmt *statement, bool breakable) const {
    clang::DynTypedNode current = clang::DynTypedNode::create(*statement);
    for (unsigned depth = 0; depth < 48; ++depth) {
      const auto parents = context_.getParents(current);
      if (parents.size() != 1) {
        return false;
      }
      const auto &parent = parents[0];
      if (const auto *loop = parent.get<ForStmt>()) {
        return loop == target_;
      }
      if (parent.get<clang::WhileStmt>() != nullptr ||
          parent.get<clang::DoStmt>() != nullptr ||
          (breakable && parent.get<clang::SwitchStmt>() != nullptr)) {
        return false;
      }
      current = parent;
    }
    return false;
  }

  ASTContext &context_;
  const ForStmt *target_;
};

int layerLoopScore(const ForStmt *loop, llvm::StringRef activation,
                   const SourceManager &sm, const clang::LangOptions &lang) {
  const auto *body = llvm::dyn_cast<CompoundStmt>(loop->getBody());
  const auto *condition =
      llvm::dyn_cast_or_null<BinaryOperator>(loop->getCond());
  if (body == nullptr || condition == nullptr) {
    return -1;
  }

  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr) {
        return true;
      }
      const std::string name = callee->getNameAsString();
      has_cvec |= name == "build_cvec";
      if (name == "cb" && call->getNumArgs() >= 2) {
        const auto *label = llvm::dyn_cast<StringLiteral>(
            call->getArg(1)->IgnoreParenImpCasts());
        if (label != nullptr) {
          const llvm::StringRef value = label->getString();
          has_layer_output |= value == "l_out" || value == "l_last";
        }
      }
      return true;
    }

    bool has_cvec = false;
    bool has_layer_output = false;
  } visitor;
  visitor.TraverseStmt(const_cast<CompoundStmt *>(body));

  int score = 0;
  score += visitor.has_cvec ? 16 : 0;
  score += visitor.has_layer_output ? 16 : 0;
  score += containsName(body, "layers") ? 4 : 0;
  score += assignmentsTo(body, activation).empty() ? 0 : 8;
  score +=
      sourceText(condition->getRHS()->getSourceRange(), sm, lang) == "il_end"
          ? 2
          : 0;
  return score;
}

std::optional<std::string>
layerCarriedName(const CompoundStmt *body, llvm::StringRef embedding_activation,
                 ASTContext &context, const SourceManager &sm,
                 const clang::LangOptions &lang) {
  if (!assignmentsTo(body, embedding_activation).empty()) {
    return embedding_activation.str();
  }

  struct Candidate {
    uint64_t offset;
    std::string name;
  };
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(ASTContext &context, const SourceManager &sm,
            const clang::LangOptions &lang, std::vector<Candidate> &candidates)
        : context_(context), sm_(sm), lang_(lang), candidates_(candidates) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr) {
        return true;
      }
      const std::string callee_name = callee->getNameAsString();
      std::optional<std::string> name;
      if (callee_name == "build_cvec") {
        name = assignedName(call, context_);
        if (!name && call->getNumArgs() > 0) {
          name = referencedName(call->getArg(0));
        }
      } else if (callee_name == "cb" && call->getNumArgs() >= 2) {
        const auto *label = llvm::dyn_cast<StringLiteral>(
            call->getArg(1)->IgnoreParenImpCasts());
        if (label != nullptr &&
            (label->getString() == "l_out" || label->getString() == "l_last")) {
          name = referencedName(call->getArg(0));
        }
      }
      const auto offset = fileOffset(call->getBeginLoc(), sm_);
      if (name && offset) {
        candidates_.push_back(Candidate{*offset, *name});
      }
      return true;
    }

  private:
    ASTContext &context_;
    const SourceManager &sm_;
    const clang::LangOptions &lang_;
    std::vector<Candidate> &candidates_;
  };

  std::vector<Candidate> candidates;
  Visitor output_visitor(context, sm, lang, candidates);
  output_visitor.TraverseStmt(const_cast<CompoundStmt *>(body));
  if (candidates.empty()) {
    return std::nullopt;
  }
  return std::max_element(candidates.begin(), candidates.end(),
                          [](const Candidate &left, const Candidate &right) {
                            return left.offset < right.offset;
                          })
      ->name;
}

class FactVisitor final : public RecursiveASTVisitor<FactVisitor> {
public:
  bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

  bool VisitCallExpr(CallExpr *call) {
    const auto *callee = directCallee(call);
    if (callee == nullptr) {
      return true;
    }
    const std::string name = callee->getNameAsString();
    calls[name].push_back(call);
    return true;
  }

  bool VisitForStmt(ForStmt *loop) {
    const auto *init = llvm::dyn_cast_or_null<clang::DeclStmt>(loop->getInit());
    if (init == nullptr || !init->isSingleDecl()) {
      return true;
    }
    const auto *variable = llvm::dyn_cast<VarDecl>(init->getSingleDecl());
    const auto *condition =
        llvm::dyn_cast_or_null<BinaryOperator>(loop->getCond());
    if (variable == nullptr || !variable->hasInit() || condition == nullptr ||
        condition->getOpcode() != clang::BO_LT) {
      return true;
    }
    if (!containsName(condition->getLHS(), variable->getNameAsString()) ||
        (!containsLayerBound(condition->getRHS()) &&
         !containsName(condition->getRHS(), "il_end"))) {
      return true;
    }
    layer_loops.push_back(loop);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *ref) {
    has_stage_filter |= ref->getDecl()->getNameAsString() == "stage_filter";
    return true;
  }

  std::map<std::string, std::vector<const CallExpr *>> calls;
  std::vector<const ForStmt *> layer_loops;
  bool has_stage_filter = false;
};

bool isCallbackOnly(const Stmt *statement) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    bool TraverseLambdaExpr(clang::LambdaExpr *) {
      valid_ = false;
      return false;
    }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr || callee->getNameAsString() != "cb") {
        valid_ = false;
        return false;
      }
      saw_callback_ = true;
      return true;
    }

    bool VisitBinaryOperator(BinaryOperator *binary) {
      if (binary->isAssignmentOp()) {
        valid_ = false;
        return false;
      }
      return true;
    }

    bool valid() const { return valid_ && saw_callback_; }

  private:
    bool valid_ = true;
    bool saw_callback_ = false;
  } visitor;
  visitor.TraverseStmt(const_cast<Stmt *>(statement));
  return visitor.valid();
}

std::vector<const BinaryOperator *> assignmentsTo(const Stmt *root,
                                                  llvm::StringRef variable) {
  class AssignmentVisitor final
      : public RecursiveASTVisitor<AssignmentVisitor> {
  public:
    AssignmentVisitor(llvm::StringRef variable,
                      std::vector<const BinaryOperator *> &results)
        : variable_(variable), results_(results) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitBinaryOperator(BinaryOperator *binary) {
      if (!binary->isAssignmentOp()) {
        return true;
      }
      const auto lhs = referencedName(binary->getLHS());
      if (lhs && *lhs == variable_) {
        results_.push_back(binary);
      }
      return true;
    }

  private:
    llvm::StringRef variable_;
    std::vector<const BinaryOperator *> &results_;
  };

  std::vector<const BinaryOperator *> assignments;
  AssignmentVisitor visitor(variable, assignments);
  visitor.TraverseStmt(const_cast<Stmt *>(root));
  return assignments;
}

std::optional<HyperconnectionPrelude>
hyperconnectionPrelude(const CompoundStmt *constructor_body,
                       const ForStmt *loop, const Stmt *embedding_statement,
                       llvm::StringRef embedding_activation,
                       llvm::StringRef carried, const SourceManager &sm,
                       const clang::LangOptions &lang) {
  if (embedding_activation == carried) {
    return std::nullopt;
  }
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  if (!loop_offset) {
    return std::nullopt;
  }

  const VarDecl *carried_decl = nullptr;
  class DeclVisitor final : public RecursiveASTVisitor<DeclVisitor> {
  public:
    DeclVisitor(llvm::StringRef carried, uint64_t loop_offset,
                const SourceManager &sm, const VarDecl *&result)
        : carried_(carried), loop_offset_(loop_offset), sm_(sm),
          result_(result) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitVarDecl(VarDecl *decl) {
      const auto offset = fileOffset(decl->getBeginLoc(), sm_);
      if (decl->getName() == carried_ && decl->hasInit() && offset &&
          *offset < loop_offset_) {
        result_ = result_ == nullptr ? decl : nullptr;
        ambiguous_ |= result_ == nullptr;
      }
      return true;
    }

    bool ambiguous() const { return ambiguous_; }

  private:
    llvm::StringRef carried_;
    uint64_t loop_offset_;
    const SourceManager &sm_;
    const VarDecl *&result_;
    bool ambiguous_ = false;
  } decl_visitor(carried, *loop_offset, sm, carried_decl);
  decl_visitor.TraverseStmt(const_cast<CompoundStmt *>(constructor_body));
  if (decl_visitor.ambiguous() || carried_decl == nullptr) {
    return std::nullopt;
  }

  const BinaryOperator *repeat_assignment = nullptr;
  const CallExpr *repeat_call = nullptr;
  bool repeat_is_initializer = false;

  const auto is_repeat_call = [&](const Expr *expression,
                                  llvm::StringRef input) -> const CallExpr * {
    const auto *call =
        llvm::dyn_cast<CallExpr>(expression->IgnoreParenImpCasts());
    const auto *callee = call == nullptr ? nullptr : directCallee(call);
    if (callee == nullptr || callee->getNameAsString() != "ggml_repeat_4d" ||
        call->getNumArgs() != 6 ||
        sourceText(call->getArg(5)->getSourceRange(), sm, lang) != "1" ||
        !containsName(call->getArg(1), input)) {
      return nullptr;
    }
    return call;
  };

  if (const auto *call =
          is_repeat_call(carried_decl->getInit(), embedding_activation)) {
    repeat_call = call;
    repeat_is_initializer = true;
  }
  for (const BinaryOperator *assignment :
       assignmentsTo(constructor_body, carried)) {
    const auto offset = fileOffset(assignment->getBeginLoc(), sm);
    if (!offset || *offset >= *loop_offset) {
      continue;
    }
    const auto *call = is_repeat_call(assignment->getRHS(), carried);
    if (call == nullptr || repeat_call != nullptr) {
      return std::nullopt;
    }
    repeat_assignment = assignment;
    repeat_call = call;
  }
  if (repeat_call == nullptr) {
    return std::nullopt;
  }
  const Stmt *repeat_statement =
      directChildContaining(constructor_body, repeat_call, sm, lang);
  if (repeat_statement == nullptr) {
    return std::nullopt;
  }

  HyperconnectionPrelude result;
  result.carried_decl = carried_decl;
  result.repeat_assignment = repeat_assignment;
  result.repeat_statement = repeat_statement;
  result.repeat_call = repeat_call;
  result.repeat_is_initializer = repeat_is_initializer;
  const auto embedding_range =
      tokenRange(embedding_statement->getSourceRange(), sm, lang);
  const auto repeat_range =
      tokenRange(repeat_statement->getSourceRange(), sm, lang);
  if (!embedding_range || !repeat_range) {
    return std::nullopt;
  }
  const uint64_t embedding_end =
      embedding_range->first + embedding_range->second;
  for (const Stmt *statement : constructor_body->body()) {
    const auto statement_range =
        tokenRange(statement->getSourceRange(), sm, lang);
    if (!statement_range || statement == repeat_statement ||
        statement_range->first < embedding_end ||
        statement_range->first >= repeat_range->first ||
        !containsName(statement, embedding_activation)) {
      continue;
    }
    const auto carried_init_range =
        tokenRange(carried_decl->getInit()->getSourceRange(), sm, lang);
    if (carried_init_range &&
        statement_range->first <= carried_init_range->first &&
        statement_range->first + statement_range->second >=
            carried_init_range->first + carried_init_range->second) {
      continue;
    }
    result.embedding_prelude_statements.push_back(statement);
  }
  result.width = sourceText(repeat_call->getArg(2)->getSourceRange(), sm, lang);
  result.multiplicity =
      sourceText(repeat_call->getArg(3)->getSourceRange(), sm, lang);
  result.tokens =
      sourceText(repeat_call->getArg(4)->getSourceRange(), sm, lang);
  if (result.width.empty() || result.multiplicity.empty() ||
      result.tokens.empty()) {
    return std::nullopt;
  }
  return result;
}

// Find the file-local static helper the builder uses to view one 2D slice of
// a 3D activation tensor. The helper is recognized by its shape: a static
// free function taking (ggml_context *, ggml_tensor *, index) whose body
// performs a ggml_view_2d. Returns nullopt unless exactly one such helper is
// referenced, so irregular builders are refused instead of guessed at.
std::optional<std::string>
viewSliceHelper(const CompoundStmt *constructor_body, const SourceManager &sm) {
  std::string found;
  bool ambiguous = false;
  class SliceHelperVisitor final : public RecursiveASTVisitor<SliceHelperVisitor> {
  public:
    SliceHelperVisitor(const SourceManager &sm, std::string &found,
                       bool &ambiguous)
        : sm_(sm), found_(found), ambiguous_(ambiguous) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      if (callee == nullptr || call->getNumArgs() != 3 ||
          callee->getNumParams() != 3) {
        return true;
      }
      if (callee->getStorageClass() != clang::SC_Static ||
          !callee->getDeclContext()->isFileContext()) {
        return true;
      }
      const FunctionDecl *definition = nullptr;
      if (!callee->hasBody(definition) || definition == nullptr ||
          !containsName(definition->getBody(), "ggml_view_2d")) {
        return true;
      }
      const auto name = callee->getNameAsString();
      if (found_.empty()) {
        found_ = name;
      } else if (found_ != name) {
        ambiguous_ = true;
      }
      return true;
    }

  private:
    const SourceManager &sm_;
    std::string &found_;
    bool &ambiguous_;
  } visitor(sm, found, ambiguous);
  visitor.TraverseStmt(const_cast<CompoundStmt *>(constructor_body));
  if (ambiguous || found.empty()) {
    return std::nullopt;
  }
  return found;
}

std::optional<AltupPrelude>
altupPrelude(const CompoundStmt *constructor_body, const ForStmt *loop,
             llvm::StringRef activation, llvm::StringRef carried,
             const SourceManager &sm, const clang::LangOptions &lang) {
  if (activation != carried || !containsName(constructor_body, "i_altup_act")) {
    return std::nullopt;
  }
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  if (!loop_offset) {
    return std::nullopt;
  }

  const CallExpr *repeat = nullptr;
  class RepeatVisitor final : public RecursiveASTVisitor<RepeatVisitor> {
  public:
    RepeatVisitor(llvm::StringRef activation, uint64_t loop_offset,
                  const SourceManager &sm, const CallExpr *&result)
        : activation_(activation), loop_offset_(loop_offset), sm_(sm),
          result_(result) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitCallExpr(CallExpr *call) {
      const auto *callee = directCallee(call);
      const auto offset = fileOffset(call->getBeginLoc(), sm_);
      if (callee == nullptr || callee->getNameAsString() != "ggml_repeat_4d" ||
          call->getNumArgs() != 6 || !offset || *offset >= loop_offset_ ||
          !containsName(call->getArg(1), activation_)) {
        return true;
      }
      result_ = result_ == nullptr ? call : nullptr;
      ambiguous_ |= result_ == nullptr;
      return true;
    }

    bool ambiguous() const { return ambiguous_; }

  private:
    llvm::StringRef activation_;
    uint64_t loop_offset_;
    const SourceManager &sm_;
    const CallExpr *&result_;
    bool ambiguous_ = false;
  } repeat_visitor(activation, *loop_offset, sm, repeat);
  repeat_visitor.TraverseStmt(const_cast<CompoundStmt *>(constructor_body));
  if (repeat_visitor.ambiguous() || repeat == nullptr) {
    return std::nullopt;
  }

  const Stmt *direct =
      directChildContaining(constructor_body, repeat, sm, lang);
  const auto *statement = llvm::dyn_cast_or_null<CompoundStmt>(direct);
  if (statement == nullptr || !containsName(statement, "ggml_concat")) {
    return std::nullopt;
  }

  const Expr *count_expression = repeat->getArg(4)->IgnoreParenImpCasts();
  const auto *count_subtract = llvm::dyn_cast<BinaryOperator>(count_expression);
  if (count_subtract == nullptr ||
      count_subtract->getOpcode() != clang::BO_Sub) {
    return std::nullopt;
  }
  const auto count = referencedName(count_subtract->getLHS());
  const auto *one = llvm::dyn_cast<clang::IntegerLiteral>(
      count_subtract->getRHS()->IgnoreParenImpCasts());
  if (!count || one == nullptr || one->getValue() != 1) {
    return std::nullopt;
  }

  AltupPrelude result;
  result.statement = statement;
  result.repeat_call = repeat;
  result.width = sourceText(repeat->getArg(2)->getSourceRange(), sm, lang);
  result.tokens = sourceText(repeat->getArg(3)->getSourceRange(), sm, lang);
  result.count = *count;
  if (result.width.empty() || result.tokens.empty() || result.count.empty()) {
    return std::nullopt;
  }
  // The altup stage boundary and per-layer projection fallback must slice the
  // carried activation with the builder's own view helper. Discover its name
  // from the constructor body; a builder without one is not transformable.
  const auto slice_helper = viewSliceHelper(constructor_body, sm);
  if (!slice_helper) {
    return std::nullopt;
  }
  result.slice_helper = *slice_helper;
  return result;
}

std::optional<PerLayerTokenProjection>
perLayerTokenProjection(const CompoundStmt *constructor_body,
                        const ForStmt *loop, llvm::StringRef activation,
                        const FactVisitor &facts, ASTContext &context,
                        const SourceManager &sm,
                        const clang::LangOptions &lang) {
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  if (!loop_offset || facts.calls.count("project_per_layer_inputs") == 0 ||
      facts.calls.count("build_inp_per_layer") == 0) {
    return std::nullopt;
  }
  const auto &project_calls = facts.calls.at("project_per_layer_inputs");
  const auto &build_calls = facts.calls.at("build_inp_per_layer");
  if (project_calls.size() != 1 || build_calls.size() != 1 ||
      project_calls.front()->getNumArgs() < 2 ||
      !containsName(project_calls.front()->getArg(0), activation)) {
    return std::nullopt;
  }
  const auto project_offset =
      fileOffset(project_calls.front()->getBeginLoc(), sm);
  const auto build_offset = fileOffset(build_calls.front()->getBeginLoc(), sm);
  if (!project_offset || !build_offset || *project_offset >= *loop_offset ||
      *build_offset >= *loop_offset || *build_offset > *project_offset) {
    return std::nullopt;
  }
  const Stmt *build_statement =
      directChildContaining(constructor_body, build_calls.front(), sm, lang);
  const auto *owner = llvm::dyn_cast_or_null<IfStmt>(build_statement);
  const auto variable = assignedName(build_calls.front(), context);
  if (build_statement == nullptr || !variable) {
    return std::nullopt;
  }
  return PerLayerTokenProjection{build_calls.front(), project_calls.front(),
                                 build_statement, owner, *variable};
}

std::optional<RangeAwareInput>
rangeAwareInput(const CompoundStmt *constructor_body, const ForStmt *loop,
                const CompoundStmt *loop_body, const FactVisitor &facts,
                llvm::StringRef build_name, llvm::StringRef layer_predicate,
                ASTContext &context, const SourceManager &sm,
                const clang::LangOptions &lang) {
  const auto calls = facts.calls.find(build_name.str());
  if (calls == facts.calls.end() || calls->second.size() != 1 ||
      !containsName(loop_body, layer_predicate)) {
    return std::nullopt;
  }
  const CallExpr *call = calls->second.front();
  const auto call_offset = fileOffset(call->getBeginLoc(), sm);
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  const auto variable = assignedName(call, context);
  if (!call_offset || !loop_offset || *call_offset >= *loop_offset ||
      !variable || !containsName(loop_body, *variable)) {
    return std::nullopt;
  }
  const Stmt *statement =
      directChildContaining(constructor_body, call, sm, lang);
  if (statement == nullptr) {
    return std::nullopt;
  }
  return RangeAwareInput{call, statement, *variable};
}

std::optional<RwkvFirstValue>
rwkvFirstValue(const CompoundStmt *constructor_body,
               const CompoundStmt *loop_body, const FactVisitor &facts,
               const SourceManager &sm, const clang::LangOptions &lang) {
  const auto calls = facts.calls.find("build_rwkv7_time_mix");
  if (calls == facts.calls.end() || calls->second.size() != 1 ||
      calls->second.front()->getNumArgs() < 4) {
    return std::nullopt;
  }
  const CallExpr *call = calls->second.front();
  const auto variable = referencedName(call->getArg(3));
  if (!variable) {
    return std::nullopt;
  }

  const VarDecl *declaration = nullptr;
  class DeclarationVisitor final
      : public RecursiveASTVisitor<DeclarationVisitor> {
  public:
    DeclarationVisitor(llvm::StringRef variable, const VarDecl *&result)
        : variable_(variable), result_(result) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitVarDecl(VarDecl *candidate) {
      if (candidate->getName() != variable_) {
        return true;
      }
      if (result_ != nullptr) {
        ambiguous_ = true;
      } else {
        result_ = candidate;
      }
      return true;
    }

    bool ambiguous() const { return ambiguous_; }

  private:
    llvm::StringRef variable_;
    const VarDecl *&result_;
    bool ambiguous_ = false;
  } declaration_visitor(*variable, declaration);
  declaration_visitor.TraverseStmt(
      const_cast<CompoundStmt *>(constructor_body));
  if (declaration_visitor.ambiguous() || declaration == nullptr ||
      !declaration->hasInit() ||
      sourceText(declaration->getInit()->getSourceRange(), sm, lang) !=
          "nullptr") {
    return std::nullopt;
  }

  const Stmt *time_mix_statement =
      directChildContaining(loop_body, call, sm, lang);
  if (time_mix_statement == nullptr) {
    return std::nullopt;
  }
  const Stmt *next_statement = nullptr;
  bool found = false;
  for (const Stmt *statement : loop_body->body()) {
    if (found) {
      next_statement = statement;
      break;
    }
    found = statement == time_mix_statement;
  }
  if (!found || next_statement == nullptr) {
    return std::nullopt;
  }
  return RwkvFirstValue{call, time_mix_statement, next_statement, *variable};
}

std::vector<const IfStmt *> stageZeroSidebands(const CompoundStmt *loop_body,
                                               llvm::StringRef loop_var) {
  class Visitor final : public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(llvm::StringRef loop_var, std::vector<const IfStmt *> &results)
        : loop_var_(loop_var), results_(results) {}

    bool TraverseLambdaExpr(clang::LambdaExpr *) { return true; }

    bool VisitIfStmt(IfStmt *statement) {
      if (containsName(statement->getCond(), loop_var_) &&
          containsName(statement, "t_inp_embd")) {
        results_.push_back(statement);
      }
      return true;
    }

  private:
    llvm::StringRef loop_var_;
    std::vector<const IfStmt *> &results_;
  };

  std::vector<const IfStmt *> results;
  Visitor visitor(loop_var, results);
  visitor.TraverseStmt(const_cast<CompoundStmt *>(loop_body));
  return results;
}

std::vector<const IfStmt *> stageZeroEmbeddingModeChecks(
    const CompoundStmt *constructor_body, const Stmt *embedding_statement,
    const ForStmt *loop, llvm::StringRef carried, const SourceManager &sm,
    const clang::LangOptions &lang) {
  const auto embedding_range =
      tokenRange(embedding_statement->getSourceRange(), sm, lang);
  const auto loop_offset = fileOffset(loop->getBeginLoc(), sm);
  if (!embedding_range || !loop_offset) {
    return {};
  }
  const uint64_t embedding_end =
      embedding_range->first + embedding_range->second;
  std::vector<const IfStmt *> results;
  for (const Stmt *statement : constructor_body->body()) {
    const auto statement_offset = fileOffset(statement->getBeginLoc(), sm);
    const auto *conditional = llvm::dyn_cast<IfStmt>(statement);
    if (conditional == nullptr || !statement_offset ||
        *statement_offset < embedding_end ||
        *statement_offset >= *loop_offset) {
      continue;
    }
    const std::string text = sourceText(statement->getSourceRange(), sm, lang);
    if (llvm::StringRef(text).contains("ubatch.embd") &&
        assignmentsTo(conditional, carried).empty()) {
      results.push_back(conditional);
    }
  }
  return results;
}

bool addReplace(std::vector<Edit> &edits, llvm::StringRef kind,
                llvm::StringRef file, clang::SourceRange range,
                llvm::StringRef replacement, const SourceManager &sm,
                const clang::LangOptions &lang) {
  const auto bytes = tokenRange(range, sm, lang);
  if (!bytes) {
    return false;
  }
  edits.push_back(Edit{kind.str(), file.str(), bytes->first, bytes->second,
                       replacement.str()});
  return true;
}

bool addInsert(std::vector<Edit> &edits, llvm::StringRef kind,
               llvm::StringRef file, SourceLocation location,
               llvm::StringRef text, const SourceManager &sm) {
  const auto offset = fileOffset(location, sm);
  if (!offset) {
    return false;
  }
  edits.push_back(Edit{kind.str(), file.str(), *offset, 0, text.str()});
  return true;
}

bool nonOverlapping(std::vector<Edit> edits) {
  std::sort(edits.begin(), edits.end(),
            [](const Edit &left, const Edit &right) {
              return std::tie(left.file, left.offset, left.length) <
                     std::tie(right.file, right.offset, right.length);
            });
  for (size_t i = 1; i < edits.size(); ++i) {
    if (edits[i - 1].file != edits[i].file) {
      continue;
    }
    if (edits[i - 1].offset == edits[i].offset ||
        edits[i - 1].offset + edits[i - 1].length > edits[i].offset) {
      return false;
    }
  }
  return true;
}

class BuilderCallback final : public MatchFinder::MatchCallback {
public:
  void run(const MatchFinder::MatchResult &result) override {
    const auto *constructor =
        result.Nodes.getNodeAs<CXXConstructorDecl>("constructor");
    if (constructor == nullptr || constructor->getBody() == nullptr ||
        result.SourceManager == nullptr || result.Context == nullptr) {
      return;
    }
    if (constructor->getParent()->isLambda()) {
      return;
    }

    ASTContext &context = *result.Context;
    const SourceManager &sm = *result.SourceManager;
    const auto &lang = context.getLangOpts();
    const SourceLocation location =
        sm.getSpellingLoc(constructor->getLocation());
    if (!location.isValid() || !sm.isWrittenInMainFile(location)) {
      return;
    }

    llvm::SmallString<256> canonical_file;
    if (llvm::sys::fs::real_path(sm.getFilename(location), canonical_file)) {
      return;
    }
    const std::string file = canonical_file.str().str();
    const std::string models_root = SourceRoot + "/src/models/";
    if (!llvm::StringRef(file).starts_with(models_root)) {
      return;
    }
    const std::string qualified = constructor->getQualifiedNameAsString();
    if (qualified.find("::graph") == std::string::npos) {
      return;
    }

    BuilderReport report;
    report.file = llvm::StringRef(file).drop_front(SourceRoot.size() + 1).str();
    report.constructor = qualified;
    report.line = sm.getSpellingLineNumber(location);

    FactVisitor facts;
    facts.TraverseStmt(const_cast<Stmt *>(constructor->getBody()));

    const auto *constructor_body =
        llvm::dyn_cast<CompoundStmt>(constructor->getBody());

    const bool has_begin = facts.calls["begin_block"].size() == 1;
    // A transformed loop has one terminal end marker plus one marker on each
    // loop-level continue path. More than one end marker is therefore valid.
    const bool has_end = !facts.calls["end_block"].empty();
    if (!facts.has_stage_filter && has_begin && has_end) {
      report.verdict = "already_transformed";
      reports_.push_back(std::move(report));
      return;
    }
    if (facts.has_stage_filter) {
      refuse(report, "legacy model-local stage filter is not supported");
      reports_.push_back(std::move(report));
      return;
    }
    if (has_begin != has_end) {
      refuse(report, "partial block-boundary annotation");
      reports_.push_back(std::move(report));
      return;
    }

    // MTP/draft heads and encoder sidecars execute in a distinct context
    // attached to the final pipeline stage. They do not consume the primary
    // decoder's [layer_start, layer_end) interval and must never receive the
    // generic stage-loop rewrite. Prove that role from the builder type or
    // its graph inputs/results rather than from an architecture or file name.
    const bool typed_mtp_builder =
        llvm::StringRef(qualified).contains("::graph_mtp::graph_mtp");
    // A cross-context reference alone does not make the whole constructor a
    // sidecar. Some model builders (for example a combined trunk/MTP graph)
    // borrow tensors only in an auxiliary branch while still constructing a
    // normal decoder graph from their own token embedding. Classify the whole
    // constructor as a context sidecar only when it has no primary embedding
    // producer of its own.
    const bool context_sidecar = constructor_body != nullptr &&
                                 containsName(constructor_body, "ctx_other") &&
                                 facts.calls["build_inp_embd"].empty();
    const bool encoder_sidecar =
        facts.calls["build_inp_embd_enc"].size() == 1 &&
        constructor_body != nullptr &&
        containsName(constructor_body, "t_h_nextn");
    if (typed_mtp_builder || context_sidecar || encoder_sidecar) {
      report.verdict = "supported_auxiliary";
      report.proof.execution_scope = "final_stage_sidecar";
      if (typed_mtp_builder) {
        report.proof.scope_evidence.emplace_back("typed_mtp_builder");
      }
      if (context_sidecar) {
        report.proof.scope_evidence.emplace_back("cross_context_input");
      }
      if (encoder_sidecar) {
        report.proof.scope_evidence.emplace_back("encoder_sidecar_output");
      }
      reports_.push_back(std::move(report));
      return;
    }
    const auto &embedding_calls = facts.calls["build_inp_embd"];
    const CallExpr *embedding = nullptr;
    bool standard_embedding = false;
    if (embedding_calls.size() == 1 &&
        embedding_calls.front()->getNumArgs() > 0) {
      embedding = embedding_calls.front();
      standard_embedding = true;
    } else if (embedding_calls.empty()) {
      std::vector<const CallExpr *> token_gathers;
      for (const CallExpr *call : facts.calls["ggml_get_rows"]) {
        if (call->getNumArgs() >= 2 &&
            containsName(call->getArg(1), "tok_embd")) {
          token_gathers.push_back(call);
        }
      }
      if (token_gathers.size() == 1) {
        embedding = token_gathers.front();
      }
    }
    if (embedding == nullptr) {
      refuse(report, "cannot prove a unique token embedding producer");
      reports_.push_back(std::move(report));
      return;
    }
    const auto activation = assignedName(embedding, context);
    const Stmt *embedding_statement =
        constructor_body == nullptr
            ? nullptr
            : directChildContaining(constructor_body, embedding, sm, lang);
    if (!activation || embedding_statement == nullptr) {
      refuse(report, "cannot prove the embedding activation owner");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.embedding_owner = true;

    if (facts.layer_loops.empty()) {
      refuse(report, "no layer block loop");
      reports_.push_back(std::move(report));
      return;
    }
    std::vector<std::pair<int, const ForStmt *>> scored_loops;
    for (const ForStmt *candidate : facts.layer_loops) {
      scored_loops.emplace_back(
          layerLoopScore(candidate, *activation, sm, lang), candidate);
    }
    const int best_score =
        std::max_element(scored_loops.begin(), scored_loops.end(),
                         [](const auto &left, const auto &right) {
                           return left.first < right.first;
                         })
            ->first;
    std::vector<const ForStmt *> best_loops;
    for (const auto &[score, candidate] : scored_loops) {
      if (score == best_score) {
        best_loops.push_back(candidate);
      }
    }
    if (best_score <= 0 || best_loops.size() != 1) {
      if (best_score > 0 && best_loops.size() > 1) {
        std::vector<std::string> domains;
        for (const ForStmt *candidate : best_loops) {
          const auto *candidate_condition =
              llvm::dyn_cast_or_null<BinaryOperator>(candidate->getCond());
          if (candidate_condition == nullptr) {
            domains.clear();
            break;
          }
          domains.push_back(sourceText(
              candidate_condition->getRHS()->getSourceRange(), sm, lang));
        }
        std::sort(domains.begin(), domains.end());
        domains.erase(std::unique(domains.begin(), domains.end()),
                      domains.end());
        if (domains.size() > 1) {
          report.verdict = "supported_whole_model";
          report.proof.execution_scope = "multiple_sequential_layer_domains";
          report.proof.scope_evidence = std::move(domains);
          reports_.push_back(std::move(report));
          return;
        }
      }
      refuse(report, best_loops.size() == 1
                         ? "cannot prove selected layer block loop"
                         : "multiple equally ranked layer block loops");
      reports_.push_back(std::move(report));
      return;
    }
    const ForStmt *loop = best_loops.front();
    const auto *loop_body = llvm::dyn_cast<CompoundStmt>(loop->getBody());
    const auto *init = llvm::cast<clang::DeclStmt>(loop->getInit());
    const auto *loop_var = llvm::cast<VarDecl>(init->getSingleDecl());
    const auto *condition = llvm::cast<BinaryOperator>(loop->getCond());
    report.proof.loop_var = loop_var->getNameAsString();
    report.proof.loop_start =
        sourceText(loop_var->getInit()->getSourceRange(), sm, lang);
    report.proof.loop_end = stableLoopEnd(condition->getRHS(), sm, lang);
    if (loop_body == nullptr || report.proof.loop_start != "0") {
      refuse(report,
             loop_body == nullptr
                 ? "block loop body is not compound"
                 : "block loop start does not match transformation state");
      reports_.push_back(std::move(report));
      return;
    }

    ExitVisitor exits(context, loop);
    exits.TraverseStmt(const_cast<CompoundStmt *>(loop_body));
    report.proof.nonlocal_exits = exits.exits;
    if (!exits.exits.empty()) {
      refuse(report, "block loop contains a non-local exit");
      reports_.push_back(std::move(report));
      return;
    }

    const auto carried =
        layerCarriedName(loop_body, *activation, context, sm, lang);
    if (!carried) {
      refuse(report, "cannot prove the layer-carried activation");
      reports_.push_back(std::move(report));
      return;
    }
    report.proof.activation_in = *carried;
    report.proof.activation_out = *carried;

    const auto hyperconnection =
        hyperconnectionPrelude(constructor_body, loop, embedding_statement,
                               *activation, *carried, sm, lang);
    const auto altup =
        altupPrelude(constructor_body, loop, *activation, *carried, sm, lang);
    const auto per_layer_projection = perLayerTokenProjection(
        constructor_body, loop, *activation, facts, context, sm, lang);
    if (per_layer_projection && !altup &&
        per_layer_projection->owner == nullptr) {
      refuse(report, "per-layer projection is not owned by a guarded block");
      reports_.push_back(std::move(report));
      return;
    }
    const auto attention_positions =
        rangeAwareInput(constructor_body, loop, loop_body, facts,
                        "build_inp_pos", "is_recr", context, sm, lang);
    const auto attention_scale = rangeAwareInput(
        constructor_body, loop, loop_body, facts, "build_inp_attn_scale",
        "n_no_rope_layer_step", context, sm, lang);
    const auto ple_input =
        rangeAwareInput(constructor_body, loop, loop_body, facts,
                        "build_inp_ple", "is_ple", context, sm, lang);
    const auto rwkv_first =
        rwkvFirstValue(constructor_body, loop_body, facts, sm, lang);
    const bool kimi_k3_residual_sideband =
        llvm::StringRef(report.file).ends_with("src/models/kimi-k3.cpp") &&
        containsName(constructor_body, "res_bs") &&
        containsName(constructor_body, "use_attn_res");
    const bool glm_dsa_top_k_sideband =
        facts.calls.count("build_attn_inp_k_dsa") != 0 &&
        containsName(constructor_body, "prev_top_k") &&
        containsName(constructor_body, "is_indexer_full");
    const auto stage_zero_sidebands =
        stageZeroSidebands(loop_body, report.proof.loop_var);
    const auto stage_zero_embedding_checks = stageZeroEmbeddingModeChecks(
        constructor_body, embedding_statement, loop, *carried, sm, lang);
    if (*activation != *carried && !hyperconnection) {
      refuse(report,
             "layer-carried activation differs from the embedding without a "
             "proven hyperconnection prelude");
      reports_.push_back(std::move(report));
      return;
    }
    if (hyperconnection) {
      report.proof.scope_evidence.emplace_back(
          "hyperconnection_activation_frontier");
    }
    if (altup) {
      report.proof.scope_evidence.emplace_back("altup_activation_frontier");
    }
    if (per_layer_projection) {
      report.proof.scope_evidence.emplace_back(
          "per_layer_token_projection_sideband");
    }
    if (attention_positions) {
      report.proof.scope_evidence.emplace_back(
          "range_owned_attention_positions");
    }
    if (attention_scale) {
      report.proof.scope_evidence.emplace_back("range_owned_attention_scale");
    }
    if (ple_input) {
      report.proof.scope_evidence.emplace_back("range_owned_ple_input");
    }
    if (rwkv_first) {
      report.proof.scope_evidence.emplace_back("rwkv_first_value_sideband");
    }
    if (kimi_k3_residual_sideband) {
      report.proof.scope_evidence.emplace_back("kimi_k3_residual_sideband");
    }
    if (glm_dsa_top_k_sideband) {
      report.proof.scope_evidence.emplace_back("glm_dsa_top_k_sideband");
    }
    if (!stage_zero_sidebands.empty()) {
      report.proof.scope_evidence.emplace_back("stage_zero_loop_sideband");
    }
    if (!stage_zero_embedding_checks.empty()) {
      report.proof.scope_evidence.emplace_back(
          "stage_zero_embedding_mode_check");
    }

    std::vector<const BinaryOperator *> preloop_activation_assignments;
    if (!hyperconnection && *activation == *carried) {
      const auto embedding_end =
          tokenRange(embedding_statement->getSourceRange(), sm, lang);
      const auto loop_begin = fileOffset(loop->getBeginLoc(), sm);
      if (!embedding_end || !loop_begin) {
        refuse(report, "cannot locate the pre-loop activation region");
        reports_.push_back(std::move(report));
        return;
      }
      const uint64_t prelude_begin =
          embedding_end->first + embedding_end->second;
      for (const BinaryOperator *assignment :
           assignmentsTo(constructor_body, *carried)) {
        const auto offset = fileOffset(assignment->getBeginLoc(), sm);
        if (offset && *offset >= prelude_begin && *offset < *loop_begin) {
          const Stmt *statement =
              directChildContaining(constructor_body, assignment, sm, lang);
          if (altup && statement == altup->statement) {
            continue;
          }
          if (const auto *conditional =
                  llvm::dyn_cast_or_null<IfStmt>(statement);
              conditional != nullptr && conditional->getElse() != nullptr) {
            refuse(report,
                   "pre-loop activation conditional has an else branch");
            reports_.push_back(std::move(report));
            return;
          }
          preloop_activation_assignments.push_back(assignment);
        }
      }
      if (!preloop_activation_assignments.empty()) {
        report.proof.scope_evidence.emplace_back("guarded_embedding_prelude");
      }
    }

    const auto &output_calls = facts.calls["build_inp_out_ids"];
    const CallExpr *output_call = nullptr;
    if (output_calls.size() == 1) {
      output_call = output_calls.front();
    } else if (output_calls.size() > 1) {
      const auto loop_end_offset = fileOffset(loop->getEndLoc(), sm);
      std::vector<const CallExpr *> postloop_output_calls;
      for (const CallExpr *candidate : output_calls) {
        const auto candidate_offset = fileOffset(candidate->getBeginLoc(), sm);
        if (loop_end_offset && candidate_offset &&
            *candidate_offset > *loop_end_offset) {
          postloop_output_calls.push_back(candidate);
        }
      }
      if (postloop_output_calls.size() != 1) {
        refuse(report, "multiple primary build_inp_out_ids calls");
        reports_.push_back(std::move(report));
        return;
      }
      output_call = postloop_output_calls.front();
      report.proof.scope_evidence.emplace_back("sidecar_output_excluded");
    }
    report.proof.output_owner = output_call != nullptr;

    // Graph Filter V2 keeps model builders stage-independent. Generated
    // family edits only declare semantic block boundaries; the generic
    // planner/realizer owns all slicing, frontier, and residency decisions.
    {
      bool annotations_valid = true;
      const std::string inner_indent =
          indentationAt((*loop_body->body_begin())->getBeginLoc(), sm);
      const auto body_begin = clang::Lexer::getLocForEndOfToken(
          loop_body->getLBracLoc(), 0, sm, lang);
      annotations_valid &= addInsert(
          report.edits, "insert_begin_block", report.file, body_begin,
          "\n" + inner_indent + "begin_block(" + *carried + ", " +
              report.proof.loop_var + ");\n",
          sm);

      const std::string loop_indent =
          indentationAt(loop_body->getRBracLoc(), sm);
      const std::string end_block_indent =
          llvm::StringRef(inner_indent).starts_with(loop_indent)
              ? llvm::StringRef(inner_indent)
                    .drop_front(loop_indent.size())
                    .str()
              : inner_indent;
      annotations_valid &= addInsert(
          report.edits, "insert_end_block", report.file,
          loop_body->getRBracLoc(),
          end_block_indent + "end_block(" + *carried + ", " +
              report.proof.loop_var + ");\n" + loop_indent,
          sm);
      for (const clang::ContinueStmt *statement : exits.continues) {
        const std::string continue_indent =
            indentationAt(statement->getBeginLoc(), sm);
        const auto parents = context.getParents(*statement);
        if (parents.size() == 1 &&
            parents[0].get<CompoundStmt>() != nullptr) {
          annotations_valid &= addInsert(
              report.edits, "insert_end_block_before_continue", report.file,
              statement->getBeginLoc(),
              "end_block(" + *carried + ", " + report.proof.loop_var +
                  ");\n" + continue_indent,
              sm);
          continue;
        }
        const auto next_token =
            clang::Lexer::findNextToken(statement->getEndLoc(), sm, lang);
        if (!next_token || !next_token->is(clang::tok::semi)) {
          annotations_valid = false;
          continue;
        }
        std::string block_indent = continue_indent;
        if (block_indent.empty() && parents.size() == 1) {
          if (const auto *parent = parents[0].get<Stmt>()) {
            block_indent = indentationAt(parent->getBeginLoc(), sm);
          }
        }
        annotations_valid &= addReplace(
            report.edits, "wrap_end_block_before_continue", report.file,
            clang::SourceRange(statement->getBeginLoc(),
                               next_token->getLocation()),
            "{\n" + block_indent + "    end_block(" + *carried + ", " +
                report.proof.loop_var + ");\n" + block_indent +
                "    continue;\n" + block_indent + "}",
            sm, lang);
      }
      if (!annotations_valid || !nonOverlapping(report.edits)) {
        report.edits.clear();
        refuse(report, annotations_valid ? "planned edits overlap"
                                         : "cannot map edit to source bytes");
      } else {
        report.verdict = "transformable";
      }
      reports_.push_back(std::move(report));
      return;
    }

  }

  int finish() {
    std::sort(reports_.begin(), reports_.end(),
              [](const auto &left, const auto &right) {
                return std::tie(left.file, left.line, left.constructor) <
                       std::tie(right.file, right.line, right.constructor);
              });

    int apply_result = 0;
    if (Apply) {
      apply_result = applyEdits();
    }
    const int report_result = writeReport();
    return apply_result != 0 ? apply_result : report_result;
  }

private:
  static void refuse(BuilderReport &report, llvm::StringRef reason) {
    report.verdict = "unsupported_shape";
    report.unsupported_reason = reason.str();
    report.edits.clear();
  }

  int applyEdits() {
    std::map<std::string, std::vector<Edit>> by_file;
    for (const auto &report : reports_) {
      if (report.verdict != "transformable") {
        continue;
      }
      for (const auto &edit : report.edits) {
        by_file[edit.file].push_back(edit);
      }
    }
    for (auto &[file, edits] : by_file) {
      const std::string source_file = SourceRoot + "/" + file;
      std::ifstream input(source_file, std::ios::binary);
      if (!input) {
        llvm::errs() << "cannot read " << source_file << "\n";
        return 1;
      }
      std::string contents((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
      std::sort(edits.begin(), edits.end(),
                [](const Edit &left, const Edit &right) {
                  return std::tie(left.offset, left.length) >
                         std::tie(right.offset, right.length);
                });
      for (const auto &edit : edits) {
        if (edit.offset + edit.length > contents.size()) {
          llvm::errs() << "edit outside file " << source_file << "\n";
          return 1;
        }
        contents.replace(edit.offset, edit.length, edit.text);
      }
      std::error_code error;
      llvm::raw_fd_ostream output(source_file, error, llvm::sys::fs::OF_None);
      if (error) {
        llvm::errs() << "cannot write " << source_file << ": "
                     << error.message() << "\n";
        return 1;
      }
      output << contents;
    }
    return 0;
  }

  int writeReport() const {
    llvm::json::Array builders;
    int64_t transformable = 0;
    int64_t already_transformed = 0;
    int64_t unsupported_shape = 0;
    int64_t supported_auxiliary = 0;
    int64_t supported_whole_model = 0;
    int64_t errors = 0;
    for (const auto &report : reports_) {
      if (report.verdict == "transformable") {
        ++transformable;
      } else if (report.verdict == "already_transformed") {
        ++already_transformed;
      } else if (report.verdict == "unsupported_shape") {
        ++unsupported_shape;
      } else if (report.verdict == "supported_auxiliary") {
        ++supported_auxiliary;
      } else if (report.verdict == "supported_whole_model") {
        ++supported_whole_model;
      } else {
        ++errors;
      }

      llvm::json::Array edits;
      for (const auto &edit : report.edits) {
        edits.push_back(llvm::json::Object{
            {"file", edit.file},
            {"kind", edit.kind},
            {"range", llvm::json::Array{static_cast<int64_t>(edit.offset),
                                        static_cast<int64_t>(edit.offset +
                                                             edit.length)}},
            {"text", edit.text},
        });
      }
      llvm::json::Array exits;
      for (const auto &exit : report.proof.nonlocal_exits) {
        exits.push_back(exit);
      }
      llvm::json::Object proof{
          {"activation_in", report.proof.activation_in},
          {"activation_out", report.proof.activation_out},
          {"embedding_owner", report.proof.embedding_owner},
          {"loop", llvm::json::Object{{"end", report.proof.loop_end},
                                      {"start", report.proof.loop_start},
                                      {"var", report.proof.loop_var}}},
          {"nonlocal_exits", std::move(exits)},
          {"execution_scope", report.proof.execution_scope},
          {"scope_evidence",
           [&report]() {
             llvm::json::Array evidence;
             for (const auto &item : report.proof.scope_evidence) {
               evidence.push_back(item);
             }
             return evidence;
           }()},
          {"output_owner", report.proof.output_owner},
          {"terminal_predicates", llvm::json::Array{}},
      };
      builders.push_back(llvm::json::Object{
          {"constructor", report.constructor},
          {"edits", std::move(edits)},
          {"file", report.file},
          {"line", static_cast<int64_t>(report.line)},
          {"proof", std::move(proof)},
          {"unsupported_reason",
           report.unsupported_reason.empty()
               ? llvm::json::Value(nullptr)
               : llvm::json::Value(report.unsupported_reason)},
          {"verdict", report.verdict},
      });
    }

    std::error_code error;
    llvm::raw_fd_ostream output(ReportPath, error);
    if (error) {
      llvm::errs() << "cannot write report " << ReportPath << ": "
                   << error.message() << "\n";
      return 1;
    }
    output << llvm::formatv(
        "{0:2}\n", llvm::json::Value(llvm::json::Object{
                       {"builders", std::move(builders)},
                       {"generator_version", "0.5.0"},
                       {"llama_cpp_commit", LlamaCommit},
                       {"schema_version", 1},
                       {"source_root", SourceRoot},
                       {"summary",
                        llvm::json::Object{
                            {"already_transformed", already_transformed},
                            {"error", errors},
                            {"supported_auxiliary", supported_auxiliary},
                            {"supported_whole_model", supported_whole_model},
                            {"transformable", transformable},
                            {"unsupported_shape", unsupported_shape}}},
                   }));
    return 0;
  }

  std::vector<BuilderReport> reports_;
};

} // namespace

int main(int argc, const char **argv) {
  auto parser = clang::tooling::CommonOptionsParser::create(
      argc, argv, RewriterCategory, llvm::cl::OneOrMore);
  if (!parser) {
    llvm::errs() << llvm::toString(parser.takeError());
    return 1;
  }

  llvm::SmallString<256> canonical_source_root;
  if (const std::error_code error =
          llvm::sys::fs::real_path(SourceRoot, canonical_source_root)) {
    llvm::errs() << "cannot resolve source root " << SourceRoot << ": "
                 << error.message() << "\n";
    return 1;
  }
  SourceRoot = canonical_source_root.str().str();

  clang::tooling::ClangTool tool(parser->getCompilations(),
                                 parser->getSourcePathList());
  BuilderCallback callback;
  MatchFinder finder;
  finder.addMatcher(
      cxxConstructorDecl(isDefinition(), isExpansionInMainFile(),
                         unless(clang::ast_matchers::isTemplateInstantiation()))
          .bind("constructor"),
      &callback);

  const int run_result =
      tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  if (run_result != 0) {
    return run_result;
  }
  return callback.finish();
}

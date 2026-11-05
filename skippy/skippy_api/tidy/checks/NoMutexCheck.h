#pragma once

#include "clang-tidy/ClangTidyCheck.h"

#include "llvm/Support/Regex.h"

#include <string>

namespace clang::tidy::skippy {

/// TYPE-RULES.md §3.2 "No lock on shared state": shared state is an RCU snapshot
/// behind atomic_shared_ptr (PlanStore, the inventory memo) or a CAS over
/// atomics (ContextPool: atomic<llama_context*> slots). CAS and atomics are
/// the design, not a preference. Never sunsets.
///
/// Options:
///   LockTypeRegex (default "^::std::(mutex|recursive_mutex|shared_mutex|"
///                           "timed_mutex|recursive_timed_mutex|shared_timed_mutex|"
///                           "condition_variable|lock_guard|unique_lock|scoped_lock|"
///                           "shared_lock)$")
///
/// Would have caught: any reintroduction of a lock around the plan store or
/// the context pool; skippy-no-lock-across-native-call only catches the lock
/// once it exists AND is held across a native call.
class NoMutexCheck : public ClangTidyCheck {
public:
    NoMutexCheck(StringRef Name, ClangTidyContext* Context)
        : ClangTidyCheck(Name, Context),
          LockTypeRegex_(Options.get(
              "LockTypeRegex",
              "std::(mutex|recursive_mutex|shared_mutex|timed_mutex|recursive_timed_mutex|"
              "shared_timed_mutex|condition_variable|lock_guard|unique_lock|scoped_lock|"
              "shared_lock)$")),
          LockType_(LockTypeRegex_) {}
    bool isLanguageVersionSupported(const LangOptions& LangOpts) const override {
        return LangOpts.CPlusPlus;
    }
    void storeOptions(ClangTidyOptions::OptionMap& Opts) override {
        Options.store(Opts, "LockTypeRegex", LockTypeRegex_);
    }
    void registerMatchers(ast_matchers::MatchFinder* Finder) override;
    void check(const ast_matchers::MatchFinder::MatchResult& Result) override;

private:
    std::string LockTypeRegex_;
    llvm::Regex LockType_;
};

} // namespace clang::tidy::skippy

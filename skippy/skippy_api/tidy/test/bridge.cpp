// CHECKS: skippy-bridge-barrier
// CONFIG: {CheckOptions: {skippy-bridge-barrier.FileRegex: 'bridge\.cpp$'}}
#include <string>
#include <vector>
struct Result { bool ok; };
template <typename R, typename Body> R guarded(Body&& body) noexcept { try { return body(); } catch (...) { return R{}; } }

class Bridge {
public:
    Result guarded_method(const std::string& path) noexcept {
        return guarded<Result>([&] { return Result{!path.empty()}; });
    }
    Result bare_method(const std::string& path) noexcept {
        // CHECK: [[@LINE-1]]:12: warning: noexcept bridge method 'bare_method' is not `return <barrier>(...)`
        std::string copy = path;
        return Result{!copy.empty()};
    }
};

// A result type with a non-trivial destructor: the barrier call gets wrapped
// in CXXBindTemporaryExpr/ExprWithCleanups, which must not hide the barrier.
struct RichResult {
    std::string message;
};

class RichBridge {
public:
    RichResult guarded_rich(const std::string& path) noexcept {
        return guarded<RichResult>([&] { return RichResult{path}; });
    }
};

// The suppress-token accessor used to return a growing vector directly from
// noexcept. Pin the actual old/new return shapes as well as the scalar case.
struct SuppressTokensResult {
    bool ok = false;
    std::string error;
    std::vector<int> tokens;
};
template <typename R, typename Body> R guarded_message(const char*, Body&& body) noexcept {
    try { return body(); } catch (...) { return R{}; }
}
class SuppressBridge {
public:
    std::vector<int> suppress_tokens() const noexcept {
        // CHECK: [[@LINE-1]]:22: warning: noexcept bridge method 'suppress_tokens' is not `return <barrier>(...)`
        std::vector<int> tokens;
        tokens.reserve(128);
        return tokens;
    }
    SuppressTokensResult suppress_tokens_guarded() const noexcept {
        return guarded_message<SuppressTokensResult>("suppress_tokens", [] {
            SuppressTokensResult result{};
            result.tokens.reserve(128);
            result.ok = true;
            return result;
        });
    }
};

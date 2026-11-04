// CHECKS: skippy-bridge-barrier
// CONFIG: {CheckOptions: {skippy-bridge-barrier.FileRegex: 'bridge\.cpp$'}}
#include <string>
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

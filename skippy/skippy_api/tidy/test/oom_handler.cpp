// CHECKS: skippy-no-alloc-in-oom-handler
#include <new>
#include <string>
#include "rust_stub.h"
struct Out { bool ok; rust::String error; };

Out barrier(const char* where) noexcept {
    try {
        return Out{true, {}};
    } catch (const std::bad_alloc&) {
        return Out{false, rust::String::lossy(std::string(where) + ": out of memory")};
        // CHECK: [[@LINE-1]]:27: warning: allocation inside an out-of-memory handler of a noexcept function
        // CHECK: [[@LINE-2]]:47: warning: allocation inside an out-of-memory handler of a noexcept function
    }
}

Out barrier_without_text(const char*) noexcept {
    try {
        return Out{true, {}};
    } catch (const std::bad_alloc&) {
        return Out{false, {}};
    }
}

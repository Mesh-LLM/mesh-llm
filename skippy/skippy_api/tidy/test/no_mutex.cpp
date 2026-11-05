// CHECKS: skippy-no-mutex
#include <atomic>
#include <mutex>

struct Store {
    std::mutex mutex;
    // CHECK: [[@LINE-1]]:16: warning: 'std::mutex' is a lock
};

void locked() {
    std::mutex local;
    // CHECK: [[@LINE-1]]:16: warning: 'std::mutex' is a lock
    const std::scoped_lock guard(local);
    // CHECK: [[@LINE-1]]:28: warning: 'std::scoped_lock' is a lock
    (void)guard;
}

std::atomic<int> counter{0};  // negative: atomics are the design

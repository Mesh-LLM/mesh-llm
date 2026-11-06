// CHECKS: skippy-no-lock-across-native-call
#include <mutex>
#include "llama.h"
struct Store { std::mutex mutex; llama_context* ctx; };

void held_across(Store& store) {
    const std::scoped_lock lock(store.mutex);
    llama_synchronize(store.ctx);
    // CHECK: [[@LINE-1]]:5: warning: native call 'llama_synchronize' while 'lock' is held
}

void released_first(Store& store) {
    std::unique_lock lock(store.mutex);
    lock.unlock();
    llama_synchronize(store.ctx);
}

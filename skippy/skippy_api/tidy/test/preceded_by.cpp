// CHECKS: skippy-preceded-by
#include "llama.h"

void clears_without_quiescing(llama_context* ctx) {
    if (llama_memory_t memory = llama_get_memory(ctx)) {
        llama_memory_clear(memory, true);
        // CHECK: [[@LINE-1]]:9: warning: llama_memory_clear must be immediately preceded by llama_synchronize in the same block
    }
}

void clears_after_quiescing(llama_context* ctx) {
    if (llama_memory_t memory = llama_get_memory(ctx)) {
        llama_synchronize(ctx);
        llama_memory_clear(memory, true);
    }
}

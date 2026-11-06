// CHECKS: skippy-const-native-mutation
#include "llama.h"
class Session {
public:
    const float* row(int seq) const { return llama_get_embeddings_seq(context_, seq); }
    // CHECK: [[@LINE-1]]:71: warning: const method 'row' passes a member to 'llama_get_embeddings_seq' through a non-const pointer
    unsigned capacity() const { return llama_n_ctx(context_); }
private:
    llama_context* context_;
};

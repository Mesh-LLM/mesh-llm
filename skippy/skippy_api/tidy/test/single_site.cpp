// CHECKS: skippy-single-call-site
#include "llama.h"
namespace skippy::llama {
class ContextCapacity {
public:
    explicit ContextCapacity(const llama_context& c) : tokens_(llama_n_ctx(&c) / llama_n_seq_max(&c)) {}
private:
    std::uint32_t tokens_;
};
struct Session {
    ~Session() { recycle(llama_n_ctx(context_) / 1U); }
    // CHECK: [[@LINE-1]]:26: warning: 'llama_n_ctx' may only be read from skippy::llama::ClassifierHead::detect;skippy::llama::ContextCapacity::ContextCapacity;skippy::llama::BatchLimit::BatchLimit (here: skippy::llama::Session::~Session)
    void recycle(std::uint32_t);
    llama_context* context_;
};
std::uint32_t load_model(const llama_model* m) { return llama_model_n_cls_out(m); }
// CHECK: [[@LINE-1]]:57: warning: 'llama_model_n_cls_out' may only be read from {{.*}} (here: skippy::llama::load_model)
} // namespace skippy::llama

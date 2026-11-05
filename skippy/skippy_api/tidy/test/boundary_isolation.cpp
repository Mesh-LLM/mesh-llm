// CHECKS: skippy-boundary-isolation
// CONFIG: {CheckOptions: {skippy-boundary-isolation.AllowedDirectories: 'src/llama/;include/llama/;/tools/;/benchmarks/;skippy-sys/cpp/'}}
// This fixture lives in tidy/test/ — not a boundary directory — so every
// native shape below is a violation.
#include "llama.h"
// CHECK: [[@LINE-1]]:1: warning: native include 'llama.h' outside the boundary directories

struct Holder {
    llama_context* context;  // native type outside the boundary
    // CHECK: [[@LINE-1]]:20: warning: native type
};

llama_context* handout(Holder& holder) { return holder.context; }
// CHECK: [[@LINE-1]]:16: warning: native type

void drive(llama_context* context) { llama_synchronize(context); }
// CHECK: [[@LINE-1]]:27: warning: native type
    // CHECK: [[@LINE-2]]:38: warning: native call

void clean(int* data, int count) { (void)data; (void)count; }  // negative: no native shape

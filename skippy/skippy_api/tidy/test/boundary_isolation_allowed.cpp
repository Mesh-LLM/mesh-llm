// CHECKS: skippy-boundary-isolation
// CONFIG: {CheckOptions: {skippy-boundary-isolation.AllowedDirectories: 'tidy/test/'}}
// The same shapes as boundary_isolation.cpp, in a file this configuration
// declares to be boundary code: nothing fires.
#include "llama.h"

struct Holder {
    llama_context* context;  // negative: boundary code may hold native types
};

llama_context* handout(Holder& holder) { return holder.context; }

void drive(llama_context* context) { llama_synchronize(context); }

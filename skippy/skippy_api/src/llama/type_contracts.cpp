// Compiles the llama-side type headers into the library. They are header-only;
// these includes are the compile check, and the pins inside them are the rule
// (docs/TYPE_DESIGN.md T2/T3/T4/T7).
#include "llama/batch_limit.h"
#include "llama/classifier_head.h"
#include "llama/context_capacity.h"
#include "llama/sized_type.h"

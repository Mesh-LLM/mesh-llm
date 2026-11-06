// CBMC spike (VERIFICATION.md §S): the clamped StagePlan::view in plain C.
// Proves: for every ArenaSlice {offset, length} over an arena of at most 64
// bytes, the clamped view stays inside [0, arena_size). The property in
// tests/properties_test.cpp pins the same law with random inputs; this pins
// it for ALL inputs up to the bound.
//
// Run: cbmc verification/view_clamp_harness.c --pointer-check --unwind 66
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

size_t nondet_size(void);
uint32_t nondet_u32(void);

int main(void) {
    char arena[64];
    const size_t arena_size = nondet_size();
    const uint32_t offset = nondet_u32();
    const uint32_t length = nondet_u32();
    __CPROVER_assume(arena_size <= 64);

    // The clamp under proof, mirroring StagePlan::view exactly.
    const size_t at = offset > arena_size ? arena_size : (size_t)offset;
    const size_t rest = arena_size - at;
    const size_t len = length > rest ? rest : (size_t)length;

    // Touch every byte the view exposes: an out-of-bounds read fails here
    // under --pointer-check.
    for (size_t i = 0; i < len; ++i) {
        const volatile char byte = arena[at + i];
        (void)byte;
    }
    assert(at <= arena_size);
    assert(at + len <= arena_size);
    return 0;
}

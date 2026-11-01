#ifndef SKIPPY_FFI_BINDGEN_MTMD_WRAPPER_H
#define SKIPPY_FFI_BINDGEN_MTMD_WRAPPER_H

/*
 * Narrow bindgen entry point for the llama.cpp mtmd (multimodal) surface
 * Skippy consumes: context parameters, tokenizer input, media decode helpers,
 * and generated-audio helpers.
 *
 * The bindgen allowlist in bindgen/generator.rs selects exactly the mtmd_*
 * symbols; keep this wrapper limited to the consumed headers.
 */

#include "mtmd.h"
#include "mtmd-helper.h"

#endif /* SKIPPY_FFI_BINDGEN_MTMD_WRAPPER_H */

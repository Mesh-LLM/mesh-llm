#ifndef SKIPPY_FFI_BINDGEN_LLAMA_UPSTREAM_WRAPPER_H
#define SKIPPY_FFI_BINDGEN_LLAMA_UPSTREAM_WRAPPER_H

/*
 * Narrow bindgen entry point for the upstream llama.cpp surface Skippy
 * consumes directly (logging, quantization, vocabulary probes, embedding
 * output, GGUF metadata reads, and graph-reuse performance counters).
 *
 * The bindgen allowlist in bindgen/generator.rs selects exactly the symbols
 * below; this wrapper exists so the include set is explicit and reviewed.
 * Do not widen it to all of llama.cpp: unsupported declarations must stay
 * out of the generated ABI surface.
 */

#include "ggml.h"
#include "llama.h"

#endif /* SKIPPY_FFI_BINDGEN_LLAMA_UPSTREAM_WRAPPER_H */

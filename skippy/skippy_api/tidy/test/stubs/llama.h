#pragma once
#include <cstdint>
struct llama_context;
struct llama_model;
struct llama_memory_i;
using llama_memory_t = llama_memory_i*;
extern "C" {
void llama_synchronize(llama_context*);
llama_memory_t llama_get_memory(const llama_context*);
void llama_memory_clear(llama_memory_t, bool);
std::uint32_t llama_n_ctx(const llama_context*);
std::uint32_t llama_n_batch(const llama_context*);
std::uint32_t llama_n_ubatch(const llama_context*);
std::uint32_t llama_n_seq_max(const llama_context*);
std::uint32_t llama_model_n_cls_out(const llama_model*);
const float* llama_get_embeddings_seq(llama_context*, int);
int llama_decode(llama_context*);
}

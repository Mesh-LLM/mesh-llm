// Compares trunk predictions for the same tokens decoded one at a time versus
// as one verify-sized batch, after an identical prompt.
//
//   batch_invariance_probe <model.gguf> <tokens.i32> <prompt_len> <batch>
#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

static std::vector<float> decode_rows(llama_context * ctx, const std::vector<llama_token> & tokens, int pos0,
        int n_vocab) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos0 + static_cast<int>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); std::exit(1); }
    std::vector<float> out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, static_cast<int32_t>(i));
        out.insert(out.end(), row, row + n_vocab);
    }
    llama_batch_free(batch);
    return out;
}

int main(int argc, char ** argv) {
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    std::vector<llama_token> tokens(static_cast<size_t>(in.tellg()) / sizeof(llama_token));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
    const int prompt_len = std::atoi(argv[3]);
    const int width = std::atoi(argv[4]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = std::getenv("PROBE_CPU") ? 0 : -1;
    if (std::getenv("PROBE_NO_REPACK")) mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(tokens.size() + 16);
    cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(tokens.size());
    if (std::getenv("PROBE_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);

    int windows = 0, flips = 0;
    double max_rel = 0.0;
    for (int start = prompt_len; start + width <= static_cast<int>(tokens.size()); start += width) {
        const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + start);
        const std::vector<llama_token> window(tokens.begin() + start, tokens.begin() + start + width);
        llama_memory_clear(llama_get_memory(ctx), true);
        decode_rows(ctx, prefix, 0, n_vocab);
        std::vector<float> serial;
        for (int i = 0; i < width; ++i) {
            auto row = decode_rows(ctx, {window[i]}, start + i, n_vocab);
            serial.insert(serial.end(), row.begin(), row.end());
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        decode_rows(ctx, prefix, 0, n_vocab);
        const auto batched = decode_rows(ctx, window, start, n_vocab);
        for (int i = 0; i < width; ++i) {
            const float * s = serial.data() + static_cast<size_t>(i) * n_vocab;
            const float * b = batched.data() + static_cast<size_t>(i) * n_vocab;
            double diff = 0.0, scale = 0.0;
            for (int v = 0; v < n_vocab; ++v) {
                diff = std::max(diff, static_cast<double>(std::fabs(s[v] - b[v])));
                scale = std::max(scale, static_cast<double>(std::fabs(s[v])));
            }
            max_rel = std::max(max_rel, diff / scale);
            flips += (std::max_element(s, s + n_vocab) - s) != (std::max_element(b, b + n_vocab) - b);
        }
        ++windows;
        if (windows >= 40) break;
    }
    std::printf("{\"windows\": %d, \"width\": %d, \"argmax_flips\": %d, \"max_rel_logit_diff\": %.3g}\n",
            windows, width, flips, max_rel);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

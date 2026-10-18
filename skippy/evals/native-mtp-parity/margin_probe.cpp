// Prints the trunk's logits for two candidate tokens after a prefix, decoded
// the way an unspeculated generation decodes it: the prompt in one batch, then
// each later token on its own.
//
//   margin_probe <model.gguf> <tokens.i32> <prompt_len> <token_a> <token_b>
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    std::ifstream in(argv[2], std::ios::binary | std::ios::ate);
    std::vector<llama_token> tokens(static_cast<size_t>(in.tellg()) / sizeof(llama_token));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
    const int prompt_len = std::atoi(argv[3]);
    const llama_token a = std::atoi(argv[4]), b = std::atoi(argv[5]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(tokens.size() + 16);
    cp.n_batch = cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    const float * logits = nullptr;
    for (int start = 0; start < static_cast<int>(tokens.size());) {
        const int n = start < prompt_len ? std::min(512, prompt_len - start) : 1;
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == n - 1;
        }
        if (llama_decode(ctx, batch) != 0) return 1;
        llama_batch_free(batch);
        start += n;
        logits = llama_get_logits_ith(ctx, -1);
    }
    const int best = static_cast<int>(std::max_element(logits, logits + n_vocab) - logits);
    std::vector<float> sorted(logits, logits + n_vocab);
    std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
    std::printf("{\"argmax\": %d, \"logit_a\": %.4f, \"logit_b\": %.4f, \"top1\": %.4f, \"top2\": %.4f}\n",
            best, logits[a], logits[b], sorted[0], sorted[1]);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

// Dumps a native MTP head's inputs and outputs for an offline parity check.
//
//   mtp_parity_probe <model.gguf> <tokens.i32> <out-dir>
//
// Runs the trunk over the tokens with unmasked NextN outputs and writes:
//   h.f32        [T, d]      trunk hidden state fed to the MTP head
//   logits1.f32  [T-1, V]    depth-1 MTP logits, row i = (h_i, token_{i+1}) at position i
//   z1.f32       [T-1, d]    depth-1 MTP NextN hidden state (the chain input)
//   logits2.f32  [T-2, V]    depth-2 logits, row i = (z1_i, token_{i+2}) at position i
// All rows are teacher-forced, matching splitcoder.mtp_eval.
#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void die(const char * msg) {
    std::fprintf(stderr, "mtp_parity_probe: %s\n", msg);
    std::exit(1);
}

static void write_rows(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) die(("cannot write " + path).c_str());
}

// Decodes MTP rows (hidden[i], tokens[i]) at positions 0..n-1 into a cleared
// MTP context and returns the logits and NextN hidden rows.
static void run_mtp(llama_context * mtp, int32_t d, int32_t n_vocab,
        const std::vector<float> & hidden, const std::vector<llama_token> & tokens,
        std::vector<float> & logits, std::vector<float> & next_hidden) {
    const int32_t n = static_cast<int32_t>(tokens.size());
    llama_memory_clear(llama_get_memory(mtp), true);
    llama_batch batch = llama_batch_init(n, d, 1);
    batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token) * n));
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        std::copy(hidden.begin() + static_cast<size_t>(i) * d, hidden.begin() + static_cast<size_t>(i + 1) * d,
                batch.embd + static_cast<size_t>(i) * d);
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(mtp, batch) != 0) die("MTP decode failed");
    logits.resize(static_cast<size_t>(n) * n_vocab);
    next_hidden.resize(static_cast<size_t>(n) * d);
    for (int32_t i = 0; i < n; ++i) {
        const float * row = llama_get_logits_ith(mtp, i);
        const float * h = llama_get_embeddings_nextn_ith(mtp, i);
        if (row == nullptr || h == nullptr) die("missing MTP output row");
        std::copy(row, row + n_vocab, logits.begin() + static_cast<size_t>(i) * n_vocab);
        std::copy(h, h + d, next_hidden.begin() + static_cast<size_t>(i) * d);
    }
    std::free(batch.token);
    batch.token = nullptr;
    llama_batch_free(batch);
}

int main(int argc, char ** argv) {
    if (argc != 4) die("usage: mtp_parity_probe <model.gguf> <tokens.i32> <out-dir>");
    const std::string out_dir = argv[3];

    std::ifstream tok_file(argv[2], std::ios::binary | std::ios::ate);
    if (!tok_file) die("cannot read tokens");
    std::vector<llama_token> tokens(static_cast<size_t>(tok_file.tellg()) / sizeof(llama_token));
    tok_file.seekg(0);
    tok_file.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
    const int32_t T = static_cast<int32_t>(tokens.size());
    if (T < 3) die("need at least 3 tokens");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    mp.load_mtp = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) die("model load failed");
    const int32_t d = llama_model_n_embd(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(T + 16);
    cp.n_batch = cp.n_ubatch = static_cast<uint32_t>(T);
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) die("trunk context failed");
    llama_set_embeddings_nextn(ctx, true, false);

    llama_context_params mcp = cp;
    mcp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * mtp = llama_init_from_model(model, mcp);
    if (mtp == nullptr) die("MTP context failed");
    llama_set_embeddings_nextn(mtp, true, false);

    llama_batch batch = llama_batch_init(T, 0, 1);
    batch.n_tokens = T;
    for (int32_t i = 0; i < T; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) die("trunk decode failed");
    llama_batch_free(batch);

    std::vector<float> h(static_cast<size_t>(T) * d);
    for (int32_t i = 0; i < T; ++i) {
        const float * row = llama_get_embeddings_nextn_ith(ctx, i);
        if (row == nullptr) die("missing trunk hidden row");
        std::copy(row, row + d, h.begin() + static_cast<size_t>(i) * d);
    }
    write_rows(out_dir + "/h.f32", h);

    std::vector<float> logits1, z1;
    run_mtp(mtp, d, n_vocab, h, std::vector<llama_token>(tokens.begin() + 1, tokens.end()), logits1, z1);
    write_rows(out_dir + "/logits1.f32", logits1);
    write_rows(out_dir + "/z1.f32", z1);

    std::vector<float> logits2, z2;
    z1.resize(static_cast<size_t>(T - 2) * d);
    run_mtp(mtp, d, n_vocab, z1, std::vector<llama_token>(tokens.begin() + 2, tokens.end()), logits2, z2);
    write_rows(out_dir + "/logits2.f32", logits2);

    std::printf("{\"tokens\": %d, \"n_embd\": %d, \"n_vocab\": %d}\n", T, d, n_vocab);
    llama_free(mtp);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

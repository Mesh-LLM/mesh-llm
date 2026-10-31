#include "llama.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
const std::array<ggml_type, 8> kv_types = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0};

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

ggml_type parse_type(const char * name) {
    for (auto type : kv_types) {
        if (std::strcmp(ggml_type_name(type), name) == 0) return type;
    }
    throw std::runtime_error(std::string("unsupported matrix KV type: ") + name);
}

int inspect(llama_context * ctx, ggml_type type_k, ggml_type type_v,
            const char * expected_backend, int step) {
    const auto result = ctx->get_gf_res_prev_active();
    require(result != nullptr, "no active real model graph");
    const auto graph = result->get_gf();
    int attention = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
        const auto backend = ggml_backend_sched_get_tensor_backend(ctx->get_sched(), node);
        require(backend && std::strcmp(ggml_backend_name(backend), expected_backend) == 0,
                "real attention executed on the wrong backend");
        require(node->src[5] && node->src[6], "real flash attention missing request indices/lengths");
        require(node->src[1]->type == type_k && node->src[2]->type == type_v, "actual K/V dtype changed");
        ++attention;
        if (step == 0 && attention == 1) {
            std::printf("REAL_FA backend=%s K=%s V=%s q=%lld physical=%lld logical=%lld requests=%lld\n",
                    ggml_backend_name(backend), ggml_type_name(type_k), ggml_type_name(type_v),
                    (long long) node->src[0]->ne[1], (long long) node->src[1]->ne[1],
                    (long long) node->src[5]->ne[0], (long long) node->src[5]->ne[3]);
        }
    }
    require(attention > 0, "no real flash attention operations");
    return attention;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab) {
    const std::string prompt =
        "Once upon a time, a small bear found a red ball. "
        "The bear carried the ball home and showed it to a friend. "
        "Together they made up a game and played happily in the garden.";
    std::vector<llama_token> tokens(prompt.size() + 16);
    const auto count = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
            tokens.data(), (int) tokens.size(), true, true);
    require(count >= 20 && count <= 64, "prompt must fit one wide-prefill microbatch (20..64 tokens)");
    tokens.resize(count);
    return tokens;
}

const float * finite_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing logits");
    require(std::all_of(logits, logits + n_vocab,
                [](float value) { return std::isfinite(value); }), "nonfinite output logits");
    return logits;
}

void run_pair(llama_model * model, const std::vector<llama_token> & prompt,
              const char * backend, ggml_type type_k, ggml_type type_v) {
    const auto start = clock_type::now();
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 128;
    params.n_ubatch = 64;
    params.n_seq_max = 1;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = type_k;
    params.type_v = type_v;
    using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    context_ptr ctx(llama_init_from_model(model, params), llama_free);
    require(ctx != nullptr, "context open failed (model/KV semantics or backend admission)");
    // The public batch borrows its token array; decoding does not mutate it.
    auto tokens = prompt;
    require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), (int32_t) tokens.size())) == 0,
            "prompt decode failed");
    int expected_attention = 0;
    for (uint32_t layer = 0; layer < model->hparams.n_layer(); ++layer) {
        expected_attention += !model->hparams.is_recr(layer);
    }
    const int attention = inspect(ctx.get(), type_k, type_v, backend, 0);
    require(attention == expected_attention,
            "not every attention layer used the required indexed backend");
    const auto vocab = llama_model_get_vocab(model);
    const auto n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = finite_logits(ctx.get(), n_vocab);
    std::string completion;
    for (int step = 0; step < 16; ++step) {
        auto token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
        char piece[256];
        const int bytes = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        require(bytes >= 0, "token piece buffer too small");
        completion.append(piece, bytes);
        require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "greedy decode failed");
        require(inspect(ctx.get(), type_k, type_v, backend, step + 1) == attention,
                "attention layer count changed during decode");
        logits = finite_logits(ctx.get(), n_vocab);
    }
    require(!completion.empty(), "empty completion");
    const auto elapsed = std::chrono::duration<double>(clock_type::now() - start).count();
    std::printf("PASS backend=%s K=%s V=%s prefill=%zu decoded=16 attention_layers=%d seconds=%.6f completion_hex=",
            backend, ggml_type_name(type_k), ggml_type_name(type_v), tokens.size(), attention, elapsed);
    for (unsigned char value : completion) std::printf("%02x", value);
    std::printf("\n");
    std::fflush(stdout);
}
} // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    try {
        require(argc == 3 || argc == 4 || argc == 5,
                "usage: indexed-real-kv-matrix MODEL CUDA0|MTL0|CPU [K_TYPE [V_TYPE]]");
        const char * backend = argv[2];
        const bool cpu = std::strcmp(backend, "CPU") == 0;
        const bool metal = std::strcmp(backend, "MTL0") == 0;
        require(cpu || metal || std::strcmp(backend, "CUDA0") == 0, "select CPU, MTL0 or CUDA0 explicitly");
        ggml_backend_load_all();
        const auto device = ggml_backend_dev_by_name(backend);
        require(device != nullptr, "selected backend unavailable");
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto params = llama_model_default_params();
        params.devices = cpu ? nullptr : devices;
        params.n_gpu_layers = cpu ? 0 : 99;
        const auto load_start = clock_type::now();
        model = llama_model_load_from_file(argv[1], params);
        require(model != nullptr, "model open failed");
        const auto tokens = tokenize(llama_model_get_vocab(model));
        std::printf("MODEL backend=%s path=%s loads=1 seconds=%.6f prefill=%zu ctx=512 batch=128 ubatch=64 output=16\n",
                backend, argv[1], std::chrono::duration<double>(clock_type::now() - load_start).count(), tokens.size());
        std::vector<std::array<ggml_type, 2>> pairs;
        int skipped = 0;
        if (argc >= 4) {
            const auto k = parse_type(argv[3]);
            pairs.push_back({k, argc == 5 ? parse_type(argv[4]) : k});
        } else {
            for (auto k : kv_types) {
                for (auto v : kv_types) {
                    if (metal && k != v) {
                        std::printf("SKIP backend=MTL0 K=%s V=%s reason=preexisting_Metal_requires_matching_KV_types\n",
                                ggml_type_name(k), ggml_type_name(v));
                        ++skipped;
                    } else {
                        pairs.push_back({k, v});
                    }
                }
            }
        }
        int passed = 0;
        int failed = 0;
        for (const auto & pair : pairs) {
            try {
                run_pair(model, tokens, backend, pair[0], pair[1]);
                ++passed;
            } catch (const std::exception & error) {
                std::printf("FAIL backend=%s K=%s V=%s reason=%s\n", backend,
                        ggml_type_name(pair[0]), ggml_type_name(pair[1]), error.what());
                std::fflush(stdout);
                ++failed;
            }
        }
        std::printf("SUMMARY backend=%s model_loads=1 passed=%d failed=%d skipped=%d\n", backend, passed, failed, skipped);
        llama_model_free(model);
        llama_backend_free();
        return failed == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "matrix setup failed: %s\n", error.what());
        if (model) llama_model_free(model);
        llama_backend_free();
        return 2;
    }
}

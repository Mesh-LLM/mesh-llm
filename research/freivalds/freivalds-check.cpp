/*
 * freivalds-check: Measure Freivalds verification residuals on llama.cpp inference.
 *
 * This tool runs a single forward pass through a GGUF model and captures
 * intermediate activation tensors via the eval callback. It then performs
 * Freivalds-style algebraic checks on CPU to measure the residual error
 * between GPU-computed matmul outputs and CPU-verified dot products.
 *
 * The goal: determine whether the floating-point error from different
 * accumulation orders (GPU vs CPU) is small enough that Freivalds verification
 * can distinguish "honest computation with right weights" from "computation
 * with wrong weights."
 *
 * Usage:
 *   llama-freivalds-check -m model.gguf [-p "prompt text"] [-ngl N]
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

// Internal API — declared in llama-model.h but not publicly exposed
extern const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model);

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

// ── Captured tensor data ──

struct captured_tensor {
    std::string name;
    int         il;           // layer index (-1 for non-layer tensors)
    enum ggml_type type;
    int64_t     ne[4];        // dimensions
    std::vector<uint8_t> data; // raw tensor data copied from backend
};

// Global capture state
static std::map<std::string, captured_tensor> g_captures;
static int g_target_layer = 0;  // which layer to capture

// Tensor names we want to capture for the Freivalds check on down_proj:
//   "ffn_swiglu-L"  = input to down_proj matmul (after SiLU gate * up)
//   "ffn_out-L"     = output after down_proj + residual add
//   "ffn_norm-L"    = input to the FFN (after norm, before gate/up projections)
// We also need the raw matmul output before the residual add.
// The named tensors available in the Llama graph:
//   "ffn_norm-L"    = after RMSNorm, input to gate_proj and up_proj
//   "ffn_gate-L"    = gate_proj output
//   "ffn_up-L"      = up_proj output
//   "ffn_swiglu-L"  = SiLU(gate) * up — this is the INPUT to down_proj
//   "ffn_out-L"     = after down_proj + residual — NOT the raw down_proj output
//
// For Freivalds on down_proj, we need:
//   x = ffn_swiglu-L (input to down_proj)
//   y = down_proj(x) = W_down · x (raw matmul output, BEFORE residual add)
//
// Problem: the raw matmul output isn't named separately in the graph.
// The "ffn_out-L" tensor = down_proj(swiglu) + ffn_inp (residual).
//
// We can work around this:
//   y_raw = ffn_out - ffn_inp
// where ffn_inp is the residual connection input.
// Or we capture "ffn_inp-L" and subtract.
//
// Actually, looking at the Llama model code more carefully:
//   ffn_inp = attn_out + inpSA  (residual from attention)
//   cur = build_ffn(...)         (this is the raw FFN output including down_proj)
//   cur = cur + ffn_inp          (residual add, then named "ffn_out")
//   BUT: the "ffn_out" cb() is called AFTER the residual add in llama.cpp
//
// Wait — re-reading the code: in llama.cpp model, build_ffn returns the raw
// down_proj output. Then the model code does:
//   cur = ggml_add(ctx0, cur, ffn_inp);
//   cb(cur, "ffn_out", il);
// So "ffn_out" includes the residual. We need the tensor BEFORE that add.
//
// BUT: build_ffn internally has:
//   cur = build_lora_mm(down, cur);
//   // no cb() call for the raw down_proj output!
// There IS a cb(cur, "ffn_down", il) but only when down_b exists (bias).
//
// For standard Llama models (no FFN bias), the raw down_proj output is unnamed.
// We'll need to capture both "ffn_swiglu-L" and "ffn_inp-L" and "ffn_out-L",
// then compute: y_raw = ffn_out - ffn_inp
//
// For the Q/K/V projections it's easier — "Qcur-L" IS the raw matmul output
// (before RoPE and reshape, but the first cb() call is right after the matmul).
// And the input to Q/K/V is "attn_norm-L".
//
// So let's check the Q projection as our primary test:
//   x = attn_norm-L  (input to Wq matmul)
//   y = Qcur-L       (output of Wq matmul, before RoPE)
//   W = model.layers[L].wq
//   Check: v · y ≈ (v · W) · x

// Names to capture for layer L
static std::vector<std::string> make_capture_names(int layer) {
    std::vector<std::string> names;
    // For Q projection check
    names.push_back("attn_norm-" + std::to_string(layer));
    names.push_back("Qcur-" + std::to_string(layer));
    // For FFN down_proj check (via residual subtraction)
    names.push_back("ffn_inp-" + std::to_string(layer));
    names.push_back("ffn_swiglu-" + std::to_string(layer));
    names.push_back("ffn_out-" + std::to_string(layer));
    return names;
}

static std::vector<std::string> g_capture_names;

// ── Eval callback ──

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;

    const std::string name(t->name);

    // Check if this is a tensor we want
    bool wanted = false;
    for (const auto & cn : g_capture_names) {
        if (name == cn) {
            wanted = true;
            break;
        }
    }

    if (!wanted) return true; // continue, but don't need data

    if (ask) {
        // Only request data if we haven't already captured this name
        // (same name can appear multiple times due to reshape/bias ops)
        if (g_captures.count(name)) return true; // already have it, skip
        return true; // yes, want data
    }

    // ask == false: data is ready, copy it
    // Skip if already captured (first occurrence wins — that's the raw matmul output)
    if (g_captures.count(name)) return true;

    captured_tensor cap;
    cap.name = name;
    cap.type = t->type;
    for (int i = 0; i < 4; i++) cap.ne[i] = t->ne[i];

    size_t nbytes = ggml_nbytes(t);
    cap.data.resize(nbytes);
    ggml_backend_tensor_get(t, cap.data.data(), 0, nbytes);

    g_captures[name] = std::move(cap);

    LOG_INF("captured tensor: %s [%lld x %lld x %lld x %lld] type=%d (%zu bytes)\n",
            name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], (int)t->type, nbytes);

    return true; // continue computation
}

// ── Tensor data access helpers ──

// Get float value from a tensor (handles F32 and F16 types)
static float get_float(const captured_tensor & t, int64_t idx) {
    if (t.type == GGML_TYPE_F32) {
        return ((const float *)t.data.data())[idx];
    } else if (t.type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(((const ggml_fp16_t *)t.data.data())[idx]);
    } else if (t.type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(((const ggml_bf16_t *)t.data.data())[idx]);
    }
    LOG_ERR("unsupported tensor type %d for get_float\n", (int)t.type);
    return 0.0f;
}

// ── Weight data cache ──
// Weight tensors may be on GPU; we cache a CPU copy of the full tensor
static std::map<const ggml_tensor *, std::vector<uint8_t>> g_weight_cache;

static const void * get_weight_data(const ggml_tensor * w) {
    auto it = g_weight_cache.find(w);
    if (it != g_weight_cache.end()) {
        return it->second.data();
    }

    // Always use ggml_backend_tensor_get — handles repacked/GPU buffers correctly
    size_t nbytes = ggml_nbytes(w);
    LOG_INF("  weight %s: reading %zu bytes via backend API...\n", w->name, nbytes);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(w, buf.data(), 0, nbytes);
    g_weight_cache[w] = std::move(buf);
    return g_weight_cache[w].data();
}

// ── Dequantize weight row ──
// Given a quantized weight tensor, dequantize one row to FP32
static std::vector<float> dequant_row(const ggml_tensor * w, int64_t row) {
    int64_t ne0 = w->ne[0]; // number of columns
    std::vector<float> out(ne0);

    const void * base = get_weight_data(w);
    // nb[1] is the stride in bytes between rows
    // For quantized types this accounts for block structure
    size_t row_offset = row * w->nb[1];
    const void * row_data = (const char *)base + row_offset;

    // Use ggml's type traits for dequantization
    const auto * type_traits = ggml_get_type_traits(w->type);
    if (type_traits && type_traits->to_float) {
        type_traits->to_float(row_data, out.data(), ne0);
    } else if (w->type == GGML_TYPE_F32) {
        memcpy(out.data(), row_data, ne0 * sizeof(float));
    } else if (w->type == GGML_TYPE_F16) {
        for (int64_t i = 0; i < ne0; i++) {
            out[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *)row_data)[i]);
        }
    } else {
        LOG_ERR("unsupported weight type %d for dequant_row\n", (int)w->type);
    }

    return out;
}

// ── Freivalds check ──
// Given:
//   x: input vector (length K)
//   y: output vector (length M) — claimed to be W·x
//   W: weight matrix (M x K, stored as ggml_tensor)
//   v: random vector (length M) — the verifier's secret
// Check: v·y ≈ (v·W)·x
// Returns the relative residual |v·y - (v·W)·x| / |v·y|

struct freivalds_result {
    double lhs;              // v · y
    double rhs;              // (v · W) · x
    double abs_residual;     // |lhs - rhs|
    double rel_residual;     // |lhs - rhs| / max(|lhs|, |rhs|)
    int64_t M;               // output dimension
    int64_t K;               // input dimension
};

static freivalds_result freivalds_check(
    const captured_tensor & x_cap,  // input activation [K, n_tokens, ...]
    const captured_tensor & y_cap,  // output activation [M, n_tokens, ...]
    const ggml_tensor * W,          // weight matrix [K, M] in ggml layout (transposed)
    int token_idx                   // which token position to check
) {
    // In ggml, mul_mat(W, x) computes x @ W^T (or equivalently W @ x in math notation)
    // W shape: [K, M] where K = ne[0], M = ne[1]
    // x shape: [K, n_tokens]
    // y shape: [M, n_tokens]
    int64_t K = W->ne[0];
    int64_t M = W->ne[1];

    LOG_INF("Freivalds check: W=[%lld x %lld], x=[%lld x %lld], y=[%lld x %lld], token=%d\n",
            (long long)K, (long long)M,
            (long long)x_cap.ne[0], (long long)x_cap.ne[1],
            (long long)y_cap.ne[0], (long long)y_cap.ne[1],
            token_idx);

    // Generate random vector v of length M (the verifier's secret)
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(M);
    for (int64_t i = 0; i < M; i++) {
        v[i] = dist(rng);
    }

    // Extract x for this token position: x[0..K-1]
    // x_cap layout: [ne0, n_tokens] where ne0 = K
    std::vector<float> x_vec(K);
    for (int64_t i = 0; i < K; i++) {
        x_vec[i] = get_float(x_cap, token_idx * x_cap.ne[0] + i);
    }

    // Extract y for this token position: y[0..M-1]
    std::vector<float> y_vec(M);
    for (int64_t i = 0; i < M; i++) {
        y_vec[i] = get_float(y_cap, token_idx * y_cap.ne[0] + i);
    }

    // Compute LHS: v · y (dot product, scalar)
    double lhs = 0.0;
    for (int64_t i = 0; i < M; i++) {
        lhs += (double)v[i] * (double)y_vec[i];
    }

    // Compute r = v · W (vector of length K)
    // W is [K, M] in ggml (each row of length K, M rows)
    // v · W means: for each column j of W, sum_i v[i] * W[i][j]
    // Since W is stored as M rows of K elements: r[j] = sum_i v[i] * W_row_i[j]
    LOG_INF("  computing v·W (%lld rows x %lld cols), W nb=[%zu, %zu, %zu], type=%d...\n",
            (long long)M, (long long)K,
            W->nb[0], W->nb[1], W->nb[2], (int)W->type);
    auto t_start_vw = std::chrono::high_resolution_clock::now();

    // Verify the first dequantized row looks sane
    {
        std::vector<float> test_row = dequant_row(W, 0);
        float row_sum = 0, row_abs_max = 0;
        int nan_count = 0;
        for (int64_t j = 0; j < K; j++) {
            if (std::isnan(test_row[j])) nan_count++;
            row_sum += test_row[j];
            if (fabs(test_row[j]) > row_abs_max) row_abs_max = fabs(test_row[j]);
        }
        LOG_INF("  row 0: sum=%.4f, abs_max=%.4f, nans=%d/%lld\n",
                row_sum, row_abs_max, nan_count, (long long)K);
    }

    std::vector<double> r(K, 0.0);
    for (int64_t i = 0; i < M; i++) {
        std::vector<float> w_row = dequant_row(W, i);
        for (int64_t j = 0; j < K; j++) {
            r[j] += (double)v[i] * (double)w_row[j];
        }
    }
    auto t_end_vw = std::chrono::high_resolution_clock::now();
    double vw_ms = std::chrono::duration<double, std::milli>(t_end_vw - t_start_vw).count();
    LOG_INF("  v·W took %.1f ms\n", vw_ms);

    // Compute RHS: r · x (dot product, scalar)
    double rhs = 0.0;
    for (int64_t j = 0; j < K; j++) {
        rhs += r[j] * (double)x_vec[j];
    }

    double abs_res = fabs(lhs - rhs);
    double denom = fmax(fabs(lhs), fabs(rhs));
    double rel_res = (denom > 1e-30) ? abs_res / denom : abs_res;

    return { lhs, rhs, abs_res, rel_res, M, K };
}

// ── Main ──

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "The quick brown fox jumps over the lazy dog";
    params.n_predict = 1; // just one token — we only need the forward pass

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.prompt.empty()) {
        params.prompt = "The quick brown fox jumps over the lazy dog";
    }

    common_init();

    // Set up eval callback BEFORE creating context
    params.cb_eval = eval_callback;
    params.cb_eval_user_data = nullptr;

    // Initialize model
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model: %s\n", params.model.path.c_str());
        return 1;
    }

    // Get model info
    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    LOG_INF("Model: %d layers, %d embd\n", n_layer, n_embd);

    // Check every 4th layer + first and last
    std::vector<int> layers_vec;
    layers_vec.push_back(0);
    for (int i = 4; i < n_layer - 1; i += 4) layers_vec.push_back(i);
    if (layers_vec.back() != n_layer - 1) layers_vec.push_back(n_layer - 1);
    int * layers_to_check = layers_vec.data();
    int n_check = (int)layers_vec.size();

    // Create context
    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);
    LOG_INF("Prompt: \"%s\" (%zu tokens)\n", params.prompt.c_str(), tokens.size());

    // Results storage
    struct layer_result {
        int layer;
        std::string proj_name;
        freivalds_result fr;
    };
    std::vector<layer_result> results;

    // Run forward pass for each target layer
    for (int ci = 0; ci < n_check; ci++) {
        g_target_layer = layers_to_check[ci];
        g_capture_names = make_capture_names(g_target_layer);
        g_captures.clear();

        LOG_INF("\n=== Checking layer %d ===\n", g_target_layer);

        // Reset context KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Run forward pass
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            common_batch_add(batch, tokens[i], i, {0}, (i == tokens.size() - 1));
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("failed to decode\n");
            llama_batch_free(batch);
            continue;
        }
        llama_batch_free(batch);

        LOG_INF("Captured %zu tensors\n", g_captures.size());

        // ── Check Q projection: attn_norm -> Wq -> Qcur ──
        std::string attn_norm_name = "attn_norm-" + std::to_string(g_target_layer);
        std::string qcur_name = "Qcur-" + std::to_string(g_target_layer);

        if (g_captures.count(attn_norm_name) && g_captures.count(qcur_name)) {
            // Get the weight tensor for Wq at this layer
            // We need to access model internals... use ggml_graph_get_tensor or
            // the model's named tensors
            char wq_name[64];
            snprintf(wq_name, sizeof(wq_name), "blk.%d.attn_q.weight", g_target_layer);
            const ggml_tensor * wq = nullptr;
            {
                const auto & tmap = llama_internal_get_tensor_map(model);
                for (const auto & p : tmap) {
                    if (p.first == wq_name) { wq = p.second; break; }
                }
            }

            if (wq) {
                LOG_INF("Wq tensor: [%lld x %lld], type=%d\n",
                        (long long)wq->ne[0], (long long)wq->ne[1], (int)wq->type);

                const auto & x_cap = g_captures[attn_norm_name];
                const auto & y_cap = g_captures[qcur_name];

                // Check first token position only (speed)
                int n_tokens_check = 1;
                for (int tok = 0; tok < n_tokens_check; tok++) {
                    freivalds_result fr = freivalds_check(x_cap, y_cap, wq, tok);
                    results.push_back({g_target_layer, "Wq", fr});

                    LOG_INF("  Layer %d Wq token %d: LHS=%.6e RHS=%.6e abs=%.6e rel=%.6e\n",
                            g_target_layer, tok, fr.lhs, fr.rhs, fr.abs_residual, fr.rel_residual);
                }
            } else {
                LOG_ERR("  Could not find weight tensor %s\n", wq_name);
            }
        } else {
            LOG_ERR("  Missing captures for Q projection check (have attn_norm=%d, Qcur=%d)\n",
                    (int)g_captures.count(attn_norm_name), (int)g_captures.count(qcur_name));
        }

        // ── Check down_proj: ffn_swiglu -> W_down -> (ffn_out - ffn_inp) ──
        std::string swiglu_name = "ffn_swiglu-" + std::to_string(g_target_layer);
        std::string ffn_out_name = "ffn_out-" + std::to_string(g_target_layer);
        std::string ffn_inp_name = "ffn_inp-" + std::to_string(g_target_layer);

        if (g_captures.count(swiglu_name) && g_captures.count(ffn_out_name)) {

            char wd_name[64];
            snprintf(wd_name, sizeof(wd_name), "blk.%d.ffn_down.weight", g_target_layer);
            const ggml_tensor * wd = nullptr;
            {
                const auto & tmap = llama_internal_get_tensor_map(model);
                for (const auto & p : tmap) {
                    if (p.first == wd_name) { wd = p.second; break; }
                }
            }

            if (wd) {
                LOG_INF("W_down tensor: [%lld x %lld], type=%d\n",
                        (long long)wd->ne[0], (long long)wd->ne[1], (int)wd->type);

                const auto & x_cap = g_captures[swiglu_name];
                const auto & y_cap = g_captures[ffn_out_name];

                // For Qwen2 (and most models), "ffn_out" is the raw down_proj output
                // (before residual add). For Llama models where "ffn_out" includes
                // the residual, we'd need to subtract ffn_inp — but we try direct first.

                int n_tokens_check = 1;
                for (int tok = 0; tok < n_tokens_check; tok++) {
                    freivalds_result fr = freivalds_check(x_cap, y_cap, wd, tok);
                    results.push_back({g_target_layer, "W_down", fr});

                    LOG_INF("  Layer %d W_down token %d: LHS=%.6e RHS=%.6e abs=%.6e rel=%.6e\n",
                            g_target_layer, tok, fr.lhs, fr.rhs, fr.abs_residual, fr.rel_residual);
                }
            } else {
                LOG_ERR("  Could not find weight tensor %s\n", wd_name);
            }
        } else {
            LOG_ERR("  Missing captures for FFN down_proj check (swiglu=%d, ffn_out=%d, ffn_inp=%d)\n",
                    (int)g_captures.count(swiglu_name),
                    (int)g_captures.count(ffn_out_name),
                    (int)g_captures.count(ffn_inp_name));
        }
    }

    // ── Summary ──
    LOG_INF("\n");
    LOG_INF("╔══════════════════════════════════════════════════════════════════════════╗\n");
    LOG_INF("║                     FREIVALDS RESIDUAL SUMMARY                          ║\n");
    LOG_INF("╠══════════════════════════════════════════════════════════════════════════╣\n");
    LOG_INF("║ Layer │ Proj   │ Token │ Dims        │ Abs Residual    │ Rel Residual   ║\n");
    LOG_INF("╠══════════════════════════════════════════════════════════════════════════╣\n");

    double max_rel = 0, max_abs = 0, sum_rel = 0;
    int n_results = 0;
    for (const auto & r : results) {
        LOG_INF("║ %5d │ %-6s │ %5d │ %4lldx%-5lld │ %15.6e │ %14.6e ║\n",
                r.layer, r.proj_name.c_str(), 0 /* tok */,
                (long long)r.fr.M, (long long)r.fr.K,
                r.fr.abs_residual, r.fr.rel_residual);
        if (r.fr.rel_residual > max_rel) max_rel = r.fr.rel_residual;
        if (r.fr.abs_residual > max_abs) max_abs = r.fr.abs_residual;
        sum_rel += r.fr.rel_residual;
        n_results++;
    }

    LOG_INF("╠══════════════════════════════════════════════════════════════════════════╣\n");
    if (n_results > 0) {
        LOG_INF("║ Max relative residual: %.6e                                       ║\n", max_rel);
        LOG_INF("║ Max absolute residual: %.6e                                       ║\n", max_abs);
        LOG_INF("║ Avg relative residual: %.6e                                       ║\n", sum_rel / n_results);
    }
    LOG_INF("║ Total checks: %d                                                        ║\n", n_results);
    LOG_INF("╚══════════════════════════════════════════════════════════════════════════╝\n");

    if (max_rel < 1e-4) {
        LOG_INF("\n✅ VERY TIGHT — Freivalds verification looks viable at this tolerance\n");
    } else if (max_rel < 1e-2) {
        LOG_INF("\n⚠️  MODERATE — Freivalds might work with careful threshold tuning\n");
    } else if (max_rel < 1e-1) {
        LOG_INF("\n⚠️  LOOSE — significant FP error, Freivalds may have false positives\n");
    } else {
        LOG_INF("\n❌ TOO LARGE — Freivalds residual too noisy for reliable verification\n");
    }

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

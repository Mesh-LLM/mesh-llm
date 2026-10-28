#include "fixtures.h"

#include <chrono>
#include <stdexcept>

void require(bool condition, const std::string & detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

native_runtime::native_runtime(const char * path) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model = llama_model_load_from_file(path, mp);
    require(model != nullptr, "model load failed");
    vocab = llama_model_get_vocab(model);
    require(llama_vocab_n_tokens(vocab) == 248320, "unexpected tokenizer vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context creation failed");
    stage.model = model;
    stage.ctx = ctx;
}

native_runtime::~native_runtime() {
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

void native_runtime::reset(const std::vector<llama_token> & prompt) {
    llama_memory_clear(llama_get_memory(ctx), true);
    auto tokens = prompt;
    require(tokens.size() < 512, "prompt exceeds prefill batch");
    auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    require(llama_decode(ctx, batch) == 0, "prompt decode failed");
    llama_synchronize(ctx);
}

void native_runtime::decode(llama_token token, int position) {
    llama_pos pos = position;
    int32_t nseq = 1;
    llama_seq_id seq = 0;
    llama_seq_id * seqs = &seq;
    int8_t logits = 1;
    llama_batch batch{1, &token, nullptr, &pos, &nseq, &seqs, &logits};
    require(llama_decode(ctx, batch) == 0, "token decode failed");
    llama_synchronize(ctx);
}

static json metadata(const common_chat_params & params) {
    json triggers = json::array();
    for (const auto & trigger : params.grammar_triggers) {
        triggers.push_back({{"type", static_cast<int>(trigger.type)},
                            {"value", trigger.value}, {"token", trigger.token}});
    }
    return {{"grammar", params.grammar}, {"grammar_lazy", params.grammar_lazy},
            {"grammar_triggers", triggers}, {"generation_prompt", params.generation_prompt},
            {"preserved_tokens", params.preserved_tokens}};
}

std::vector<fixture> make_fixtures(llama_model * model) {
    const auto * vocab = llama_model_get_vocab(model);
    auto templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg request;
    request.role = "user";
    request.content = "Write a small todo app in HTML using the write tool.";
    inputs.messages.push_back(request);
    inputs.tools.push_back({"write", "Write a text file", R"({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})"});
    inputs.enable_thinking = false;
    inputs.chat_template_kwargs["enable_thinking"] = "false";
    inputs.chat_template_kwargs["reasoning_effort"] = R"("none")";
    inputs.now = std::chrono::system_clock::time_point{};
    auto params = common_chat_templates_apply(templates.get(), inputs);
    require(!params.grammar.empty() && params.grammar_lazy, "expected lazy tool grammar");
    const std::string open = "<tool_call>\n<function=write>\n<parameter=path>\ntodo.html\n</parameter>\n<parameter=content>\n";
    const std::string close = "\n</parameter>\n</function>\n</tool_call>";
    std::string html = "<!doctype html><html><body><h1>Todo</h1><ul>";
    for (int i = 0; i < 18; ++i) {
        html += "<li><input type=checkbox>Task " + std::to_string(i) + "</li>";
    }
    html += "</ul><script>function add(){return 'new task';}</script></body></html>";
    const std::vector<std::pair<std::string, std::string>> texts = {
        {"lazy-before-tool-trigger", "I will create a small todo app with a list, checkboxes, and an input field. The page will work without a server."},
        {"large-HTML-write-argument", open + html + close},
        {"Unicode-and-partial-UTF8", open + "<h1>café 日本語 🚀</h1><p>naïve Ελληνικά Привет 🌍 \xF4\x8F\xBF\xBD</p>" + close},
        {"restricted-values-and-rejection-fallback", "true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\n"},
        {"closing-delimiters-and-EOS", open + "<p>Done</p>" + close}
    };
    std::vector<fixture> result;
    for (const auto & [name, text] : texts) {
        fixture f;
        f.name = name;
        f.metadata = metadata(params);
        // Keep inference inputs short while using the actual model-generated
        // tool grammar. This is a native sampler experiment, not agent quality.
        f.prompt = common_tokenize(vocab, "Write a small todo app in HTML.\n", true, true);
        f.continuation = common_tokenize(vocab, text, false, true);
        for (auto token : f.continuation) {
            const auto piece = common_token_to_piece(vocab, token, true);
            if (!piece.empty() && (static_cast<unsigned char>(piece.front()) & 0xc0) == 0x80) {
                f.partial_utf8_boundary = true;
            }
        }
        if (name == "Unicode-and-partial-UTF8") {
            // A rare valid private-use code point forces byte-level pieces;
            // verify the boundary rather than merely assuming emoji split.
            require(f.partial_utf8_boundary, "Unicode fixture lacks a partial UTF-8 token boundary");
        }
        f.force_rejection = name == "restricted-values-and-rejection-fallback";
        if (f.force_rejection) {
            f.metadata = {{"grammar", "root ::= (\"true\\n\"){8}"}, {"grammar_lazy", false}, {"generation_prompt", ""}};
        }
        f.include_eos = name == "closing-delimiters-and-EOS";
        if (f.include_eos) {
            f.continuation.push_back(llama_vocab_eos(vocab));
        }
        require(!f.continuation.empty() && f.continuation.size() <= 512, "fixture token bound");
        result.push_back(std::move(f));
    }
    return result;
}

skippy_sampling_config skippy_config(uint32_t seed) {
    skippy_sampling_config config{};
    config.version = 1;
    config.flags = SKIPPY_SAMPLING_FLAG_ENABLED;
    config.seed = seed;
    config.temperature = 0.8f;
    config.top_k = 40;
    config.top_p = 0.95f;
    config.min_p = 0.05f;
    config.repeat_penalty = 1.0f;
    config.reasoning_budget_tokens = -1;
    config.sampler_count = 4;
    config.samplers[0] = SKIPPY_SAMPLER_TOP_K;
    config.samplers[1] = SKIPPY_SAMPLER_TOP_P;
    config.samplers[2] = SKIPPY_SAMPLER_MIN_P;
    config.samplers[3] = SKIPPY_SAMPLER_TEMPERATURE;
    return config;
}

common_params_sampling reference_config(const fixture & input, const llama_vocab * vocab, uint32_t seed) {
    common_params_sampling params;
    params.no_perf = true;
    params.seed = seed;
    params.temp = 0.8f;
    params.top_k = 40;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.min_keep = 0;
    params.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P,
                       COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
    params.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, input.metadata.at("grammar").get<std::string>()};
    params.grammar_lazy = input.metadata.value("grammar_lazy", false);
    params.generation_prompt = input.metadata.value("generation_prompt", std::string());
    common_sampling_add_preserved_tokens(params, vocab, input.metadata.value("preserved_tokens", std::vector<std::string>{}));
    std::vector<common_grammar_trigger> triggers;
    for (const auto & trigger : input.metadata.value("grammar_triggers", json::array())) {
        common_grammar_trigger value;
        value.type = static_cast<common_grammar_trigger_type>(trigger.at("type").get<int>());
        value.value = trigger.value("value", std::string());
        value.token = trigger.value("token", LLAMA_TOKEN_NULL);
        triggers.push_back(std::move(value));
    }
    common_sampling_add_grammar_triggers(params, vocab, std::move(triggers));
    return params;
}

void configure_session(skippy_session & session, const fixture & input, uint32_t seed) {
    session.token_history = input.prompt;
    session.n_past = static_cast<int32_t>(input.prompt.size());
    auto config = skippy_config(seed);
    const auto meta = input.metadata.dump();
    skippy_error * error = nullptr;
    const auto status = skippy_session_configure_chat_sampling(&session, &config, meta.c_str(), input.prompt.size(), &error);
    const std::string message = error ? error->message : "sampling configuration failed";
    skippy_error_free(error);
    require(status == SKIPPY_STATUS_OK, message);
    require(session.grammar_sampler != nullptr && !session.sampling_backend_enabled, "grammar/backend configuration mismatch");
}

json fixture_manifest(const std::vector<fixture> & inputs) {
    json out = json::array();
    for (const auto & input : inputs) {
        out.push_back({{"name", input.name}, {"metadata", input.metadata}, {"prompt_tokens", input.prompt},
                       {"teacher_tokens", input.continuation}, {"force_rejection", input.force_rejection}, {"include_eos", input.include_eos},
                       {"partial_utf8_boundary", input.partial_utf8_boundary}});
    }
    return out;
}

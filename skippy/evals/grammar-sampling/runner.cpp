#include "fixtures.h"
#include "skippy/chat_sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sys/resource.h>

using clock_type = std::chrono::steady_clock;

struct samplers {
    native_runtime & runtime;
    skippy_session session;
    common_sampler * reference = nullptr;
    skippy_sampling_config config;
    bool upstream;

    samplers(native_runtime & rt, const fixture & input, uint32_t seed, bool use_upstream)
        : runtime(rt), config(skippy_config(seed)), upstream(use_upstream) {
        if (upstream) {
            auto params = reference_config(input, runtime.vocab, seed);
            reference = common_sampler_init(runtime.model, params);
            require(reference != nullptr, "reference sampler creation");
            for (auto token : input.prompt) {
                common_sampler_accept(reference, token, false);
            }
        } else {
            session.stage_model = &runtime.stage;
            session.ctx = runtime.ctx;
            configure_session(session, input, seed);
            // Both host adapters settle prompt history before measuring decode.
#ifdef SKIPPY_EVAL_COMMON_SAMPLER
            skippy_chat_accept_history(&session);
#else
            for (auto token : input.prompt) { llama_sampler_accept(session.sampling_chain, token); }
            session.sampling_accepted_token_count = input.prompt.size();
#endif
        }
    }

    ~samplers() {
        common_sampler_free(reference);
        skippy_clear_chat_sampling(&session);
    }

    llama_token sample(llama_token pending, bool have_pending) {
        if (upstream) {
            if (have_pending) {
                common_sampler_accept(reference, pending, true);
            }
            return common_sampler_sample(reference, runtime.ctx, -1);
        }
        if (have_pending) {
            session.token_history.push_back(pending);
        }
        llama_token token = LLAMA_TOKEN_NULL;
        const auto status = skippy_session_sample_current(&session, &config, &token, nullptr);
        if (status != SKIPPY_STATUS_OK) { return LLAMA_TOKEN_NULL; }
        return token;
    }
};

static void checked_read(std::istream & in, char * data, size_t bytes) {
    in.read(data, static_cast<std::streamsize>(bytes));
    require(static_cast<size_t>(in.gcount()) == bytes, "truncated logits file");
}

static std::vector<std::vector<float>> read_rows(const std::filesystem::path & directory,
                                                const fixture & input, size_t vocab) {
    std::ifstream in(directory / (input.name + ".logits"), std::ios::binary);
    require(in.good(), "missing captured logits");
    std::vector<std::vector<float>> result(input.continuation.size(), std::vector<float>(vocab));
    for (auto & row : result) {
        checked_read(in, reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
        require(std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }), "non-finite source logits");
    }
    require(in.peek() == EOF, "unexpected extra logits rows");
    return result;
}

static void capture(native_runtime & rt, const std::vector<fixture> & inputs, const std::filesystem::path & directory) {
    std::filesystem::create_directories(directory);
    std::ofstream manifest(directory / "fixtures.json");
    manifest << fixture_manifest(inputs).dump(2) << '\n';
    for (const auto & input : inputs) {
        rt.reset(input.prompt);
        std::ofstream rows(directory / (input.name + ".logits"), std::ios::binary);
        require(rows.good(), "cannot create logits file");
        llama_sampler * validator = nullptr;
        if (input.force_rejection) {
            const auto grammar = input.metadata.at("grammar").get<std::string>();
            validator = llama_sampler_init_grammar(rt.vocab, grammar.c_str(), "root");
            require(validator != nullptr, "rejection grammar creation");
        }
        for (size_t i = 0; i < input.continuation.size(); ++i) {
            auto * logits = llama_get_logits_ith(rt.ctx, -1);
            require(logits != nullptr, "missing logits");
            const auto teacher = input.continuation[i];
            if (validator) {
                llama_token invalid = LLAMA_TOKEN_NULL;
                for (llama_token token = 0; token < llama_vocab_n_tokens(rt.vocab); ++token) {
                    llama_token_data candidate{token, 1.0f, 0.0f};
                    llama_token_data_array single{&candidate, 1, -1, false};
                    llama_sampler_apply(validator, &single);
                    if (candidate.logit == -INFINITY) { invalid = token; break; }
                }
                require(invalid != LLAMA_TOKEN_NULL, "fixture needs a rejected proposal");
                // Deliberately adversarial input for the fallback case. Every
                // arm receives these same bytes; ordinary cases use model logits.
                std::fill(logits, logits + llama_vocab_n_tokens(rt.vocab), -20.0f);
                logits[teacher] = 0.0f;
                logits[invalid] = 100.0f;
                llama_token_data candidate{teacher, 1.0f, 0.0f};
                llama_token_data_array single{&candidate, 1, -1, false};
                llama_sampler_apply(validator, &single);
                require(candidate.logit != -INFINITY, "invalid fallback teacher token");
                llama_sampler_accept(validator, teacher);
            }
            rows.write(reinterpret_cast<const char *>(logits), llama_vocab_n_tokens(rt.vocab) * sizeof(float));
            rt.decode(teacher, static_cast<int>(input.prompt.size() + i));
        }
        llama_sampler_free(validator);
        std::cout << json({{"phase", "capture"}, {"case", input.name}, {"tokens", input.continuation.size()}}) << std::endl;
    }
}

static void verify_manifest(const std::filesystem::path & directory, const std::vector<fixture> & inputs) {
    std::ifstream in(directory / "fixtures.json");
    json saved;
    in >> saved;
    require(saved == fixture_manifest(inputs), "fixture/tokenizer/grammar identity mismatch");
}

static json measure(native_runtime & rt, const fixture & input, const std::filesystem::path & directory,
                    const std::string & arm, bool decode, int repetitions) {
    const auto rows = read_rows(directory, input, llama_vocab_n_tokens(rt.vocab));
    const auto count = input.continuation.size();
    std::vector<int64_t> times;
    std::vector<llama_token> decisions;
    int before = thermal_state();
    require(before <= 1, "thermal pressure before measurement");
    int64_t prefill_ns = 0;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        const auto prefill_start = clock_type::now();
        rt.reset(input.prompt);
        prefill_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - prefill_start).count();
        samplers sample(rt, input, 73, arm == "B");
        for (size_t i = 0; i < count; ++i) {
            // Logit copying and disk IO are outside both timed scopes. Decode
            // uses the same teacher-forced token sequence in every arm.
            std::memcpy(llama_get_logits_ith(rt.ctx, -1), rows[i].data(), rows[i].size() * sizeof(float));
            const auto start = clock_type::now();
            const auto selected = sample.sample(i ? input.continuation[i - 1] : 0, i != 0);
            if (decode && !llama_vocab_is_eog(rt.vocab, input.continuation[i])) {
                rt.decode(input.continuation[i], static_cast<int>(input.prompt.size() + i));
            }
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - start).count();
            if (!sample.upstream) {
#ifdef SKIPPY_EVAL_COMMON_SAMPLER
                require(sample.session.sampler != nullptr && sample.session.sampling_has_grammar, "grammar unexpectedly disabled");
#else
                require(sample.session.grammar_sampler != nullptr, "grammar unexpectedly disabled");
#endif
            }
            require(selected >= 0 && selected < llama_vocab_n_tokens(rt.vocab), "invalid sampled token");
            times.push_back(ns);
            decisions.push_back(selected);
        }
    }
    int after = thermal_state();
    require(after <= 1, "thermal pressure during measurement");
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "RSS unavailable");
    return {{"arm", arm}, {"case", input.name}, {"scope", decode ? "teacher-forced-native-decode" : "fixed-input-native-sampler"},
            {"times_ns", times}, {"selected_tokens", decisions}, {"teacher_tokens", count}, {"repetitions", repetitions},
            {"prefill_ns", prefill_ns}, {"thermal_before", before}, {"thermal_after", after},
            {"peak_rss_bytes", usage.ru_maxrss}};
}

static json correctness(native_runtime & rt, const fixture & input, uint32_t seed) {
    rt.reset(input.prompt);
    samplers reference(rt, input, seed, true);
    samplers candidate(rt, input, seed, false);
    std::vector<llama_token> generated;
    llama_token previous = 0;
    const size_t prefix = input.name == "lazy-before-tool-trigger" || input.force_rejection
        ? 0 : std::min<size_t>(64, input.continuation.size() > 8 ? input.continuation.size() - 8 : 0);
    for (size_t i = 0; i < prefix; ++i) {
        const auto b = reference.sample(previous, i != 0);
        const auto c = candidate.sample(previous, i != 0);
        require(b == c, "B/C prefix decision mismatch: " + input.name);
        previous = input.continuation[i];
        rt.decode(previous, static_cast<int>(input.prompt.size() + i));
    }
    for (int i = 0; i < 512; ++i) {
        auto b = reference.sample(previous, prefix != 0 || i != 0);
        auto c = candidate.sample(previous, prefix != 0 || i != 0);
        require(b == c, "B/C seeded decision mismatch: " + input.name + " step " + std::to_string(i));
        generated.push_back(b);
        if (llama_vocab_is_eog(rt.vocab, b)) { break; }
        rt.decode(b, static_cast<int>(input.prompt.size() + prefix) + i);
        previous = b;
    }
    return {{"case", input.name}, {"seed", seed}, {"forced_prefix_tokens", prefix}, {"B_C_identical", true}, {"tokens", generated},
            {"ended_at_eos", llama_vocab_is_eog(rt.vocab, generated.back())}};
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 4, "usage: grammar-sampling MODEL capture|correctness|replay|decode INPUT_DIR [A|B|C CASE REPETITIONS]");
        const std::string mode = argv[2];
        const std::filesystem::path directory = argv[3];
        native_runtime rt(argv[1]);
        const auto inputs = make_fixtures(rt.model);
        if (mode == "serve") {
            std::cout << json({{"ready", true}, {"fixtures", fixture_manifest(inputs)}}) << std::endl;
            std::string line;
            while (std::getline(std::cin, line)) {
                auto request = json::parse(line);
                const auto action = request.at("action").get<std::string>();
                if (action == "stop") { break; }
                if (action == "capture") {
                    // Capture progress belongs on stderr; one reply per request.
                    auto * saved = std::cout.rdbuf(std::cerr.rdbuf());
                    capture(rt, inputs, directory);
                    std::cout.rdbuf(saved);
                    std::cout << json({{"captured", true}}) << std::endl;
                } else if (action == "correctness") {
                    verify_manifest(directory, inputs);
                    json results = json::array();
                    for (const auto & input : inputs) {
                        for (uint32_t seed : {1u, 42u, 73u, 1234u}) {
                            results.push_back(correctness(rt, input, seed));
                        }
                    }
                    std::cout << json({{"correctness", results}}) << std::endl;
                } else {
                    require(action == "measure", "unknown worker action");
                    verify_manifest(directory, inputs);
                    const auto name = request.at("case").get<std::string>();
                    const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto & value) { return value.name == name; });
                    require(found != inputs.end(), "unknown worker case");
                    auto result = measure(rt, *found, directory, request.at("arm"), request.at("decode"), request.at("repetitions"));
                    result["block"] = request.at("block");
                    result["warmup"] = request.at("warmup");
                    std::cout << result << std::endl;
                }
            }
        } else if (mode == "capture") {
            capture(rt, inputs, directory);
        } else if (mode == "correctness") {
            verify_manifest(directory, inputs);
            for (const auto & input : inputs) {
                for (uint32_t seed : {1u, 42u, 73u, 1234u}) {
                    std::cout << correctness(rt, input, seed) << std::endl;
                }
            }
        } else {
            require(argc == 7, "measurement needs arm, case and repetitions");
            require(mode == "replay" || mode == "decode", "unknown mode");
            const std::string arm = argv[4];
            require(arm == "A" || arm == "B" || arm == "C", "unknown arm");
            verify_manifest(directory, inputs);
            const auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto & value) { return value.name == argv[5]; });
            require(found != inputs.end(), "unknown case");
            const int repetitions = std::stoi(argv[6]);
            require(repetitions >= 1 && repetitions <= 100, "repetition bound");
            std::cout << measure(rt, *found, directory, arm, mode == "decode", repetitions) << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "grammar-sampling: " << error.what() << '\n';
        return 1;
    }
}

#pragma once

#include "chat.h"
#include "sampling.h"
#include "skippy.h"
#include "skippy/runtime_types.h"
#include "skippy/sampling_internal.h"
#include "skippy/runtime_support.h"
#include "json.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct fixture {
    std::string name;
    json metadata;
    std::vector<llama_token> prompt;
    std::vector<llama_token> continuation;
    bool force_rejection = false;
    bool include_eos = false;
};

void require(bool condition, const std::string & detail);
std::vector<fixture> make_fixtures(llama_model * model);
common_params_sampling reference_config(const fixture & input, const llama_vocab * vocab, uint32_t seed);
skippy_sampling_config skippy_config(uint32_t seed);
void configure_session(skippy_session & session, const fixture & input, uint32_t seed);
json fixture_manifest(const std::vector<fixture> & inputs);
int thermal_state();

struct native_runtime {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    skippy_model stage;
    explicit native_runtime(const char * path);
    ~native_runtime();
    void reset(const std::vector<llama_token> & prompt);
    void decode(llama_token token, int position);
};

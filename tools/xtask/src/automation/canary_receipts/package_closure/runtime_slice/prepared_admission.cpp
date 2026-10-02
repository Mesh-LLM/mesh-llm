enum skippy_status skippy_finish_model_open(
        llama_model * model,
        const struct skippy_runtime_config * config,
        skippy_runtime_event_scope * event_scope,
        struct skippy_model ** out_model,
        struct skippy_error ** out_error) {
    if (skippy_runtime_has_stage_plan(config)) {
        const int32_t n_layer = skippy_stage_layer_count(model);
        if (config->layer_end > n_layer) {
            llama_model_free(model);
            const char * message = "layer_end exceeds model layer count";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_runtime_is_source_stage(config) != (config->layer_start == 0)) {
            llama_model_free(model);
            const char * message = "admitted activation imports disagree with the source-stage range";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
        if (skippy_runtime_is_terminal_stage(config) != (config->layer_end == n_layer)) {
            llama_model_free(model);
            const char * message = "admitted activation exports disagree with the terminal-stage range";
            if (event_scope != nullptr) {
                event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
            }
            skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
            return SKIPPY_STATUS_INVALID_ARGUMENT;
        }
    }

    const bool encoder_decoder = llama_model_has_encoder(model) && llama_model_has_decoder(model);
    const bool classifier_workload = model->cls != nullptr || model->cls_out != nullptr;
    const bool embedding_workload = !encoder_decoder &&
            (classifier_workload ||
             (llama_model_has_encoder(model) ||
              (model->hparams.pooling_type != LLAMA_POOLING_TYPE_NONE &&
               model->hparams.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED)));
    const uint32_t configured_lane_count = config != nullptr && config->lane_count > 0
        ? static_cast<uint32_t>(config->lane_count)
        : 1;
    // Encoder output is context-global in llama.cpp. Until that state can be
    // isolated per sequence, serialize encoder-decoder work on one lane.
    const uint32_t lane_count = encoder_decoder ? 1 : configured_lane_count;
    uint32_t context_size_per_lane = 0;
    uint32_t context_size_total = 0;
    if (!skippy_context_capacity(config, lane_count, context_size_per_lane, context_size_total)) {
        llama_model_free(model);
        const char * message = "ctx_size multiplied by lane_count exceeds the native context range";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        }
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    skippy_model * stage_model = new skippy_model{};
    stage_model->model = model;
    if (config != nullptr) {
        stage_model->config = *config;
        stage_model->executable = true;
        const auto copy_frontier = [](const char * const * values, size_t count, std::vector<std::string> & storage,
                                      std::vector<const char *> & pointers) {
            storage.reserve(count);
            for (size_t i = 0; i < count; ++i) storage.emplace_back(values[i]);
            pointers.reserve(storage.size());
            for (const std::string & value : storage) pointers.push_back(value.c_str());
        };
        copy_frontier(config->activation_import_identities, config->activation_import_identity_count,
                stage_model->activation_import_identities, stage_model->activation_import_identity_ptrs);
        copy_frontier(config->activation_import_bindings, config->activation_import_identity_count,
                stage_model->activation_import_bindings, stage_model->activation_import_binding_ptrs);
        copy_frontier(config->activation_export_identities, config->activation_export_identity_count,
                stage_model->activation_export_identities, stage_model->activation_export_identity_ptrs);
        copy_frontier(config->activation_export_bindings, config->activation_export_identity_count,
                stage_model->activation_export_bindings, stage_model->activation_export_binding_ptrs);
        copy_frontier(config->resident_tensor_names, config->resident_tensor_name_count,
                stage_model->resident_tensor_names, stage_model->resident_tensor_name_ptrs);
        stage_model->config.activation_import_identities = stage_model->activation_import_identity_ptrs.data();
        stage_model->config.activation_import_bindings = stage_model->activation_import_binding_ptrs.data();
        stage_model->config.activation_export_identities = stage_model->activation_export_identity_ptrs.data();
        stage_model->config.activation_export_bindings = stage_model->activation_export_binding_ptrs.data();
        stage_model->config.resident_tensor_names = stage_model->resident_tensor_name_ptrs.data();
        stage_model->execution_contract = config->execution_contract == nullptr ? "" : config->execution_contract;
        stage_model->config.execution_contract = stage_model->execution_contract.c_str();
    }
    stage_model->lane_count = lane_count;

    llama_context_params params = llama_context_default_params();
    // ctx_size is the per-lane request limit. llama.cpp's n_ctx is the total
    // cell pool, so every simultaneously admitted lane needs that capacity.
    params.n_ctx = context_size_total;
    params.n_batch = config != nullptr && config->n_batch > 0 ? static_cast<uint32_t>(config->n_batch) : context_size_per_lane;
    params.n_ubatch = config != nullptr && config->n_ubatch > 0 ? static_cast<uint32_t>(config->n_ubatch) : 0u;
    params.n_threads = config != nullptr && config->n_threads > 0 ? config->n_threads : skippy_default_thread_count();
    params.n_threads_batch = config != nullptr && config->n_threads_batch > 0 ? config->n_threads_batch : params.n_threads;
    params.n_seq_max = stage_model->lane_count;
    params.kv_unified = stage_model->lane_count > 1;
    if (config != nullptr) {
        params.kv_unified = skippy_resolve_tristate(config->kv_unified, params.kv_unified);
        params.offload_kqv = skippy_resolve_tristate(config->kv_offload, params.offload_kqv);
        params.swa_full = skippy_resolve_tristate(config->swa_full, params.swa_full);
        skippy_apply_context_hardware_config(config, params);
    }
    params.type_k = config != nullptr && config->cache_type_k > 0 ? static_cast<ggml_type>(config->cache_type_k) : GGML_TYPE_F16;
    params.type_v = config != nullptr && config->cache_type_v > 0 ? static_cast<ggml_type>(config->cache_type_v) : GGML_TYPE_F16;
    params.flash_attn_type = config != nullptr ? static_cast<llama_flash_attn_type>(config->flash_attn_type) : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (classifier_workload) {
        params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }
    const bool activation_export_stage =
            skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config);
    params.embeddings = embedding_workload || activation_export_stage;
    if (embedding_workload) {
        // Non-causal embedding graphs must fit their logical batch in one
        // physical batch, matching llama-embedding's runtime contract.
        params.n_ubatch = params.n_batch;
    }
    const bool glm_dsa_op_timing_enabled = skippy_glm_dsa_op_timing_enabled();
    const bool glm_dsa_tensor_trace_enabled = skippy_glm_dsa_tensor_trace_enabled();
    if (model->arch == LLM_ARCH_GLM_DSA && (glm_dsa_op_timing_enabled || glm_dsa_tensor_trace_enabled)) {
        stage_model->glm_dsa_timing.enabled = true;
        stage_model->glm_dsa_timing.print_timing = glm_dsa_op_timing_enabled;
        stage_model->glm_dsa_timing.trace_tensors = glm_dsa_tensor_trace_enabled;
        stage_model->glm_dsa_timing.trace_stats = skippy_glm_dsa_tensor_trace_stats_enabled();
        stage_model->glm_dsa_timing.trace_value_limit =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES", 8, 0, 4096);
        stage_model->glm_dsa_timing.trace_node_limit =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_NODES", 32, 1, 1024);
        stage_model->glm_dsa_timing.trace_stats_max_bytes =
                skippy_env_u32("SKIPPY_GLM_DSA_TENSOR_TRACE_STATS_MAX_BYTES", 32 * 1024 * 1024, 0, 1024 * 1024 * 1024);
        if (const char * trace_filter = std::getenv("SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER")) {
            stage_model->glm_dsa_timing.trace_filter = trace_filter;
        }
        stage_model->glm_dsa_timing.stage_index = config != nullptr ? config->stage_index : -1;
        params.cb_eval = skippy_glm_dsa_op_timing_cb;
        params.cb_eval_user_data = stage_model;
    }
    if (llm_arch_is_recurrent(model->arch) || llm_arch_is_hybrid(model->arch)) {
        params.n_seq_max = std::max<uint32_t>(params.n_seq_max, std::max<uint32_t>(2, stage_model->lane_count * 2));
        params.n_rs_seq = std::max<uint32_t>(params.n_rs_seq, 2);
        params.kv_unified = true;
    }
    // Batched activation exports are sliced by request offset, which is the
    // microbatch row order only while llama splits with split_simple. A
    // per-stream cache splits by sequence instead, so an exporting stage with
    // more than one lane cannot run without a unified cache. Reject the
    // configuration here, once, instead of failing every batched iteration.
    if (activation_export_stage && stage_model->lane_count > 1 && !params.kv_unified) {
        const char * message = "an exporting stage with multiple lanes requires a unified KV cache";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message);
        }
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message);
        return SKIPPY_STATUS_INVALID_ARGUMENT;
    }

    const skippy_graph_build_inputs graph_build_inputs = skippy_graph_build_inputs_from_config(config);
    stage_model->ctx = llama_init_from_model_with_graph_inputs(model, params, graph_build_inputs);
    if (stage_model->ctx == nullptr) {
        const char * message = "failed to create llama context";
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_RUNTIME_ERROR, message);
        }
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, message);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    }

    const auto fail_boundary_load = [&](const char * message) {
        if (event_scope != nullptr) {
            event_scope->emit_failure(SKIPPY_STATUS_RUNTIME_ERROR, message);
        }
        llama_free(stage_model->mtp_ctx);
        stage_model->mtp_ctx = nullptr;
        llama_free(stage_model->ctx);
        stage_model->ctx = nullptr;
        llama_model_free(model);
        delete stage_model;
        skippy_set_error(out_error, SKIPPY_STATUS_RUNTIME_ERROR, message);
        return SKIPPY_STATUS_RUNTIME_ERROR;
    };

    const auto parse_identity = [](const char * value, uint8_t output[SKIPPY_ACTIVATION_IDENTITY_BYTES]) {
        if (value == nullptr) return false;
        const size_t length = std::strlen(value);
        if (length < SKIPPY_ACTIVATION_IDENTITY_BYTES * 2) return false;
        const char * hex = value + length - SKIPPY_ACTIVATION_IDENTITY_BYTES * 2;
        const auto nibble = [](char character) -> int {
            if (character >= '0' && character <= '9') return character - '0';
            if (character >= 'a' && character <= 'f') return character - 'a' + 10;
            return -1;
        };
        for (size_t index = 0; index < SKIPPY_ACTIVATION_IDENTITY_BYTES; ++index) {
            const int high = nibble(hex[index * 2]);
            const int low = nibble(hex[index * 2 + 1]);
            if (high < 0 || low < 0) return false;
            output[index] = static_cast<uint8_t>((high << 4) | low);
        }
        return true;
    };
    const auto build_boundary = [&](bool input, skippy_activation_boundary_desc & boundary) {
        std::vector<skippy_activation_port_shape> shapes;
        const bool observed = input ? stage_model->ctx->get_input_activation_frontier(shapes)
                                    : stage_model->ctx->get_activation_frontier(shapes);
        const char * const * identities = input ? stage_model->config.activation_import_identities
                                                 : stage_model->config.activation_export_identities;
        const char * const * bindings = input ? stage_model->config.activation_import_bindings
                                               : stage_model->config.activation_export_bindings;
        const size_t identity_count = input ? stage_model->config.activation_import_identity_count
                                            : stage_model->config.activation_export_identity_count;
        if (!observed || shapes.empty() || shapes.size() != identity_count ||
                shapes.size() > SKIPPY_ACTIVATION_MAX_PARTS || bindings == nullptr) return false;
        boundary = {};
        boundary.version = SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION;
        boundary.part_count = static_cast<uint32_t>(shapes.size());
        std::set<size_t> matched_bindings;
        for (size_t index = 0; index < shapes.size(); ++index) {
            size_t binding_index = identity_count;
            for (size_t candidate = 0; candidate < identity_count; ++candidate) {
                if (bindings[candidate] != nullptr && shapes[index].binding == bindings[candidate]) {
                    binding_index = candidate;
                    break;
                }
            }
            if (binding_index == identity_count || !matched_bindings.insert(binding_index).second) return false;
            auto & part = boundary.parts[index];
            std::string stable_identity;
            if (!skippy_build_activation_port_identity(bindings[binding_index], stable_identity) ||
                    stable_identity != identities[binding_index] ||
                    !parse_identity(identities[binding_index], part.identity)) return false;
            part.ggml_type = static_cast<uint32_t>(shapes[index].type);
            part.rank = shapes[index].rank;
            part.token_axis = shapes[index].token_axis;
            part.flags = shapes[index].optional ? SKIPPY_ACTIVATION_PART_OPTIONAL : 0;
            uint64_t stride = ggml_type_size(shapes[index].type);
            for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                part.dimensions[dimension] = shapes[index].dimensions[dimension];
                part.byte_strides[dimension] = stride;
                if (shapes[index].dimensions[dimension] > 0) stride *= static_cast<uint64_t>(shapes[index].dimensions[dimension]);
            }
        }
        return skippy_build_activation_frontier_identity(
                boundary.parts, boundary.part_count, boundary.frontier_identity);
    };

    if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config)) {
        if (!build_boundary(false, stage_model->output_activation_boundary)) {
            return fail_boundary_load("stage graph output frontier does not match its admitted planner identities");
        }
        stage_model->has_output_activation_boundary = true;
    }

    if (skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_source_stage(config)) {
        if (!build_boundary(true, stage_model->input_activation_boundary)) {
            return fail_boundary_load("stage graph input frontier does not match its admitted planner identities");
        }
        stage_model->has_input_activation_boundary = true;
    }

    if (skippy_should_create_native_mtp_sidecar(model, config)) {
        llama_context_params mtp_params = params;
        mtp_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_params.embeddings = false;
        stage_model->mtp_ctx = llama_init_from_model_with_graph_inputs(model, mtp_params, graph_build_inputs);
        if (stage_model->mtp_ctx != nullptr) {
            llama_set_embeddings_nextn(stage_model->ctx, true, false);
            llama_set_embeddings_nextn(stage_model->mtp_ctx, true, false);
        } else {
            fprintf(stderr, "skippy: native MTP sidecar unavailable for this final stage; continuing without drafts\n");
        }
    } else if (config != nullptr &&
               config->mtp_source == SKIPPY_MTP_SOURCE_INTEGRATED &&
               skippy_runtime_is_terminal_stage(config) &&
               model->hparams.n_layer_nextn > 0) {
        fprintf(stderr, "skippy: integrated MTP requested, but this split stage is missing required MTP graph tensors; continuing without drafts\n");
    }

    if (skippy_runtime_has_stage_plan(config)) {
        std::string error;
        if (!stage_model->ctx->install_stage_program(graph_build_inputs, error)) {
            const std::string message = "failed to install reusable stage program: " + error;
            return fail_boundary_load(message.c_str());
        }
        if (stage_model->mtp_ctx != nullptr && !stage_model->mtp_ctx->install_stage_program(graph_build_inputs, error)) {
            const std::string message = "failed to install reusable MTP stage program: " + error;
            return fail_boundary_load(message.c_str());
        }
    }

    stage_model->lane_in_use.assign(stage_model->lane_count, false);
    stage_model->lane_resident_prefix_tokens.resize(stage_model->lane_count);

    const uint64_t native_model_id = reinterpret_cast<uint64_t>(stage_model);
    if (event_scope != nullptr) {
        event_scope->set_model_id(native_model_id);
        skippy_emit_observable_backend_device(event_scope, config, model);
    }

    // These are the concrete boundaries that are known after context
    // creation. They are sent to the operation reporter when one is active;
    // otherwise the process reporter receives the same observation. Do not
    // synthesize tensor/offload events here: llama.cpp does not expose a
    // stable count for those transitions at this boundary.
    skippy_emit_model_load_observation(
            event_scope,
            SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_PHASE_CHANGED,
            native_model_id,
            static_cast<uint64_t>(skippy_stage_layer_count(model)),
            0,
            "execution context ready",
            sizeof("execution context ready") - 1);
    skippy_emit_model_load_observation(
            event_scope,
            SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_TOKENIZER_READY,
            native_model_id,
            static_cast<uint64_t>(llama_vocab_n_tokens(&model->vocab)),
            0,
            "tokenizer ready",
            sizeof("tokenizer ready") - 1);
    if (stage_model->mtp_ctx != nullptr) {
        skippy_emit_model_load_observation(
                event_scope,
                SKIPPY_RUNTIME_EVENT_KIND_MODEL_LOAD_AUX_COMPONENT_READY,
                native_model_id,
                1,
                0,
                "mtp component ready",
                sizeof("mtp component ready") - 1);
    }
    if (event_scope != nullptr) {
        event_scope->emit_finished();
    }

    *out_model = stage_model;
    return skippy_success(out_error);
}

enum skippy_status skippy_model_open_impl(void) {}

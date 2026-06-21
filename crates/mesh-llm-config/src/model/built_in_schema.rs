use super::*;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltInConfigPathResolution {
    pub requested_path: ConfigPath,
    pub normalized_path: ConfigPath,
    pub canonical_path: ConfigPath,
    pub matched_alias: Option<ConfigPath>,
    pub support: ConfigSupportState,
}

impl BuiltInConfigPathResolution {
    pub fn canonical_identifier(&self) -> String {
        self.canonical_path.render()
    }

    pub fn used_legacy_alias(&self) -> bool {
        self.matched_alias.is_some()
    }
}

pub fn built_in_config_settings() -> Vec<ConfigSettingSchema> {
    built_in_config_schema_cache().settings.clone()
}

pub fn built_in_config_schema_descriptor(path: &ConfigPath) -> Option<ConfigSettingSchema> {
    let normalized = path.normalize_builtin_layout();
    built_in_config_schema_cache()
        .settings
        .iter()
        .find(|setting| setting.path == normalized)
        .cloned()
}

pub fn resolve_built_in_config_path(path: &ConfigPath) -> Option<BuiltInConfigPathResolution> {
    let requested_path = path.clone();
    let normalized_path = path.normalize_builtin_layout();

    for setting in &built_in_config_schema_cache().settings {
        if setting.path == normalized_path {
            return Some(BuiltInConfigPathResolution {
                requested_path,
                normalized_path,
                canonical_path: setting.path.clone(),
                matched_alias: None,
                support: setting.support,
            });
        }
        if let Some(alias) = setting
            .alias_policy
            .aliases
            .iter()
            .find(|alias| alias.path == normalized_path)
        {
            return Some(BuiltInConfigPathResolution {
                requested_path,
                normalized_path,
                canonical_path: setting.path.clone(),
                matched_alias: Some(alias.path.clone()),
                support: setting.support,
            });
        }
    }

    None
}

pub fn resolve_built_in_config_identifier(rendered: &str) -> Option<BuiltInConfigPathResolution> {
    let parsed = ConfigPath::parse_rendered(rendered).ok()?;
    resolve_built_in_config_path(&parsed)
}

pub fn canonicalize_built_in_config_path(path: &ConfigPath) -> Option<ConfigPath> {
    resolve_built_in_config_path(path).map(|resolution| resolution.canonical_path)
}

pub fn canonicalize_built_in_config_identifier(rendered: &str) -> Option<String> {
    resolve_built_in_config_identifier(rendered).map(|resolution| resolution.canonical_identifier())
}

fn built_in_config_schema_cache() -> &'static ConfigSchema {
    static SCHEMA: OnceLock<ConfigSchema> = OnceLock::new();
    SCHEMA.get_or_init(build_built_in_config_schema)
}

fn build_built_in_config_schema() -> ConfigSchema {
    let mut settings = vec![
        top_level_setting("version", ConfigValueSchema::Integer),
        top_level_setting("gpu.assignment", string_enum(["auto", "pinned"])),
        top_level_setting("gpu.parallel", ConfigValueSchema::Integer),
        top_level_setting(
            "mesh_requirements.min_node_version",
            ConfigValueSchema::String,
        ),
        top_level_setting(
            "mesh_requirements.max_node_version",
            ConfigValueSchema::String,
        ),
        top_level_setting(
            "mesh_requirements.min_protocol_version",
            ConfigValueSchema::Integer,
        ),
        top_level_setting(
            "mesh_requirements.max_protocol_version",
            ConfigValueSchema::Integer,
        ),
        top_level_setting(
            "mesh_requirements.require_release_attestation",
            ConfigValueSchema::Boolean,
        ),
        top_level_setting(
            "mesh_requirements.release_signer_keys",
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        owner_control_setting("owner_control.bind", ConfigValueSchema::SocketAddr),
        owner_control_setting(
            "owner_control.advertise_addr",
            ConfigValueSchema::SocketAddr,
        ),
        telemetry_setting("telemetry.enabled", ConfigValueSchema::Boolean),
        telemetry_setting("telemetry.service_name", ConfigValueSchema::String),
        telemetry_setting("telemetry.endpoint", ConfigValueSchema::String),
        telemetry_setting("telemetry.headers", ConfigValueSchema::Object),
        telemetry_setting("telemetry.export_interval_secs", ConfigValueSchema::Integer),
        telemetry_setting("telemetry.queue_size", ConfigValueSchema::Integer),
        unsupported_setting(
            "telemetry.prompt_shape_metrics",
            ConfigValueSchema::Boolean,
            "Prompt-shape telemetry is intentionally disabled until the telemetry surface is reviewed.",
        ),
        telemetry_setting("telemetry.metrics.endpoint", ConfigValueSchema::String),
        runtime_setting(
            "runtime.reconcile_model_targets",
            ConfigValueSchema::Boolean,
        ),
        runtime_setting(
            "runtime.reconcile_model_target_demand_upgrades",
            ConfigValueSchema::Boolean,
        ),
        native_runtime_setting(
            "runtime.native_runtime.mesh_version",
            ConfigValueSchema::String,
        ),
        native_runtime_setting(
            "runtime.native_runtime.skippy_abi",
            ConfigValueSchema::String,
        ),
        native_runtime_setting(
            "runtime.native_runtime.selection",
            ConfigValueSchema::String,
        ),
        runtime_setting(
            "runtime.model_target_demand_upgrade_min_requests",
            ConfigValueSchema::Integer,
        ),
        runtime_setting(
            "runtime.model_target_demand_upgrade_max_age_secs",
            ConfigValueSchema::Integer,
        ),
    ];

    settings.extend(model_defaults_settings());
    settings.extend(model_entry_settings());
    settings.extend(plugin_entry_settings());

    ConfigSchema { settings }
}

fn model_defaults_settings() -> Vec<ConfigSettingSchema> {
    let mut settings = Vec::new();
    settings.extend(model_fit_settings(
        "defaults.model_fit",
        &[
            flat_alias("defaults.ctx_size"),
            flat_alias("defaults.batch"),
            flat_alias("defaults.ubatch"),
            flat_alias("defaults.cache_type_k"),
            flat_alias("defaults.cache_type_v"),
            flat_alias("defaults.flash_attention"),
        ],
    ));
    settings.extend(hardware_settings(
        "defaults.hardware",
        &[flat_alias("defaults.gpu_id")],
    ));
    settings.extend(throughput_settings(
        "defaults.throughput",
        &[flat_alias("defaults.parallel")],
    ));
    settings.extend(skippy_settings("defaults.skippy"));
    settings.extend(speculative_settings("defaults.speculative"));
    settings.extend(request_defaults_settings("defaults.request_defaults"));
    settings.extend(multimodal_settings(
        "defaults.multimodal",
        &[flat_alias("defaults.mmproj")],
    ));
    settings.extend(advanced_settings("defaults.advanced"));
    settings
}

fn model_entry_settings() -> Vec<ConfigSettingSchema> {
    let model_prefix = format!("models.{CANONICAL_MODEL_REF_SEGMENT}");
    let mut settings = vec![basic_setting(
        &format!("{model_prefix}.model"),
        ConfigValueSchema::String,
    )];
    settings.extend(model_fit_settings(
        &format!("{model_prefix}.model_fit"),
        &[
            flat_alias(&format!("{model_prefix}.ctx_size")),
            flat_alias(&format!("{model_prefix}.batch")),
            flat_alias(&format!("{model_prefix}.ubatch")),
            flat_alias(&format!("{model_prefix}.cache_type_k")),
            flat_alias(&format!("{model_prefix}.cache_type_v")),
            flat_alias(&format!("{model_prefix}.flash_attention")),
        ],
    ));
    settings.extend(hardware_settings(
        &format!("{model_prefix}.hardware"),
        &[flat_alias(&format!("{model_prefix}.gpu_id"))],
    ));
    settings.extend(throughput_settings(
        &format!("{model_prefix}.throughput"),
        &[flat_alias(&format!("{model_prefix}.parallel"))],
    ));
    settings.extend(skippy_settings(&format!("{model_prefix}.skippy")));
    settings.extend(speculative_settings(&format!("{model_prefix}.speculative")));
    settings.extend(request_defaults_settings(&format!(
        "{model_prefix}.request_defaults"
    )));
    settings.extend(multimodal_settings(
        &format!("{model_prefix}.multimodal"),
        &[flat_alias(&format!("{model_prefix}.mmproj"))],
    ));
    settings.extend(advanced_settings(&format!("{model_prefix}.advanced")));
    settings
}

fn plugin_entry_settings() -> Vec<ConfigSettingSchema> {
    let plugin_prefix = format!("plugin.{CANONICAL_PLUGIN_NAME_SEGMENT}");
    vec![
        plugin_setting(&format!("{plugin_prefix}.name"), ConfigValueSchema::String),
        plugin_setting(
            &format!("{plugin_prefix}.enabled"),
            ConfigValueSchema::Boolean,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.command"),
            ConfigValueSchema::String,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.args"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        plugin_setting(&format!("{plugin_prefix}.url"), ConfigValueSchema::String),
        plugin_setting(
            &format!("{plugin_prefix}.startup.connect_timeout_secs"),
            ConfigValueSchema::Integer,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.init_timeout_secs"),
            ConfigValueSchema::Integer,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.optional"),
            ConfigValueSchema::Boolean,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.lazy_start"),
            ConfigValueSchema::Boolean,
        ),
    ]
}

fn model_fit_settings(
    prefix: &str,
    legacy_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.ctx_size"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.batch"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.ubatch"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.cache_type_k"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.cache_type_v"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.kv_cache_policy"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.kv_offload"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.kv_unified"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.cache_ram_mib"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.cache_idle_slots"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.prompt_cache"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.prefix_cache.enabled"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.max_entries"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.max_bytes"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.shared_stride_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.shared_record_limit"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.payload_mode"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.keep_tokens"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.context_shift"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.swa_full"), ConfigValueSchema::Boolean),
        basic_setting(
            &format!("{prefix}.checkpoint_interval"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.checkpoint_count"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.lookup_cache_static"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.lookup_cache_dynamic"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.flash_attention"),
            ConfigValueSchema::String,
        ),
    ];

    if !legacy_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.ctx_size"),
            &legacy_aliases[0..1],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.batch"),
            &legacy_aliases[1..2],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.ubatch"),
            &legacy_aliases[2..3],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.cache_type_k"),
            &legacy_aliases[3..4],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.cache_type_v"),
            &legacy_aliases[4..5],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.flash_attention"),
            &legacy_aliases[5..6],
        );
    }

    settings
}

fn hardware_settings(
    prefix: &str,
    legacy_device_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(
            &format!("{prefix}.model_runtime"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.device"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.gpu_layers"), integer_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.stage_layer_start"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.stage_layer_end"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.placement"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.tensor_split"), tensor_split_schema()),
        basic_setting(&format!("{prefix}.split_mode"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.main_gpu"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.cpu_moe"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.n_cpu_moe"), ConfigValueSchema::Integer),
        rejected_setting(
            &format!("{prefix}.rpc_backend"),
            ConfigValueSchema::Object,
            "The legacy rpc_backend escape hatch is explicitly unsupported by the embedded runtime.",
        ),
        basic_setting(
            &format!("{prefix}.fit_target_mib"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.safety_margin_gb"),
            ConfigValueSchema::Float,
        ),
        basic_setting(&format!("{prefix}.fit_context"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.model_path"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.hf_repo"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.hf_file"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj_offload"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.lora_adapters"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        basic_setting(
            &format!("{prefix}.control_vectors"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        basic_setting(
            &format!("{prefix}.check_tensors"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(&format!("{prefix}.mmap"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.mlock"), ConfigValueSchema::Boolean),
        basic_setting(&format!("{prefix}.direct_io"), ConfigValueSchema::Boolean),
        basic_setting(&format!("{prefix}.repack"), ConfigValueSchema::Boolean),
        basic_setting(&format!("{prefix}.op_offload"), ConfigValueSchema::Boolean),
        basic_setting(
            &format!("{prefix}.no_host_buffer"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(&format!("{prefix}.warmup"), bool_or_auto_schema()),
    ];

    if !legacy_device_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.device"),
            legacy_device_aliases,
        );
    }

    settings
}

fn throughput_settings(
    prefix: &str,
    legacy_parallel_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.parallel"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.continuous_batching"),
            bool_or_auto_schema(),
        ),
        basic_setting(&format!("{prefix}.threads"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.threads_batch"),
            ConfigValueSchema::Integer,
        ),
        rejected_setting(
            &format!("{prefix}.threads_http"),
            ConfigValueSchema::Integer,
            "Dedicated HTTP worker tuning is rejected on the current embedded runtime path.",
        ),
        basic_setting(&format!("{prefix}.priority"), integer_or_string_schema()),
        basic_setting(
            &format!("{prefix}.poll"),
            bool_or_string_enum(["auto", "busy", "sleep"]),
        ),
        basic_setting(&format!("{prefix}.cpu_affinity"), string_or_list_schema()),
        basic_setting(&format!("{prefix}.numa"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.slot_prompt_similarity"),
            ConfigValueSchema::Float,
        ),
        rejected_setting(
            &format!("{prefix}.sleep_idle_seconds"),
            ConfigValueSchema::Integer,
            "The sleep-idle tuning knob is documented as rejected and must never become a live exported identifier.",
        ),
        basic_setting(
            &format!("{prefix}.tuning_profile"),
            ConfigValueSchema::String,
        ),
    ];

    if !legacy_parallel_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.parallel"),
            legacy_parallel_aliases,
        );
    }

    settings
}

fn skippy_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        basic_setting(
            &format!("{prefix}.stage_model_path"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.stage_role"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.stage_topology"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.activation_wire_dtype"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.binary_stage_transport"),
            ConfigValueSchema::String,
        ),
        rejected_setting(
            &format!("{prefix}.openai_frontend_mode"),
            ConfigValueSchema::Object,
            "OpenAI frontend override wiring is intentionally rejected on the built-in schema surface.",
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_startup_timeout_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_readiness_interval_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_health_interval_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunking"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunk_size"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunk_schedule"),
            ConfigValueSchema::String,
        ),
    ]
}

fn speculative_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        basic_setting(&format!("{prefix}.strategy"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mode"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.draft_model_path"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_hf_repo"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_hf_file"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_selection_policy"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.pairing_fault"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_acceptance_threshold"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.draft_split_probability"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.draft_gpu_layers"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.draft_device"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.draft_threads"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_cache_type_k"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_cache_type_v"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.ngram_min"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.ngram_max"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.spec_default"), bool_or_auto_schema()),
    ]
}

fn request_defaults_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        basic_setting(&format!("{prefix}.max_tokens"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.stop"), string_or_list_schema()),
        basic_setting(&format!("{prefix}.temperature"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_k"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.min_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.typical_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_nsigma"), ConfigValueSchema::Float),
        basic_setting(
            &format!("{prefix}.dynatemp_range"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.dynatemp_exponent"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.repeat_penalty"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.repeat_last_n"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.presence_penalty"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.frequency_penalty"),
            ConfigValueSchema::Float,
        ),
        unwired_setting(
            &format!("{prefix}.dry"),
            ConfigValueSchema::Object,
            "Reserved sampler object accepted for compatibility but not wired into the current runtime.",
        ),
        unwired_setting(
            &format!("{prefix}.xtc"),
            ConfigValueSchema::Object,
            "Reserved sampler object accepted for compatibility but not wired into the current runtime.",
        ),
        unwired_setting(
            &format!("{prefix}.adaptive"),
            ConfigValueSchema::Object,
            "Reserved sampler object accepted for compatibility but not wired into the current runtime.",
        ),
        basic_setting(
            &format!("{prefix}.mirostat_mode"),
            integer_or_string_enum(["disabled", "1", "2"]),
        ),
        basic_setting(
            &format!("{prefix}.mirostat_entropy"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.mirostat_learning_rate"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.samplers"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        basic_setting(
            &format!("{prefix}.sampler_sequence"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.seed"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.logit_bias"), ConfigValueSchema::Object),
        basic_setting(&format!("{prefix}.ignore_eos"), ConfigValueSchema::Boolean),
        rejected_setting(
            &format!("{prefix}.backend_sampling"),
            ConfigValueSchema::Object,
            "Backend-owned sampler blocks are explicitly rejected from the built-in control surface.",
        ),
        basic_setting(
            &format!("{prefix}.reasoning_format"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.reasoning_enabled"),
            bool_or_string_enum(["auto", "off", "on"]),
        ),
        basic_setting(
            &format!("{prefix}.reasoning_budget"),
            integer_or_string_enum(["auto", "low", "medium", "high"]),
        ),
        basic_setting(
            &format!("{prefix}.chat_template"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.chat_template_file"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.jinja"), ConfigValueSchema::Boolean),
        basic_setting(
            &format!("{prefix}.chat_template_kwargs"),
            ConfigValueSchema::Object,
        ),
        basic_setting(
            &format!("{prefix}.skip_chat_parsing"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.prefill_assistant"),
            ConfigValueSchema::Object,
        ),
        basic_setting(
            &format!("{prefix}.system_prompt"),
            ConfigValueSchema::String,
        ),
        rejected_setting(
            &format!("{prefix}.grammar"),
            ConfigValueSchema::Object,
            "Grammar injection is explicitly rejected on the built-in config surface.",
        ),
        rejected_setting(
            &format!("{prefix}.json_schema"),
            ConfigValueSchema::Object,
            "JSON schema response shaping is intentionally rejected until a stable runtime contract exists.",
        ),
        rejected_setting(
            &format!("{prefix}.logprobs"),
            ConfigValueSchema::Object,
            "Logprobs request defaults are explicitly rejected from persisted config.",
        ),
    ]
}

fn multimodal_settings(
    prefix: &str,
    legacy_mmproj_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj_url"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj_offload"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.image_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.image_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        rejected_setting(
            &format!("{prefix}.embeddings"),
            ConfigValueSchema::Object,
            "Built-in multimodal embeddings controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.reranking"),
            ConfigValueSchema::Object,
            "Built-in reranking controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.pooling"),
            ConfigValueSchema::Object,
            "Built-in pooling controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.vocoder"),
            ConfigValueSchema::Object,
            "Built-in vocoder controls are explicitly rejected from persisted config.",
        ),
    ];

    if !legacy_mmproj_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.mmproj"),
            legacy_mmproj_aliases,
        );
    }

    settings
}

fn advanced_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        rejected_setting(
            &format!("{prefix}.server.host"),
            ConfigValueSchema::String,
            "Server host overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.port"),
            ConfigValueSchema::Integer,
            "Server port overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.reuse_port"),
            ConfigValueSchema::Boolean,
            "reuse_port overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.timeout"),
            ConfigValueSchema::Integer,
            "Server timeout overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.metrics"),
            ConfigValueSchema::Boolean,
            "Server metrics overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.slots"),
            ConfigValueSchema::Boolean,
            "Server slot overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.props"),
            ConfigValueSchema::Boolean,
            "Server props overrides are explicitly rejected from persisted model config.",
        ),
        basic_setting(&format!("{prefix}.server.alias"), ConfigValueSchema::String),
        rejected_setting(
            &format!("{prefix}.server.api_prefix"),
            ConfigValueSchema::String,
            "API prefix overrides are explicitly rejected from persisted model config.",
        ),
    ]
}

fn top_level_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.visibility = if path == "version" {
        ConfigVisibility::Internal
    } else {
        ConfigVisibility::Advanced
    };
    setting
}

fn owner_control_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::OwnerControl,
    ];
    setting.apply_mode = ConfigApplyMode::DynamicApply;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn telemetry_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting
}

fn runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.apply_mode = ConfigApplyMode::DynamicValidationOnly;
    setting
}

fn native_runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.apply_mode = ConfigApplyMode::DynamicValidationOnly;
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting.description = Some(
        "Native runtime selection is read before dynamic runtime libraries are loaded.".into(),
    );
    setting
}

fn plugin_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![
        ConfigControlSurface::ConfigFile,
        ConfigControlSurface::PluginManifest,
    ];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
    setting
}

fn basic_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    ConfigSettingSchema {
        path: schema_path(path),
        alias_policy: ConfigAliasPolicy::default(),
        owner: ConfigSettingOwner::BuiltIn,
        value_schema,
        support: ConfigSupportState::Supported,
        control_surfaces: vec![ConfigControlSurface::ConfigFile],
        apply_mode: ConfigApplyMode::StaticOnLoad,
        restart_scope: ConfigRestartScope::ModelReload,
        visibility: ConfigVisibility::Advanced,
        constraints: Vec::new(),
        description: Some(path.to_string()),
    }
}

fn unsupported_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Unsupported;
    setting.restart_scope = ConfigRestartScope::None;
    setting.description = Some(description.to_string());
    setting
}

fn rejected_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Rejected;
    setting.restart_scope = ConfigRestartScope::None;
    setting.description = Some(description.to_string());
    setting
}

fn unwired_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.support = ConfigSupportState::Unwired;
    setting.description = Some(description.to_string());
    setting
}

fn schema_path(path: &str) -> ConfigPath {
    ConfigPath::parse_rendered(path).expect("static schema path should parse")
}

fn flat_alias(path: &str) -> ConfigPathAlias {
    ConfigPathAlias {
        path: schema_path(path),
        kind: ConfigPathAliasKind::LegacyLayout,
        note: Some("legacy flattened TOML field".into()),
    }
}

fn string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    ConfigValueSchema::Enum {
        values: values.into_iter().map(str::to_string).collect(),
    }
}

fn one_of<const N: usize>(variants: [ConfigValueSchema; N]) -> ConfigValueSchema {
    ConfigValueSchema::OneOf {
        variants: variants.into_iter().collect(),
    }
}

fn bool_or_auto_schema() -> ConfigValueSchema {
    bool_or_string_enum(["auto", "true", "false"])
}

fn bool_or_string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    one_of([ConfigValueSchema::Boolean, string_enum(values)])
}

fn integer_or_auto_schema() -> ConfigValueSchema {
    integer_or_string_enum(["auto"])
}

fn integer_or_string_schema() -> ConfigValueSchema {
    one_of([ConfigValueSchema::Integer, ConfigValueSchema::String])
}

fn integer_or_string_enum<const N: usize>(values: [&str; N]) -> ConfigValueSchema {
    one_of([ConfigValueSchema::Integer, string_enum(values)])
}

fn string_or_list_schema() -> ConfigValueSchema {
    one_of([
        ConfigValueSchema::String,
        ConfigValueSchema::Array {
            items: Box::new(ConfigValueSchema::String),
        },
    ])
}

fn tensor_split_schema() -> ConfigValueSchema {
    one_of([
        ConfigValueSchema::Array {
            items: Box::new(ConfigValueSchema::Float),
        },
        ConfigValueSchema::String,
    ])
}

fn apply_aliases(
    settings: &mut [ConfigSettingSchema],
    canonical_path: &str,
    aliases: &[ConfigPathAlias],
) {
    if let Some(setting) = settings
        .iter_mut()
        .find(|setting| setting.path.render() == canonical_path)
    {
        setting.alias_policy.mode = ConfigAliasMode::CanonicalWithLegacyAliases;
        setting.alias_policy.aliases.extend_from_slice(aliases);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_schema_preserves_union_typed_fields() {
        for path in [
            "models.<model-ref>.model_fit.kv_offload",
            "models.<model-ref>.model_fit.kv_unified",
            "models.<model-ref>.model_fit.prompt_cache",
            "models.<model-ref>.model_fit.context_shift",
            "models.<model-ref>.hardware.cpu_moe",
            "models.<model-ref>.hardware.fit_context",
            "models.<model-ref>.hardware.mmproj_offload",
            "models.<model-ref>.hardware.mmap",
            "models.<model-ref>.hardware.warmup",
            "models.<model-ref>.throughput.continuous_batching",
            "models.<model-ref>.speculative.spec_default",
            "models.<model-ref>.multimodal.mmproj_offload",
        ] {
            assert_eq!(schema_value(path), bool_or_auto_schema());
        }

        assert_eq!(
            schema_value("models.<model-ref>.hardware.gpu_layers"),
            integer_or_auto_schema()
        );
        assert_eq!(
            schema_value("models.<model-ref>.hardware.tensor_split"),
            tensor_split_schema()
        );
        assert_eq!(
            schema_value("models.<model-ref>.throughput.priority"),
            integer_or_string_schema()
        );
        assert_eq!(
            schema_value("models.<model-ref>.throughput.poll"),
            bool_or_string_enum(["auto", "busy", "sleep"])
        );
        assert_eq!(
            schema_value("models.<model-ref>.throughput.cpu_affinity"),
            string_or_list_schema()
        );
        assert_eq!(
            schema_value("models.<model-ref>.request_defaults.stop"),
            string_or_list_schema()
        );
        assert_eq!(
            schema_value("models.<model-ref>.request_defaults.mirostat_mode"),
            integer_or_string_enum(["disabled", "1", "2"])
        );
        assert_eq!(
            schema_value("models.<model-ref>.request_defaults.reasoning_enabled"),
            bool_or_string_enum(["auto", "off", "on"])
        );
        assert_eq!(
            schema_value("models.<model-ref>.request_defaults.reasoning_budget"),
            integer_or_string_enum(["auto", "low", "medium", "high"])
        );
    }

    fn schema_value(path: &str) -> ConfigValueSchema {
        built_in_config_schema_descriptor(&schema_path(path))
            .expect("schema setting should exist")
            .value_schema
    }
}

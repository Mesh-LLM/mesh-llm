use super::*;
mod control_behavior;
mod presentation;
use self::control_behavior::apply_built_in_control_behavior;
use self::presentation::apply_built_in_presentation_metadata;
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
            string_enum_from_slice(known_mesh_llm_versions()),
        ),
        top_level_setting(
            "mesh_requirements.max_node_version",
            string_enum_from_slice(known_mesh_llm_versions()),
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
        telemetry_setting("telemetry.endpoint", ConfigValueSchema::Url),
        telemetry_setting("telemetry.headers", ConfigValueSchema::Object),
        telemetry_setting("telemetry.export_interval_secs", ConfigValueSchema::Integer),
        telemetry_setting("telemetry.queue_size", ConfigValueSchema::Integer),
        unsupported_setting(
            "telemetry.prompt_shape_metrics",
            ConfigValueSchema::Boolean,
            "Prompt-shape telemetry is intentionally disabled until the telemetry surface is reviewed.",
        ),
        telemetry_setting("telemetry.metrics.endpoint", ConfigValueSchema::Url),
        startup_runtime_setting("runtime.debug", ConfigValueSchema::Boolean),
        startup_runtime_setting("runtime.listen_all", ConfigValueSchema::Boolean),
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
    settings
        .iter_mut()
        .for_each(apply_built_in_control_behavior);
    settings
        .iter_mut()
        .for_each(apply_built_in_presentation_metadata);

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
            &format!("{plugin_prefix}.web_ui_enabled"),
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
        plugin_setting(&format!("{plugin_prefix}.url"), ConfigValueSchema::Url),
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
        basic_setting(&format!("{prefix}.cache_type_k"), kv_cache_type_schema()),
        basic_setting(&format!("{prefix}.cache_type_v"), kv_cache_type_schema()),
        basic_setting(
            &format!("{prefix}.kv_cache_policy"),
            string_enum(["auto", "quality", "balanced", "saver"]),
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
            string_enum(["resident-kv", "kv-recurrent", "full-state", "auto"]),
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
            string_enum(["auto", "disabled", "enabled"]),
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
        hidden_setting(
            &format!("{prefix}.model_runtime"),
            string_enum(["auto", "cpu", "cuda", "rocm", "metal", "vulkan"]),
            "Model runtime is selected by the installed native runtime and hardware resolver, not by the web configuration UI.",
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
        basic_setting(
            &format!("{prefix}.placement"),
            string_enum(["auto", "pooled", "separated"]),
        ),
        basic_setting(&format!("{prefix}.tensor_split"), tensor_split_schema()),
        basic_setting(
            &format!("{prefix}.split_mode"),
            string_enum(["auto", "none", "layer", "row"]),
        ),
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
        basic_setting(&format!("{prefix}.model_path"), ConfigValueSchema::Path),
        basic_setting(&format!("{prefix}.hf_repo"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.hf_file"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::Path),
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
            string_enum(["throughput", "balanced", "saver"]),
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
            ConfigValueSchema::Path,
        ),
        basic_setting(&format!("{prefix}.stage_role"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.stage_topology"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.activation_wire_dtype"),
            string_enum(["auto", "f16", "f32", "q8"]),
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
            string_enum(["auto", "fixed", "schedule", "adaptive-ramp"]),
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
        basic_setting(
            &format!("{prefix}.mode"),
            string_enum(["auto", "disabled", "draft"]),
        ),
        basic_setting(&format!("{prefix}.draft_model"), ConfigValueSchema::Path),
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
            string_enum(["manual", "auto"]),
        ),
        basic_setting(
            &format!("{prefix}.pairing_fault"),
            string_enum([
                "warn_disable",
                "fail-open",
                "fail-closed",
                "fail_open",
                "fail_closed",
            ]),
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
            kv_cache_type_schema(),
        ),
        basic_setting(
            &format!("{prefix}.draft_cache_type_v"),
            kv_cache_type_schema(),
        ),
        basic_setting(&format!("{prefix}.ngram_min"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.ngram_max"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.ngram_proposer"),
            string_enum(["cache", "suffix"]),
        ),
        basic_setting(
            &format!("{prefix}.ngram_max_proposal_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.extension_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_reject_cooldown_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_suppress_cooldown_drafts"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_suppress_cooldown_draft_limit"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_pipeline_depth"),
            ConfigValueSchema::Integer,
        ),
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
            string_enum(["auto", "none", "deepseek", "deepseek-legacy", "hidden"]),
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
            ConfigValueSchema::Path,
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
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::Path),
        basic_setting(&format!("{prefix}.mmproj_url"), ConfigValueSchema::Url),
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

fn startup_runtime_setting(path: &str, value_schema: ConfigValueSchema) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.control_surfaces = vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api];
    setting.restart_scope = ConfigRestartScope::ProcessRestart;
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
        presentation: None,
        control_behavior: None,
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

fn hidden_setting(
    path: &str,
    value_schema: ConfigValueSchema,
    description: &str,
) -> ConfigSettingSchema {
    let mut setting = basic_setting(path, value_schema);
    setting.visibility = ConfigVisibility::Hidden;
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

fn string_enum_from_slice(values: &[&str]) -> ConfigValueSchema {
    ConfigValueSchema::Enum {
        values: values.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn kv_cache_type_schema() -> ConfigValueSchema {
    string_enum([
        "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
    ])
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

/// Returns the list of known mesh-llm versions from GitHub releases.
/// This list should be updated during the release process.
fn known_mesh_llm_versions() -> &'static [&'static str] {
    &[
        "0.72.1", "0.72.0", "0.71.0", "0.70.0", "0.69.0", "0.68.0", "0.67.0", "0.66.0", "0.65.0",
        "0.64.0", "0.63.0", "0.62.0", "0.61.0", "0.60.0",
    ]
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

    #[test]
    fn built_in_schema_marks_curated_defaults_user_visible() {
        for path in [
            "defaults.throughput.threads",
            "defaults.throughput.parallel",
            "defaults.model_fit.kv_cache_policy",
            "defaults.request_defaults.temperature",
            "defaults.skippy.binary_stage_transport",
            "defaults.multimodal.mmproj_offload",
        ] {
            assert_eq!(
                schema_setting(path).visibility,
                ConfigVisibility::User,
                "{path}"
            );
        }

        assert_eq!(
            schema_setting("defaults.model_fit.prompt_cache").visibility,
            ConfigVisibility::Advanced
        );
        assert_eq!(
            schema_setting("defaults.hardware.model_runtime").visibility,
            ConfigVisibility::Hidden
        );
        assert_eq!(
            schema_setting("defaults.advanced.server.alias").visibility,
            ConfigVisibility::Advanced
        );
    }

    #[test]
    fn built_in_schema_uses_explicit_path_and_url_value_kinds() {
        assert_eq!(schema_value("telemetry.endpoint"), ConfigValueSchema::Url);
        assert_eq!(
            schema_value("telemetry.metrics.endpoint"),
            ConfigValueSchema::Url
        );
        assert_eq!(
            schema_value("plugin.<plugin-name>.url"),
            ConfigValueSchema::Url
        );
        assert_eq!(
            schema_value("defaults.hardware.model_path"),
            ConfigValueSchema::Path
        );
        assert_eq!(
            schema_value("defaults.hardware.mmproj"),
            ConfigValueSchema::Path
        );
        assert_eq!(
            schema_value("defaults.multimodal.mmproj"),
            ConfigValueSchema::Path
        );
        assert_eq!(
            schema_value("defaults.multimodal.mmproj_url"),
            ConfigValueSchema::Url
        );
        assert_eq!(
            schema_value("defaults.speculative.draft_model"),
            ConfigValueSchema::Path
        );
    }

    #[test]
    fn startup_runtime_settings_require_process_restart() {
        for path in ["runtime.debug", "runtime.listen_all"] {
            let setting = schema_setting(path);

            assert_eq!(
                setting.control_surfaces,
                vec![ConfigControlSurface::ConfigFile, ConfigControlSurface::Api],
                "{path}"
            );
            assert_eq!(setting.apply_mode, ConfigApplyMode::StaticOnLoad, "{path}");
            assert_eq!(
                setting.restart_scope,
                ConfigRestartScope::ProcessRestart,
                "{path}"
            );
        }
    }

    #[test]
    fn built_in_schema_exports_model_fit_numeric_controls_and_relative_bounds() {
        let defaults_batch = schema_setting("defaults.model_fit.batch");
        let defaults_ubatch = schema_setting("defaults.model_fit.ubatch");
        let model_ubatch = schema_setting("models.<model-ref>.model_fit.ubatch");

        assert_eq!(numeric_control(&defaults_batch).min, Some(1.0));
        assert_eq!(numeric_control(&defaults_batch).step, Some(1.0));
        assert_eq!(
            numeric_control(&defaults_batch).unit.as_deref(),
            Some("tokens")
        );

        assert_eq!(
            numeric_control(&defaults_ubatch),
            numeric_control(&model_ubatch)
        );
        assert_has_range_constraint(&defaults_ubatch, None, Some("defaults.model_fit.batch"));
        assert_has_range_constraint(
            &model_ubatch,
            None,
            Some("models.<model-ref>.model_fit.batch"),
        );
    }

    #[test]
    fn built_in_schema_keeps_defaults_and_model_hardware_device_semantics_in_sync() {
        let defaults_device = schema_setting("defaults.hardware.device");
        let model_device = schema_setting("models.<model-ref>.hardware.device");

        assert_eq!(
            control_behavior(&defaults_device).options_source,
            Some(ConfigOptionsSource::RuntimeGpus)
        );
        assert_eq!(
            defaults_device.control_behavior,
            model_device.control_behavior
        );
        assert_eq!(
            control_behavior(&defaults_device).enable_when,
            vec![equals_condition("gpu.assignment", "pinned")]
        );
        assert_eq!(
            control_behavior(&defaults_device).disable_when,
            vec![dependency_disable(
                equals_condition("gpu.assignment", "auto"),
                "Set gpu.assignment = \"pinned\" to edit a concrete GPU device.",
            )]
        );
    }

    #[test]
    fn built_in_schema_marks_rejected_hardware_escape_hatches_non_editable() {
        let setting = schema_setting("models.<model-ref>.hardware.rpc_backend");
        let behavior = control_behavior(&setting);

        assert_eq!(setting.support, ConfigSupportState::Rejected);
        assert_eq!(
            behavior.availability.as_ref().map(|value| value.enabled),
            Some(false)
        );
        assert_eq!(
            behavior.availability.as_ref().map(|value| value.source),
            Some(ConfigControlAvailabilitySource::Static)
        );
        assert_eq!(
            setting.default_disabled_write_policy(None),
            Some(ConfigDisabledWritePolicy::RejectWhenDisabled)
        );
    }

    #[test]
    fn built_in_schema_exports_throughput_and_skippy_t5_controls() {
        let threads = schema_setting("defaults.throughput.threads");
        let prefill_chunk_size = schema_setting("defaults.skippy.prefill_chunk_size");
        let prefill_chunk_schedule = schema_setting("defaults.skippy.prefill_chunk_schedule");

        assert_static_choices(
            "defaults.throughput.tuning_profile",
            &["throughput", "balanced", "saver"],
        );
        assert_eq!(numeric_control(&threads).min, Some(0.0));
        assert_eq!(numeric_control(&threads).step, Some(1.0));

        assert_static_choices(
            "defaults.skippy.activation_wire_dtype",
            &["auto", "f16", "f32", "q8"],
        );
        assert_static_choices(
            "defaults.skippy.prefill_chunking",
            &["auto", "fixed", "schedule", "adaptive-ramp"],
        );
        assert_eq!(numeric_control(&prefill_chunk_size).min, Some(1.0));
        assert_eq!(
            control_behavior(&prefill_chunk_size).enable_when,
            vec![equals_condition(
                "defaults.skippy.prefill_chunking",
                "fixed"
            )]
        );
        assert_eq!(
            control_behavior(&prefill_chunk_schedule).text_format,
            Some(ConfigTextFormat::CsvPositiveInts)
        );
        assert_eq!(
            control_behavior(&prefill_chunk_schedule).enable_when,
            vec![equals_condition(
                "defaults.skippy.prefill_chunking",
                "schedule"
            )]
        );
    }

    #[test]
    fn built_in_schema_exports_speculative_and_request_default_t5_controls() {
        let draft_min = schema_setting("defaults.speculative.draft_min_tokens");
        let ngram_max = schema_setting("defaults.speculative.ngram_max");
        let mirostat_entropy = schema_setting("defaults.request_defaults.mirostat_entropy");

        assert_static_choices("defaults.speculative.mode", &["auto", "disabled", "draft"]);
        assert_static_choices(
            "defaults.speculative.draft_selection_policy",
            &["manual", "auto"],
        );
        assert_static_choices(
            "defaults.speculative.pairing_fault",
            &[
                "warn_disable",
                "fail-open",
                "fail-closed",
                "fail_open",
                "fail_closed",
            ],
        );
        assert_has_range_constraint(
            &draft_min,
            None,
            Some("defaults.speculative.draft_max_tokens"),
        );
        assert_has_range_constraint(&ngram_max, Some("defaults.speculative.ngram_min"), None);

        assert_static_choices(
            "defaults.request_defaults.reasoning_format",
            &["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
        );
        assert_eq!(
            control_behavior(&mirostat_entropy).enable_when,
            vec![in_condition(
                "defaults.request_defaults.mirostat_mode",
                &[
                    ConfigConditionValue::Integer(1),
                    ConfigConditionValue::Integer(2),
                    ConfigConditionValue::String("1".to_string()),
                    ConfigConditionValue::String("2".to_string()),
                ],
            )]
        );
        assert_eq!(
            control_behavior(&mirostat_entropy).disable_when,
            vec![dependency_disable(
                not_in_condition(
                    "defaults.request_defaults.mirostat_mode",
                    &[
                        ConfigConditionValue::Integer(1),
                        ConfigConditionValue::Integer(2),
                        ConfigConditionValue::String("1".to_string()),
                        ConfigConditionValue::String("2".to_string()),
                    ],
                ),
                "defaults.request_defaults.mirostat_entropy requires defaults.request_defaults.mirostat_mode = 1 or 2",
            )]
        );
    }

    #[test]
    fn built_in_schema_disables_duplicate_multimodal_projector_controls_with_preserve_policy() {
        let mmproj = schema_setting("defaults.hardware.mmproj");
        let offload = schema_setting("defaults.hardware.mmproj_offload");

        assert_eq!(
            control_behavior(&mmproj).availability,
            Some(ConfigControlAvailability {
                enabled: false,
                reason: Some(
                    "Edit defaults.multimodal.mmproj instead of the legacy hardware duplicate."
                        .to_string(),
                ),
                note: Some(
                    "Existing values are preserved on save unless you change defaults.multimodal.mmproj."
                        .to_string(),
                ),
                source: ConfigControlAvailabilitySource::Static,
            })
        );
        assert_eq!(
            control_behavior(&mmproj).write_policy,
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
        assert_eq!(
            control_behavior(&offload)
                .availability
                .as_ref()
                .map(|value| value.enabled),
            Some(false)
        );
        assert_eq!(
            control_behavior(&offload).write_policy,
            Some(ConfigDisabledWritePolicy::PreserveExisting)
        );
    }

    #[test]
    fn built_in_schema_exports_telemetry_owner_control_attestation_and_plugin_timeout_controls() {
        let telemetry_interval = schema_setting("telemetry.export_interval_secs");
        let advertise_addr = schema_setting("owner_control.advertise_addr");
        let signer_keys = schema_setting("mesh_requirements.release_signer_keys");
        let plugin_timeout = schema_setting("plugin.<plugin-name>.startup.connect_timeout_secs");

        assert_eq!(numeric_control(&telemetry_interval).min, Some(1.0));
        assert_eq!(
            numeric_control(&telemetry_interval).unit.as_deref(),
            Some("sec")
        );

        assert_eq!(
            control_behavior(&advertise_addr).enable_when,
            vec![present_condition("owner_control.bind")]
        );
        assert_eq!(
            control_behavior(&advertise_addr).disable_when,
            vec![dependency_disable(
                absent_condition("owner_control.bind"),
                "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening",
            )]
        );

        assert_eq!(
            control_behavior(&schema_setting("mesh_requirements.min_node_version")).text_format,
            Some(ConfigTextFormat::Semver)
        );
        assert_eq!(
            control_behavior(&signer_keys).text_format,
            Some(ConfigTextFormat::Ed25519Key)
        );
        assert_eq!(
            control_behavior(&signer_keys).enable_when,
            vec![equals_bool_condition(
                "mesh_requirements.require_release_attestation",
                true,
            )]
        );

        assert_eq!(numeric_control(&plugin_timeout).min, Some(1.0));
        assert_eq!(
            numeric_control(&plugin_timeout).unit.as_deref(),
            Some("sec")
        );
    }

    #[test]
    fn built_in_schema_covers_t5_fallback_choices_or_keeps_open_text_intentional() {
        for (path, expected) in [
            (
                "defaults.throughput.tuning_profile",
                vec!["throughput", "balanced", "saver"],
            ),
            (
                "defaults.speculative.mode",
                vec!["auto", "disabled", "draft"],
            ),
            (
                "defaults.speculative.draft_selection_policy",
                vec!["manual", "auto"],
            ),
            (
                "defaults.speculative.pairing_fault",
                vec![
                    "warn_disable",
                    "fail-open",
                    "fail-closed",
                    "fail_open",
                    "fail_closed",
                ],
            ),
            (
                "defaults.request_defaults.reasoning_format",
                vec!["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
            ),
            (
                "defaults.speculative.draft_cache_type_k",
                vec![
                    "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
                ],
            ),
            (
                "defaults.speculative.draft_cache_type_v",
                vec![
                    "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
                ],
            ),
        ] {
            assert_eq!(
                schema_enum_values(path),
                expected.into_iter().map(str::to_string).collect::<Vec<_>>(),
                "{path}"
            );
        }

        for path in [
            "defaults.throughput.numa",
            "defaults.skippy.binary_stage_transport",
        ] {
            assert!(schema_enum_values(path).is_empty(), "{path}");
            assert_ne!(
                schema_setting(path)
                    .control_behavior
                    .as_ref()
                    .and_then(|behavior| behavior.options_source),
                Some(ConfigOptionsSource::Static),
                "{path}"
            );
        }
    }

    fn schema_value(path: &str) -> ConfigValueSchema {
        schema_setting(path).value_schema
    }

    fn schema_setting(path: &str) -> ConfigSettingSchema {
        built_in_config_schema_descriptor(&schema_path(path)).expect("schema setting should exist")
    }

    fn control_behavior(setting: &ConfigSettingSchema) -> &ConfigControlBehavior {
        setting
            .control_behavior
            .as_ref()
            .expect("control behavior should be present")
    }

    fn numeric_control(setting: &ConfigSettingSchema) -> ConfigNumericControl {
        control_behavior(setting)
            .numeric
            .clone()
            .expect("numeric control should be present")
    }

    fn assert_has_range_constraint(
        setting: &ConfigSettingSchema,
        expected_min: Option<&str>,
        expected_max: Option<&str>,
    ) {
        assert!(
            setting.constraints.iter().any(|constraint| {
                matches!(
                    constraint,
                    ConfigConstraint::Range { min, max }
                        if min.as_deref() == expected_min && max.as_deref() == expected_max
                )
            }),
            "expected range constraint min={expected_min:?} max={expected_max:?} on {}",
            setting.path.render()
        );
    }

    fn equals_condition(path: &str, expected: &str) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::Equals,
            values: vec![ConfigConditionValue::String(expected.to_string())],
        }
    }

    fn equals_bool_condition(path: &str, expected: bool) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::Equals,
            values: vec![ConfigConditionValue::Bool(expected)],
        }
    }

    fn in_condition(path: &str, values: &[ConfigConditionValue]) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::In,
            values: values.to_vec(),
        }
    }

    fn not_in_condition(path: &str, values: &[ConfigConditionValue]) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::NotIn,
            values: values.to_vec(),
        }
    }

    fn present_condition(path: &str) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::Present,
            values: Vec::new(),
        }
    }

    fn absent_condition(path: &str) -> ConfigControlCondition {
        ConfigControlCondition {
            path: schema_path(path),
            operator: ConfigConditionOperator::Absent,
            values: Vec::new(),
        }
    }

    fn dependency_disable(
        condition: ConfigControlCondition,
        reason: &str,
    ) -> ConfigConditionalDisable {
        ConfigConditionalDisable {
            condition,
            reason: reason.to_string(),
            note: None,
            write_policy: ConfigDisabledWritePolicy::OmitWhenDisabled,
        }
    }

    fn assert_static_choices(path: &str, expected: &[&str]) {
        let setting = schema_setting(path);

        assert_eq!(
            control_behavior(&setting).options_source,
            Some(ConfigOptionsSource::Static),
            "{path}"
        );
        assert_eq!(
            schema_enum_values(path),
            expected
                .iter()
                .map(|value| (*value).to_string())
                .collect::<Vec<_>>(),
            "{path}"
        );
    }

    fn schema_enum_values(path: &str) -> Vec<String> {
        enum_values(&schema_value(path))
    }

    fn enum_values(schema: &ConfigValueSchema) -> Vec<String> {
        match schema {
            ConfigValueSchema::Enum { values } => values.clone(),
            ConfigValueSchema::OneOf { variants } => {
                variants.iter().flat_map(enum_values).collect()
            }
            _ => Vec::new(),
        }
    }
}

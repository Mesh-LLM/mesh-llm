use super::super::{
    HardwareConfig, ModelConfigDefaults, ModelConfigEntry, ModelFitConfig, ThroughputConfig,
    merge_model_topology,
};

impl ModelConfigEntry {
    /// Merge profile-shaping defaults beneath this model entry.
    pub fn with_profile_defaults(&self, defaults: Option<&ModelConfigDefaults>) -> Self {
        let Some(defaults) = defaults else {
            return self.clone();
        };
        let mut effective = self.clone();
        merge_model_fit(&mut effective, defaults);
        merge_hardware(&mut effective, defaults);
        merge_throughput(&mut effective, defaults);
        effective.topology =
            merge_model_topology(defaults.topology.as_ref(), effective.topology.as_ref());
        effective
    }
}

fn merge_model_fit(effective: &mut ModelConfigEntry, defaults: &ModelConfigDefaults) {
    let Some(default_fit) = defaults.model_fit.as_ref() else {
        return;
    };
    let fit = effective
        .model_fit
        .get_or_insert_with(ModelFitConfig::default);
    fit.ctx_size = fit.ctx_size.or(default_fit.ctx_size);
    fit.batch = fit.batch.or(default_fit.batch);
    fit.ubatch = fit.ubatch.or(default_fit.ubatch);
    fit.cache_type_k = fit
        .cache_type_k
        .clone()
        .or_else(|| default_fit.cache_type_k.clone());
    fit.cache_type_v = fit
        .cache_type_v
        .clone()
        .or_else(|| default_fit.cache_type_v.clone());
    fit.flash_attention = fit.flash_attention.or(default_fit.flash_attention);
    fit.kv_cache_policy = fit
        .kv_cache_policy
        .clone()
        .or_else(|| default_fit.kv_cache_policy.clone());
    fit.kv_offload = fit.kv_offload.clone().or(default_fit.kv_offload.clone());
    fit.kv_unified = fit.kv_unified.clone().or(default_fit.kv_unified.clone());
    fit.cache_ram_mib = fit.cache_ram_mib.or(default_fit.cache_ram_mib);
    fit.cache_idle_slots = fit.cache_idle_slots.or(default_fit.cache_idle_slots);
    fit.prompt_cache = fit
        .prompt_cache
        .clone()
        .or(default_fit.prompt_cache.clone());
    fit.prefix_cache = fit
        .prefix_cache
        .clone()
        .or_else(|| default_fit.prefix_cache.clone());
    fit.keep_tokens = fit.keep_tokens.or(default_fit.keep_tokens);
    fit.context_shift = fit
        .context_shift
        .clone()
        .or(default_fit.context_shift.clone());
    fit.swa_full = fit.swa_full.or(default_fit.swa_full);
    fit.checkpoint_interval = fit.checkpoint_interval.or(default_fit.checkpoint_interval);
    fit.checkpoint_count = fit.checkpoint_count.or(default_fit.checkpoint_count);
    fit.lookup_cache_static = fit
        .lookup_cache_static
        .clone()
        .or_else(|| default_fit.lookup_cache_static.clone());
    fit.lookup_cache_dynamic = fit
        .lookup_cache_dynamic
        .clone()
        .or_else(|| default_fit.lookup_cache_dynamic.clone());
}

fn merge_hardware(effective: &mut ModelConfigEntry, defaults: &ModelConfigDefaults) {
    let Some(default_hardware) = defaults.hardware.as_ref() else {
        return;
    };
    let hardware = effective
        .hardware
        .get_or_insert_with(HardwareConfig::default);
    hardware.device = hardware
        .device
        .clone()
        .or_else(|| default_hardware.device.clone());
    hardware.model_runtime = hardware.model_runtime.or(default_hardware.model_runtime);
    hardware.stage_layer_start = hardware
        .stage_layer_start
        .or(default_hardware.stage_layer_start);
    hardware.stage_layer_end = hardware
        .stage_layer_end
        .or(default_hardware.stage_layer_end);
    hardware.gpu_layers = hardware
        .gpu_layers
        .clone()
        .or_else(|| default_hardware.gpu_layers.clone());
    hardware.tensor_split = hardware
        .tensor_split
        .clone()
        .or_else(|| default_hardware.tensor_split.clone());
    hardware.split_mode = hardware
        .split_mode
        .clone()
        .or_else(|| default_hardware.split_mode.clone());
    hardware.main_gpu = hardware.main_gpu.or(default_hardware.main_gpu);
    hardware.cpu_moe = hardware
        .cpu_moe
        .clone()
        .or_else(|| default_hardware.cpu_moe.clone());
    hardware.n_cpu_moe = hardware.n_cpu_moe.or(default_hardware.n_cpu_moe);
    hardware.fit_target_mib = hardware.fit_target_mib.or(default_hardware.fit_target_mib);
    hardware.mmap = hardware
        .mmap
        .clone()
        .or_else(|| default_hardware.mmap.clone());
    hardware.use_mmap_prefetch = hardware
        .use_mmap_prefetch
        .or(default_hardware.use_mmap_prefetch);
    hardware.use_mmap_buffer = hardware
        .use_mmap_buffer
        .or(default_hardware.use_mmap_buffer);
    hardware.mlock = hardware.mlock.or(default_hardware.mlock);
    hardware.safety_margin_gb = hardware
        .safety_margin_gb
        .or(default_hardware.safety_margin_gb);
    hardware.fit_context = hardware
        .fit_context
        .clone()
        .or(default_hardware.fit_context.clone());
    hardware.model_path = hardware
        .model_path
        .clone()
        .or_else(|| default_hardware.model_path.clone());
    hardware.hf_repo = hardware
        .hf_repo
        .clone()
        .or_else(|| default_hardware.hf_repo.clone());
    hardware.hf_file = hardware
        .hf_file
        .clone()
        .or_else(|| default_hardware.hf_file.clone());
    hardware.mmproj = hardware
        .mmproj
        .clone()
        .or_else(|| default_hardware.mmproj.clone());
    hardware.mmproj_offload = hardware
        .mmproj_offload
        .clone()
        .or(default_hardware.mmproj_offload.clone());
    if hardware.lora_adapters.is_empty() {
        hardware
            .lora_adapters
            .clone_from(&default_hardware.lora_adapters);
    }
    if hardware.control_vectors.is_empty() {
        hardware
            .control_vectors
            .clone_from(&default_hardware.control_vectors);
    }
    hardware.check_tensors = hardware.check_tensors.or(default_hardware.check_tensors);
    hardware.direct_io = hardware.direct_io.or(default_hardware.direct_io);
    hardware.repack = hardware.repack.or(default_hardware.repack);
    hardware.op_offload = hardware.op_offload.or(default_hardware.op_offload);
    hardware.no_host_buffer = hardware.no_host_buffer.or(default_hardware.no_host_buffer);
    hardware.warmup = hardware.warmup.clone().or(default_hardware.warmup.clone());
}

fn merge_throughput(effective: &mut ModelConfigEntry, defaults: &ModelConfigDefaults) {
    let Some(default_throughput) = defaults.throughput.as_ref() else {
        return;
    };
    let throughput = effective
        .throughput
        .get_or_insert_with(ThroughputConfig::default);
    throughput.parallel = throughput.parallel.or(default_throughput.parallel);
    throughput.continuous_batching = throughput
        .continuous_batching
        .clone()
        .or_else(|| default_throughput.continuous_batching.clone());
    throughput.threads = throughput.threads.or(default_throughput.threads);
    throughput.threads_batch = throughput
        .threads_batch
        .or(default_throughput.threads_batch);
    throughput.threads_http = throughput.threads_http.or(default_throughput.threads_http);
    throughput.priority = throughput
        .priority
        .clone()
        .or(default_throughput.priority.clone());
    throughput.poll = throughput.poll.clone().or(default_throughput.poll.clone());
    throughput.cpu_affinity = throughput
        .cpu_affinity
        .clone()
        .or(default_throughput.cpu_affinity.clone());
    throughput.numa = throughput
        .numa
        .clone()
        .or_else(|| default_throughput.numa.clone());
    throughput.slot_prompt_similarity = throughput
        .slot_prompt_similarity
        .or(default_throughput.slot_prompt_similarity);
    throughput.sleep_idle_seconds = throughput
        .sleep_idle_seconds
        .or(default_throughput.sleep_idle_seconds);
    throughput.tuning_profile = throughput
        .tuning_profile
        .clone()
        .or_else(|| default_throughput.tuning_profile.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ModelTopologyConfig, ModelTopologyMode, ModelTopologyNodeSelector, ModelTopologyStageConfig,
    };

    #[test]
    fn profile_defaults_inherit_stage_boundaries_without_overriding_model_values() {
        let defaults = ModelConfigDefaults {
            hardware: Some(HardwareConfig {
                stage_layer_start: Some(4),
                stage_layer_end: Some(20),
                ..HardwareConfig::default()
            }),
            ..ModelConfigDefaults::default()
        };

        let inherited = ModelConfigEntry {
            model: "inherited".to_string(),
            ..ModelConfigEntry::default()
        }
        .with_profile_defaults(Some(&defaults));
        assert_eq!(
            inherited
                .hardware
                .as_ref()
                .map(|hardware| (hardware.stage_layer_start, hardware.stage_layer_end)),
            Some((Some(4), Some(20)))
        );

        let overridden = ModelConfigEntry {
            model: "overridden".to_string(),
            hardware: Some(HardwareConfig {
                stage_layer_start: Some(8),
                stage_layer_end: Some(24),
                ..HardwareConfig::default()
            }),
            ..ModelConfigEntry::default()
        }
        .with_profile_defaults(Some(&defaults));
        assert_eq!(
            overridden
                .hardware
                .as_ref()
                .map(|hardware| (hardware.stage_layer_start, hardware.stage_layer_end)),
            Some((Some(8), Some(24)))
        );
    }

    #[test]
    fn profile_defaults_merge_topology_without_overriding_model_values() {
        let defaults = ModelConfigDefaults {
            topology: Some(ModelTopologyConfig {
                mode: Some(ModelTopologyMode::Locked),
                manifest_sha256: Some("d".repeat(64)),
                stages: Some(vec![ModelTopologyStageConfig {
                    node: ModelTopologyNodeSelector {
                        hostname: Some("default.example".to_string()),
                        ..ModelTopologyNodeSelector::default()
                    },
                    layer_start: 0,
                    layer_end: 10,
                }]),
            }),
            ..ModelConfigDefaults::default()
        };
        let model_stages = vec![ModelTopologyStageConfig {
            node: ModelTopologyNodeSelector {
                endpoint_id: Some("model-endpoint".to_string()),
                ..ModelTopologyNodeSelector::default()
            },
            layer_start: 0,
            layer_end: 20,
        }];
        let model = ModelConfigEntry {
            model: "model".to_string(),
            topology: Some(ModelTopologyConfig {
                stages: Some(model_stages.clone()),
                ..ModelTopologyConfig::default()
            }),
            ..ModelConfigEntry::default()
        };

        let effective = model.with_profile_defaults(Some(&defaults));

        assert_eq!(
            effective.topology,
            Some(ModelTopologyConfig {
                mode: Some(ModelTopologyMode::Locked),
                manifest_sha256: Some("d".repeat(64)),
                stages: Some(model_stages),
            })
        );
    }
}

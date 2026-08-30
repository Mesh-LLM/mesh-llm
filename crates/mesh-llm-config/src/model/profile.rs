use super::{HardwareConfig, ModelConfigEntry, ModelFitConfig, ThroughputConfig};
use std::hash::{Hash, Hasher};
use std::io::Write;

mod defaults;

macro_rules! write_option {
    ($buffer:expr, $key:literal, $value:expr) => {
        if let Some(ref value) = $value {
            let _ = write!($buffer, concat!($key, "={:?}\0"), value);
        }
    };
}

impl ModelConfigEntry {
    /// Compute a derived profile hash from the runtime-shaping fields of this entry.
    ///
    /// Returns an 8-hex-character string, or an empty string when all profile-input
    /// fields are at their defaults.
    pub fn derived_profile(&self) -> String {
        let mut buffer = Vec::new();
        write_effective_fit_profile(&mut buffer, self);
        write_effective_hardware_profile(&mut buffer, self);
        write_effective_throughput_profile(&mut buffer, self);
        write_effective_topology_profile(&mut buffer, self);

        if buffer.is_empty() {
            return String::new();
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        buffer.hash(&mut hasher);
        format!("{:08x}", hasher.finish() & 0xFFFF_FFFF)
    }
}

fn write_effective_fit_profile(buffer: &mut Vec<u8>, entry: &ModelConfigEntry) {
    let fit = entry.model_fit.as_ref();
    write_option!(
        buffer,
        "ctx_size",
        fit.and_then(|fit| fit.ctx_size).or(entry.ctx_size)
    );
    write_option!(
        buffer,
        "batch",
        fit.and_then(|fit| fit.batch).or(entry.batch)
    );
    write_option!(
        buffer,
        "ubatch",
        fit.and_then(|fit| fit.ubatch).or(entry.ubatch)
    );
    write_option!(
        buffer,
        "cache_type_k",
        fit.and_then(|fit| fit.cache_type_k.as_ref())
            .or(entry.cache_type_k.as_ref())
    );
    write_option!(
        buffer,
        "cache_type_v",
        fit.and_then(|fit| fit.cache_type_v.as_ref())
            .or(entry.cache_type_v.as_ref())
    );
    write_option!(
        buffer,
        "flash_attention",
        fit.and_then(|fit| fit.flash_attention)
            .or(entry.flash_attention)
    );
    if let Some(fit) = fit {
        write_fit_cache_profile(buffer, fit);
    }
}

fn write_fit_cache_profile(buffer: &mut Vec<u8>, fit: &ModelFitConfig) {
    write_option!(buffer, "kv_cache_policy", fit.kv_cache_policy);
    write_option!(buffer, "kv_offload", fit.kv_offload);
    write_option!(buffer, "kv_unified", fit.kv_unified);
    write_option!(buffer, "cache_ram_mib", fit.cache_ram_mib);
    write_option!(buffer, "cache_idle_slots", fit.cache_idle_slots);
    write_option!(buffer, "prompt_cache", fit.prompt_cache);
    write_option!(buffer, "prefix_cache", fit.prefix_cache);
    write_option!(buffer, "keep_tokens", fit.keep_tokens);
    write_option!(buffer, "context_shift", fit.context_shift);
    write_option!(buffer, "swa_full", fit.swa_full);
    write_option!(buffer, "checkpoint_interval", fit.checkpoint_interval);
    write_option!(buffer, "checkpoint_count", fit.checkpoint_count);
    write_option!(buffer, "lookup_cache_static", fit.lookup_cache_static);
    write_option!(buffer, "lookup_cache_dynamic", fit.lookup_cache_dynamic);
}

fn write_effective_hardware_profile(buffer: &mut Vec<u8>, entry: &ModelConfigEntry) {
    let hardware = entry.hardware.as_ref();
    write_option!(
        buffer,
        "gpu_id",
        hardware
            .and_then(|hardware| hardware.device.as_ref())
            .or(entry.gpu_id.as_ref())
    );
    if let Some(hardware) = hardware {
        write_hardware_placement_profile(buffer, hardware);
        write_hardware_source_profile(buffer, hardware);
        write_hardware_flag_profile(buffer, hardware);
    }
}

fn write_hardware_placement_profile(buffer: &mut Vec<u8>, hardware: &HardwareConfig) {
    write_option!(buffer, "model_runtime", hardware.model_runtime);
    write_option!(buffer, "gpu_layers", hardware.gpu_layers);
    write_option!(buffer, "tensor_split", hardware.tensor_split);
    write_option!(buffer, "split_mode", hardware.split_mode);
    write_option!(buffer, "main_gpu", hardware.main_gpu);
    write_option!(buffer, "cpu_moe", hardware.cpu_moe);
    write_option!(buffer, "n_cpu_moe", hardware.n_cpu_moe);
    write_option!(buffer, "fit_target_mib", hardware.fit_target_mib);
    write_option!(buffer, "safety_margin_gb", hardware.safety_margin_gb);
    write_option!(buffer, "fit_context", hardware.fit_context);
}

fn write_hardware_source_profile(buffer: &mut Vec<u8>, hardware: &HardwareConfig) {
    write_option!(buffer, "model_path", hardware.model_path);
    write_option!(buffer, "hf_repo", hardware.hf_repo);
    write_option!(buffer, "hf_file", hardware.hf_file);
    write_option!(buffer, "mmproj", hardware.mmproj);
    write_option!(buffer, "mmproj_offload", hardware.mmproj_offload);
    if !hardware.lora_adapters.is_empty() {
        let _ = write!(buffer, "lora_adapters={:?}\0", hardware.lora_adapters);
    }
    if !hardware.control_vectors.is_empty() {
        let _ = write!(buffer, "control_vectors={:?}\0", hardware.control_vectors);
    }
}

fn write_hardware_flag_profile(buffer: &mut Vec<u8>, hardware: &HardwareConfig) {
    write_option!(buffer, "check_tensors", hardware.check_tensors);
    write_option!(buffer, "mmap", hardware.mmap);
    write_option!(buffer, "use_mmap_prefetch", hardware.use_mmap_prefetch);
    write_option!(buffer, "use_mmap_buffer", hardware.use_mmap_buffer);
    write_option!(buffer, "mlock", hardware.mlock);
    write_option!(buffer, "direct_io", hardware.direct_io);
    write_option!(buffer, "repack", hardware.repack);
    write_option!(buffer, "op_offload", hardware.op_offload);
    write_option!(buffer, "no_host_buffer", hardware.no_host_buffer);
    write_option!(buffer, "warmup", hardware.warmup);
}

fn write_effective_throughput_profile(buffer: &mut Vec<u8>, entry: &ModelConfigEntry) {
    let throughput = entry.throughput.as_ref();
    write_option!(
        buffer,
        "parallel",
        throughput
            .and_then(|throughput| throughput.parallel)
            .or(entry.parallel)
    );
    if let Some(throughput) = throughput {
        write_throughput_fields(buffer, throughput);
    }
}

fn write_effective_topology_profile(buffer: &mut Vec<u8>, entry: &ModelConfigEntry) {
    let Some(topology) = entry.topology.as_ref() else {
        return;
    };
    write_option!(buffer, "topology_mode", topology.mode);
    write_option!(buffer, "topology_manifest_sha256", topology.manifest_sha256);
    write_option!(buffer, "topology_stages", topology.stages);
}

fn write_throughput_fields(buffer: &mut Vec<u8>, throughput: &ThroughputConfig) {
    write_option!(
        buffer,
        "continuous_batching",
        throughput.continuous_batching
    );
    write_option!(buffer, "threads", throughput.threads);
    write_option!(buffer, "threads_batch", throughput.threads_batch);
    write_option!(buffer, "threads_http", throughput.threads_http);
    write_option!(buffer, "priority", throughput.priority);
    write_option!(buffer, "poll", throughput.poll);
    write_option!(buffer, "cpu_affinity", throughput.cpu_affinity);
    write_option!(buffer, "numa", throughput.numa);
    write_option!(
        buffer,
        "slot_prompt_similarity",
        throughput.slot_prompt_similarity
    );
    write_option!(buffer, "sleep_idle_seconds", throughput.sleep_idle_seconds);
    write_option!(buffer, "tuning_profile", throughput.tuning_profile);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelTopologyConfig, ModelTopologyMode};

    #[test]
    fn derived_profile_includes_topology() {
        let entry = ModelConfigEntry {
            model: "model".to_string(),
            ..ModelConfigEntry::default()
        };
        let topology_entry = ModelConfigEntry {
            model: "model".to_string(),
            topology: Some(ModelTopologyConfig {
                mode: Some(ModelTopologyMode::Locked),
                ..ModelTopologyConfig::default()
            }),
            ..ModelConfigEntry::default()
        };

        assert_ne!(entry.derived_profile(), topology_entry.derived_profile());
    }
}

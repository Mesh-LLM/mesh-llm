use crate::NativeRuntimeFlavor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostGpuProfile {
    pub display_name: String,
    pub backend_device: Option<String>,
    pub stable_id: Option<String>,
    pub vram_bytes: Option<u64>,
    pub unified_memory: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostRuntimeProfile {
    pub os: String,
    pub arch: String,
    pub target_triple: Option<String>,
    pub available_flavors: BTreeSet<NativeRuntimeFlavor>,
    pub gpus: Vec<HostGpuProfile>,
}

impl HostRuntimeProfile {
    pub fn current_without_gpu_probe() -> Self {
        let mut available_flavors = BTreeSet::from([NativeRuntimeFlavor::Cpu]);
        if cfg!(target_os = "macos") {
            available_flavors.insert(NativeRuntimeFlavor::Metal);
        }
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            target_triple: option_env!("TARGET").map(str::to_string),
            available_flavors,
            gpus: Vec::new(),
        }
    }

    pub fn supports_flavor(&self, flavor: &NativeRuntimeFlavor) -> bool {
        self.available_flavors.contains(flavor)
    }

    pub fn has_gpu_name_matching(&self, needle: &str) -> bool {
        let needle = needle.trim().to_ascii_lowercase();
        !needle.is_empty()
            && self
                .gpus
                .iter()
                .any(|gpu| gpu.display_name.to_ascii_lowercase().contains(&needle))
    }
}

use crate::NativeRuntimeBackendKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostGpuProfile {
    pub display_name: String,
    pub backend_device: Option<String>,
    pub stable_id: Option<String>,
    pub vram_bytes: Option<u64>,
    pub unified_memory: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_sm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm_gfx: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostCudaProfile {
    #[serde(default)]
    pub toolkit_majors: BTreeSet<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
    #[serde(default)]
    pub gpu_arches: BTreeSet<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostRocmProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub gpu_arches: BTreeSet<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostVulkanProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HostRuntimeProfile {
    pub os: String,
    pub arch: String,
    pub target_triple: Option<String>,
    pub available_flavors: BTreeSet<NativeRuntimeBackendKind>,
    pub gpus: Vec<HostGpuProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<HostCudaProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm: Option<HostRocmProfile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan: Option<HostVulkanProfile>,
}

impl HostRuntimeProfile {
    pub fn current_without_gpu_probe() -> Self {
        let mut available_flavors = BTreeSet::from([NativeRuntimeBackendKind::Cpu]);
        if cfg!(target_os = "macos") {
            available_flavors.insert(NativeRuntimeBackendKind::Metal);
        }
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            target_triple: option_env!("TARGET").map(str::to_string),
            available_flavors,
            gpus: Vec::new(),
            cuda: None,
            rocm: None,
            vulkan: None,
        }
    }

    pub fn supports_flavor(&self, flavor: &NativeRuntimeBackendKind) -> bool {
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

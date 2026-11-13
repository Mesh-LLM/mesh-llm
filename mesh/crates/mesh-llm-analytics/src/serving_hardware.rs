//! The machine class attached to every per-request serving event.
//!
//! `hardware_profile` describes a machine once per process, but a request
//! event cannot be joined back to it reliably: an install's profile can
//! change between runs, and the per-minute sampling of request events makes
//! the join lossy. So the few dimensions that explain serving speed are
//! copied onto each `inference_request` instead. Each one is low-cardinality
//! by construction.

use crate::event::{Properties, Value};
use std::sync::OnceLock;

static SERVING_HARDWARE: OnceLock<ServingHardware> = OnceLock::new();

/// Machine dimensions for per-request events.
#[derive(Clone, Debug, PartialEq)]
pub struct ServingHardware {
    /// The compute backend, one of `cuda`, `rocm`, `metal`, `vulkan`, `cpu`.
    pub backend: &'static str,
    /// A device family such as `apple-m4` or `rtx-4090`, or `none` and
    /// `unreported` as `hardware_profile` uses them.
    pub gpu_family: Value,
    /// GPU memory in whole GiB, without any RAM-offload credit.
    pub vram_gb: Option<u64>,
}

/// The backend a runtime device name belongs to: `CUDA0` is `cuda`, `MTL0`
/// is `metal`, `Vulkan0` is `vulkan`, `ROCm0` and `HIP0` are `rocm`.
#[must_use]
pub fn backend_of_device(device: &str) -> Option<&'static str> {
    let device = device.to_ascii_lowercase();
    [
        ("cuda", "cuda"),
        ("rocm", "rocm"),
        ("hip", "rocm"),
        ("mtl", "metal"),
        ("metal", "metal"),
        ("vulkan", "vulkan"),
    ]
    .into_iter()
    .find(|(prefix, _)| device.starts_with(prefix))
    .map(|(_, backend)| backend)
}

impl ServingHardware {
    /// Attach these dimensions to `properties`, naming `backend` as the
    /// backend when the caller knows the one that actually served.
    #[must_use]
    pub fn apply(&self, properties: Properties, backend: Option<&'static str>) -> Properties {
        let properties = properties
            .with("backend", backend.unwrap_or(self.backend))
            .with("gpu_family", self.gpu_family.clone());
        match self.vram_gb {
            Some(vram_gb) => properties.with("vram_gb", vram_gb),
            None => properties,
        }
    }
}

/// Record this process's machine class. The first call wins: the hardware
/// does not change under a running process.
pub fn set_serving_hardware(hardware: ServingHardware) {
    let _ = SERVING_HARDWARE.set(hardware);
}

/// This process's machine class, once the hardware probe has finished.
#[must_use]
pub fn serving_hardware() -> Option<&'static ServingHardware> {
    SERVING_HARDWARE.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_names_map_to_backends() {
        assert_eq!(backend_of_device("CUDA0"), Some("cuda"));
        assert_eq!(backend_of_device("MTL0"), Some("metal"));
        assert_eq!(backend_of_device("Vulkan1"), Some("vulkan"));
        assert_eq!(backend_of_device("ROCm0"), Some("rocm"));
        assert_eq!(backend_of_device("CPU"), None);
    }

    #[test]
    fn the_backend_that_served_wins_over_the_survey() {
        let hardware = ServingHardware {
            backend: "cuda",
            gpu_family: Value::from("none"),
            vram_gb: None,
        };
        let properties = hardware.apply(Properties::new(), Some("vulkan"));
        assert_eq!(properties.get("backend"), Some(&Value::Static("vulkan")));
        assert!(properties.get("vram_gb").is_none());
    }
}

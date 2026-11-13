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
    /// GPU memory available to mesh-llm, in whole GiB.
    pub vram_gb: Option<u64>,
}

impl ServingHardware {
    /// Attach these dimensions to `properties`.
    #[must_use]
    pub fn apply(&self, properties: Properties) -> Properties {
        let properties = properties
            .with("backend", self.backend)
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

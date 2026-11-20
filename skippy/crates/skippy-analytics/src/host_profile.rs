//! The `host_profile` event, and the machine class `generation` carries.
//!
//! Reported once per serving process from Skippy's own hardware probe, so a
//! standalone server and an embedding host describe machines the same way.
//! Device names are slugged and reduced to a family, memory is whole GiB, and
//! the probe's stable IDs and bus addresses are never read.

use skippy_native_runtime::{HostGpuProfile, HostRuntimeProfile, NativeRuntimeBackendKind};

use crate::event::{Event, Label, Properties, Value};
use crate::serving_hardware::{ServingHardware, backend_of_device};

/// Probe this machine on a blocking thread, then report `host_profile` and
/// record the machine class for `generation` events. Startup never waits.
pub fn report() {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    runtime.spawn_blocking(|| {
        let profile = skippy_hardware_profile::host_runtime_profile();
        let ram =
            skippy_hardware_profile::model_capacity::host_memory_bytes().map(|(total, _)| total);
        crate::set_serving_hardware(serving_hardware(&profile));
        crate::capture(Event::HostProfile, properties(&profile, ram));
    });
}

/// `host_profile` properties for `profile`, with `ram` total bytes.
#[must_use]
pub fn properties(profile: &HostRuntimeProfile, ram: Option<u64>) -> Properties {
    let flavors = &profile.available_flavors;
    let mut properties = Properties::new()
        .with("gpu_model", gpu_model(&profile.gpus))
        .with("gpu_family", gpu_family_value(&profile.gpus))
        .with("gpu_count", crate::bucket_count(profile.gpus.len() as u64))
        .with(
            "unified_memory",
            profile.gpus.iter().any(|gpu| gpu.unified_memory),
        )
        .with("backend", serving_backend(profile))
        .with(
            "backend_metal",
            flavors.contains(&NativeRuntimeBackendKind::Metal),
        )
        .with(
            "backend_cuda",
            flavors.contains(&NativeRuntimeBackendKind::Cuda),
        )
        .with(
            "backend_rocm",
            flavors.contains(&NativeRuntimeBackendKind::Rocm),
        )
        .with(
            "backend_vulkan",
            flavors.contains(&NativeRuntimeBackendKind::Vulkan),
        );
    if let Some(vram_gb) = accelerator_memory_gb(&profile.gpus) {
        properties = properties.with("vram_gb", vram_gb);
    }
    if let Some(ram) = ram.filter(|bytes| *bytes > 0) {
        properties = properties.with("system_ram_gb", crate::gigabytes(ram));
    }
    properties
}

/// The machine class `generation` events carry.
#[must_use]
pub fn serving_hardware(profile: &HostRuntimeProfile) -> ServingHardware {
    ServingHardware {
        backend: serving_backend(profile),
        gpu_family: gpu_family_value(&profile.gpus),
        vram_gb: accelerator_memory_gb(&profile.gpus),
    }
}

/// The first GPU's name, slugged; `none` with no GPU; `unreported` when a
/// GPU has no usable name.
fn gpu_model(gpus: &[HostGpuProfile]) -> Value {
    match gpus.first() {
        None => Value::from("none"),
        Some(gpu) => {
            Label::slug(&gpu.display_name).map_or_else(|| Value::from("unreported"), Value::from)
        }
    }
}

fn gpu_family_value(gpus: &[HostGpuProfile]) -> Value {
    match gpu_model(gpus) {
        Value::Text(slug) => {
            gpu_family(slug.as_str()).map_or_else(|| Value::from("unreported"), Value::from)
        }
        other => other,
    }
}

/// GPU memory across devices in whole GiB, or `None` with no GPU memory.
fn accelerator_memory_gb(gpus: &[HostGpuProfile]) -> Option<u64> {
    let bytes = gpus
        .iter()
        .filter_map(|gpu| gpu.vram_bytes)
        .fold(0u64, u64::saturating_add);
    (bytes > 0).then(|| crate::gigabytes(bytes))
}

/// The backend this machine serves on as probed: a named device first, then
/// the best runtime flavor it can load. `generation` replaces it with the
/// device a model actually loaded onto when it knows that.
fn serving_backend(profile: &HostRuntimeProfile) -> &'static str {
    if let Some(backend) = profile
        .gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_deref())
        .find_map(backend_of_device)
    {
        return backend;
    }
    [
        (NativeRuntimeBackendKind::Cuda, "cuda"),
        (NativeRuntimeBackendKind::Rocm, "rocm"),
        (NativeRuntimeBackendKind::Metal, "metal"),
        (NativeRuntimeBackendKind::Vulkan, "vulkan"),
    ]
    .into_iter()
    .find(|(kind, _)| profile.available_flavors.contains(kind))
    .map_or("cpu", |(_, name)| name)
}

/// Words in a device slug that name a vendor or a form factor, not a part.
const GPU_FAMILY_NOISE: &[&str] = &[
    "nvidia",
    "geforce",
    "amd",
    "intel",
    "r",
    "tm",
    "gpu",
    "graphics",
    "laptop",
    "generation",
];

/// Reduce a device slug to its family: `apple-m4-max` is `apple-m4`,
/// `nvidia-geforce-rtx-4090-laptop-gpu` is `rtx-4090`, `nvidia-gb10` is
/// `gb10`. At most three segments, so there are fewer families than models.
fn gpu_family(slug: &str) -> Option<Label> {
    let mut segments = slug
        .split('-')
        .skip_while(|segment| segment.chars().all(|c| c.is_ascii_digit()))
        .peekable();
    let first = *segments.peek()?;
    if first == "apple" {
        let family = segments.take(2).collect::<Vec<_>>().join("-");
        return Label::sanitize(&family);
    }
    let mut kept = Vec::new();
    for (index, segment) in segments.enumerate() {
        if index > 0 && segment == first {
            break;
        }
        if !GPU_FAMILY_NOISE.contains(&segment) {
            kept.push(segment);
        }
    }
    kept.truncate(3);
    Label::sanitize(&kept.join("-"))
}

#[cfg(test)]
#[path = "host_profile/tests.rs"]
mod tests;

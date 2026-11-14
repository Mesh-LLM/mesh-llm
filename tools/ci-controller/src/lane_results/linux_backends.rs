//! Static Linux backend calls preserve catalog row and consumer identities.

use super::Checked;
use crate::plan::document::Json;

pub(super) fn grouped_layout(needs: &Json) -> Checked<bool> {
    // In-flight workflows can still report the pre-fan-out dependency names.
    // Identify that graph by its keys, never by whether its jobs passed.
    let grouped = ["native_runtimes", "runtime_product", "product_smoke"]
        .iter()
        .any(|job| needs.get(job).is_some());
    let backend = JOBS.iter().any(|(_, runtime, product, smoke)| {
        [runtime, product, smoke]
            .iter()
            .any(|job| needs.get(job).is_some())
    }) || ["sdk_rust", "sdk_kotlin"]
        .iter()
        .any(|job| needs.get(job).is_some());
    if grouped && backend {
        return Err("mixed Linux job layouts".to_owned());
    }
    Ok(grouped)
}

pub(super) const JOBS: [(&str, &str, &str, &str); 4] = [
    (
        "linux-cpu",
        "native_runtime_cpu",
        "product_cpu",
        "product_smoke_cpu",
    ),
    (
        "linux-cuda",
        "native_runtime_cuda",
        "product_cuda",
        "product_smoke_cuda",
    ),
    (
        "linux-rocm",
        "native_runtime_rocm",
        "product_rocm",
        "product_smoke_rocm",
    ),
    (
        "linux-vulkan",
        "native_runtime_vulkan",
        "product_vulkan",
        "product_smoke_vulkan",
    ),
];

pub(super) fn smoke_selected(row: &str, smoke: &[String]) -> bool {
    smoke.iter().any(|id| match row {
        "linux-cpu" => matches!(
            id.as_str(),
            "core" | "two-node-client" | "two-node-split" | "model-download"
        ),
        "linux-cuda" => id == "core-cuda",
        "linux-rocm" | "linux-vulkan" => id == "core",
        _ => false,
    })
}

pub(super) fn consumes_automation(jobs: &[&str]) -> bool {
    JOBS.iter()
        .any(|(_, runtime, product, _)| jobs.contains(runtime) || jobs.contains(product))
}

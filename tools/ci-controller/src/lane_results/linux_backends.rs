//! Static Linux backend calls preserve catalog row and consumer identities.

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

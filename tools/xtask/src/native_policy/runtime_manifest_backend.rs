use crate::command::DynResult;
use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub(super) enum Backend {
    Cpu,
    Metal,
    Cuda { cuda: Cuda },
    Rocm { rocm: Rocm },
    Vulkan { vulkan: Vulkan },
}

#[derive(Serialize)]
pub(super) struct Cuda {
    toolkit_major: u32,
    gpu_arches: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_driver: Option<String>,
}

#[derive(Serialize)]
pub(super) struct Rocm {
    gpu_arches: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

#[derive(Serialize)]
pub(super) struct Vulkan {
    #[serde(skip_serializing_if = "Option::is_none")]
    min_api_version: Option<String>,
}

pub(super) fn environment(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn arches(primary: &str, fallback: &str, default: &str) -> Vec<String> {
    environment(primary)
        .or_else(|| environment(fallback))
        .unwrap_or_else(|| default.into())
        .split([',', ';'])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(super) fn build(backend: &str, cuda_major: &str) -> DynResult<Backend> {
    Ok(match backend {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal,
        "cuda" | "cuda-blackwell" => Backend::Cuda {
            cuda: Cuda {
                toolkit_major: cuda_major.parse()?,
                gpu_arches: arches(
                    "LLAMA_STAGE_CUDA_ARCHITECTURES",
                    "SKIPPY_CUDA_ARCHITECTURES",
                    if backend == "cuda-blackwell" {
                        "sm_120"
                    } else {
                        ""
                    },
                ),
                min_driver: environment("MESH_LLM_CUDA_MIN_DRIVER"),
            },
        },
        "hip" | "rocm" => Backend::Rocm {
            rocm: Rocm {
                gpu_arches: arches("LLAMA_STAGE_AMDGPU_TARGETS", "SKIPPY_AMDGPU_TARGETS", ""),
                version: environment("MESH_LLM_ROCM_VERSION"),
            },
        },
        "vulkan" => Backend::Vulkan {
            vulkan: Vulkan {
                min_api_version: environment("MESH_LLM_VULKAN_MIN_API_VERSION"),
            },
        },
        _ => return Err(format!("unsupported runtime backend: {backend}").into()),
    })
}

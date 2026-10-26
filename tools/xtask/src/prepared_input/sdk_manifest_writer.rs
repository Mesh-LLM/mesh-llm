use crate::command::DynResult;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct Manifest<'a> {
    schema_version: u32,
    artifact_id: &'a str,
    native_runtime_id: &'a str,
    sdk_version: &'a str,
    mesh_version: &'a str,
    target_triple: &'a str,
    platform: &'a str,
    os: &'a str,
    arch: &'a str,
    backend: &'a str,
    flavor: &'a str,
    cargo_profile: &'a str,
    library: &'a str,
    library_paths: [&'a str; 1],
    uniffi_library: &'a str,
    library_sha256: String,
    url: Option<String>,
    sha256: Option<String>,
    signature: Option<String>,
    requirements: [String; 0],
    llama_upstream_sha: Option<&'a str>,
    llama_patched_sha: Option<&'a str>,
    llama_patch_digest: Option<&'a str>,
    cuda_architectures: Option<String>,
    amdgpu_targets: Option<String>,
    features: [&'static str; 5],
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        output,
        id,
        version,
        target,
        platform,
        os,
        arch,
        backend,
        flavor,
        profile,
        library,
        uniffi,
        upstream,
        patched,
        patch_digest,
    ] = args
    else {
        return Err("native-sdk-manifest-write MANIFEST ID VERSION TARGET PLATFORM OS ARCH BACKEND FLAVOR PROFILE LIBRARY UNIFFI UPSTREAM PATCHED PATCH_DIGEST".into());
    };
    let root = Path::new(output)
        .parent()
        .ok_or("manifest requires a parent directory")?
        .canonicalize()?;
    crate::native_policy::manifest_identity::verify(target, os, arch)?;
    if ![
        "cpu",
        "metal",
        "cuda",
        "cuda-blackwell",
        "rocm",
        "hip",
        "vulkan",
    ]
    .contains(&backend.as_str())
        || !["release", "debug"].contains(&profile.as_str())
    {
        return Err("unsupported SDK backend or Cargo profile".into());
    }
    if id.is_empty() || version.is_empty() || flavor.is_empty() || platform.is_empty() {
        return Err("SDK identity fields must be non-empty".into());
    }
    let primary = super::sdk_artifact_file::artifact_file(
        &root,
        "library",
        Some(&crate::ci_plan::document::Json::String(library.clone())),
    )
    .map_err(|error| error.0)?;
    let alias = super::sdk_artifact_file::artifact_file(
        &root,
        "uniffi library",
        Some(&crate::ci_plan::document::Json::String(uniffi.clone())),
    )
    .map_err(|error| error.0)?;
    let digest = super::sdk_artifact_file::sha256_file(&primary).map_err(|error| error.0)?;
    if super::sdk_artifact_file::sha256_file(&alias).map_err(|error| error.0)? != digest {
        return Err("UniFFI library must contain identical primary library bytes".into());
    }
    let manifest = Manifest {
        schema_version: 1,
        artifact_id: id,
        native_runtime_id: id,
        sdk_version: version,
        mesh_version: version,
        target_triple: target,
        platform,
        os,
        arch,
        backend,
        flavor,
        cargo_profile: profile,
        library,
        library_paths: [library],
        uniffi_library: uniffi,
        library_sha256: digest,
        url: None,
        sha256: None,
        signature: None,
        requirements: [],
        llama_upstream_sha: optional(upstream),
        llama_patched_sha: optional(patched),
        llama_patch_digest: optional(patch_digest),
        cuda_architectures: environment("LLAMA_STAGE_CUDA_ARCHITECTURES")
            .or_else(|| environment("SKIPPY_CUDA_ARCHITECTURES")),
        amdgpu_targets: environment("LLAMA_STAGE_AMDGPU_TARGETS")
            .or_else(|| environment("SKIPPY_AMDGPU_TARGETS")),
        features: [
            "mesh-inference",
            "model-management",
            "local-serving",
            "chat",
            "responses",
        ],
    };
    let mut bytes = serde_json::to_vec_pretty(&serde_json::to_value(&manifest)?)?;
    bytes.push(b'\n');
    std::fs::write(output, bytes)?;
    Ok(())
}

fn optional(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}
fn environment(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

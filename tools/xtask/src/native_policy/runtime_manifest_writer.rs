use crate::command::DynResult;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

#[path = "runtime_manifest_backend.rs"]
mod backend;
#[path = "runtime_manifest_files.rs"]
mod files;

#[derive(Serialize)]
struct Manifest<'a> {
    runtime: Runtime<'a>,
    build: Build<'a>,
}

#[derive(Serialize)]
struct Runtime<'a> {
    id: &'a str,
    mesh_version: &'a str,
    skippy_abi: &'a str,
    platform: Platform<'a>,
    backend: backend::Backend,
    rank: i64,
    libraries: &'a [String],
    files: BTreeMap<String, String>,
    tools: BTreeMap<String, String>,
    url: Option<String>,
    sha256: Option<String>,
    signature: Option<String>,
}

#[derive(Serialize)]
struct Platform<'a> {
    os: &'a str,
    arch: &'a str,
    target: &'a str,
    min_glibc: Option<String>,
}

#[derive(Serialize)]
struct Build<'a> {
    platform: &'a str,
    backend: &'a str,
    primary_library: &'a str,
    relocatable_libraries: &'a [String],
    library_sha256: &'a str,
    llama_upstream_sha: Option<&'a str>,
    llama_patched_sha: Option<&'a str>,
    llama_patch_digest: Option<&'a str>,
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [
        output,
        id,
        version,
        abi,
        os,
        arch,
        target,
        platform,
        backend,
        cuda_major,
        primary,
        upstream,
        patched,
        patch_digest,
        rest @ ..,
    ] = args
    else {
        return Err("runtime-manifest-write MANIFEST ID VERSION ABI OS ARCH TARGET PLATFORM BACKEND CUDA_MAJOR PRIMARY UPSTREAM PATCHED PATCH_DIGEST LIBRARY... -- TOOL... -- LICENSE... -- RELOCATABLE...".into());
    };
    let groups: Vec<_> = rest.split(|value| value == "--").collect();
    super::manifest_identity::verify(target, os, arch)?;
    if id.is_empty() || version.is_empty() || abi.is_empty() {
        return Err("runtime id, mesh version and Skippy ABI must be non-empty".into());
    }
    if (backend == "metal" && os != "macos")
        || (os == "macos" && !["cpu", "metal"].contains(&backend.as_str()))
    {
        return Err("runtime backend is unsupported on selected OS".into());
    }
    let [libraries, tools, licenses, relocatable] = groups.as_slice() else {
        return Err("expected libraries, tools, licenses and relocatable groups".into());
    };
    if libraries.is_empty() || !libraries.contains(primary) {
        return Err("primary library must be declared in non-empty libraries".into());
    }
    if relocatable.iter().any(|path| !libraries.contains(path)) {
        return Err("relocatable libraries must belong to libraries".into());
    }
    let root = Path::new(output)
        .parent()
        .ok_or("manifest requires a parent directory")?;
    let hashes = files::hashes(root, libraries.iter().chain(licenses.iter()))?;
    let tool_hashes = files::hashes(root, tools.iter())?;
    let primary_sha = hashes.get(primary).ok_or("missing primary checksum")?;
    let manifest = Manifest {
        runtime: Runtime {
            id,
            mesh_version: version,
            skippy_abi: abi,
            platform: Platform {
                os,
                arch,
                target,
                min_glibc: files::glibc_floor(root, os, libraries.iter().chain(tools.iter()))?,
            },
            backend: backend::build(backend, cuda_major)?,
            rank: backend::environment("MESH_LLM_NATIVE_RUNTIME_RANK")
                .map_or(Ok(0), |value| value.parse::<i64>())?,
            libraries,
            files: hashes.clone(),
            tools: tool_hashes,
            url: None,
            sha256: None,
            signature: None,
        },
        build: Build {
            platform,
            backend,
            primary_library: primary,
            relocatable_libraries: if os == "linux" { relocatable } else { &[] },
            library_sha256: primary_sha,
            llama_upstream_sha: optional(upstream),
            llama_patched_sha: optional(patched),
            llama_patch_digest: optional(patch_digest),
        },
    };
    let mut bytes = serde_json::to_vec_pretty(&serde_json::to_value(&manifest)?)?;
    bytes.push(b'\n');
    std::fs::write(output, bytes)?;
    Ok(())
}

fn optional(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

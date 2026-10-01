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
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_version: Option<u32>,
    runtime: Runtime<'a>,
    build: Build<'a>,
}

#[derive(Serialize)]
struct Runtime<'a> {
    id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mesh_version: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    release_version: Option<&'a str>,
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
    let schema_v2 = args
        .first()
        .is_some_and(|argument| argument == "--schema-v2");
    let args = if schema_v2 { &args[1..] } else { args };
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
        return Err("runtime-manifest-write [--schema-v2] MANIFEST ID VERSION ABI OS ARCH TARGET PLATFORM BACKEND CUDA_MAJOR PRIMARY UPSTREAM PATCHED PATCH_DIGEST LIBRARY... -- TOOL... -- LICENSE... -- RELOCATABLE...".into());
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
        schema_version: schema_v2.then_some(2),
        runtime: Runtime {
            id,
            mesh_version: (!schema_v2).then_some(version),
            release_version: schema_v2.then_some(version),
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

#[cfg(test)]
mod schema_tests {
    #[test]
    fn writer_preserves_legacy_mesh_and_independent_schema_two_runtime_identity() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("runtime-schema-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(root.join("lib")).unwrap();
        std::fs::write(root.join("lib/llama.dll"), b"fixture").unwrap();
        for schema_v2 in [false, true] {
            let output = root.join(if schema_v2 { "v2.json" } else { "legacy.json" });
            let mut args: Vec<String> = vec![
                output.to_string_lossy().into_owned(),
                "runtime-id".into(),
                "0.75.1".into(),
                "1.2.3".into(),
                "windows".into(),
                "x86_64".into(),
                "x86_64-pc-windows-msvc".into(),
                "windows".into(),
                "cpu".into(),
                "".into(),
                "lib/llama.dll".into(),
                "".into(),
                "".into(),
                "".into(),
                "lib/llama.dll".into(),
                "--".into(),
                "--".into(),
                "--".into(),
            ];
            if schema_v2 {
                args.insert(0, "--schema-v2".into());
            }
            super::run(&args).unwrap();
            let actual: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
            if schema_v2 {
                assert_eq!(actual["schema_version"], 2);
                assert_eq!(actual["runtime"]["release_version"], "0.75.1");
                assert!(actual["runtime"].get("mesh_version").is_none());
            } else {
                assert!(actual.get("schema_version").is_none());
                assert_eq!(actual["runtime"]["mesh_version"], "0.75.1");
                assert!(actual["runtime"].get("release_version").is_none());
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}

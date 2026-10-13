//! Fixed external research source admission for the existing source-dependent Rust gates.
use super::{
    hf_certify::admission::{digest, read},
    immutable_sdk_checkout,
};
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const REPOSITORY: &str = "Mesh-LLM/mesh-llm-research";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    schema_version: u32,
    repository: String,
    revision: String,
    manifest_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    repository: String,
    files: BTreeMap<String, String>,
}
fn admit_pin(root: &Path, pin: &Pin) -> DynResult<PathBuf> {
    if pin.schema_version != 1
        || pin.repository != REPOSITORY
        || pin.manifest_sha256.len() != 64
        || !pin.manifest_sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("external research source pin refused".into());
    }
    let root = immutable_sdk_checkout::checkout(root, &pin.revision)?;
    let bytes = read(&root.join("project-inputs.json"), 1048576)?;
    if digest(&bytes) != pin.manifest_sha256 {
        return Err("external research manifest digest refused".into());
    }
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != 1 || manifest.repository != REPOSITORY {
        return Err("external research manifest shape refused".into());
    }
    // The pinned converter source owns vocabulary GGUF fixtures up to 15,776,467 bytes.
    immutable_sdk_checkout::research_files(&root, &manifest.files)?;
    Ok(root)
}

pub(crate) fn run(mesh: &Path, args: &[String]) -> DynResult<()> {
    if args.iter().map(String::as_str).collect::<Vec<_>>() != ["--kind", "root"] {
        return Err("research-source requires --kind root".into());
    }
    let bytes = read(&mesh.join("ci/python-research-source.json"), 65536)?;
    if bytes
        != include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../ci/python-research-source.json"
        ))
    {
        return Err("external research pin differs from compiled controller".into());
    }
    let pin: Pin = serde_json::from_slice(&bytes)?;
    let root = std::env::var_os("MESH_PYTHON_RESEARCH_SOURCE")
        .ok_or("prepare the pinned external research source first")?;
    let root = admit_pin(Path::new(&root), &pin)?;
    use std::io::Write as _;
    writeln!(crate::cli_output::stdout(), "{}", root.display())?;
    Ok(())
}

#[cfg(all(test, unix))]
#[path = "python_research_source/tests.rs"]
mod tests;

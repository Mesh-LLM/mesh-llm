//! Immutable external Python SDK source admission before dependency or SDK execution.
use super::hf_certify::admission::{digest, read};
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
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
    mesh_source: String,
    generator: serde_json::Value,
    files: BTreeMap<String, String>,
}
fn source_path(root: &Path, relative: &str) -> PathBuf {
    Path::new(relative)
        .components()
        .fold(root.to_path_buf(), |mut path, component| {
            path.push(component.as_os_str());
            path
        })
}
pub(crate) struct Source {
    pub root: PathBuf,
    pub pins: Vec<(PathBuf, String)>,
}
#[cfg(all(test, unix))]
use super::immutable_sdk_checkout::git;
fn admit_pin(root: &Path, pin: &Pin) -> DynResult<Source> {
    if !root.is_absolute()
        || pin.schema_version != 1
        || pin.repository != "Mesh-LLM/mesh-llm-python-sdk"
        || pin.revision.len() != 40
        || !pin.revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("external SDK source pin refused".into());
    }
    let root = super::immutable_sdk_checkout::checkout(root, &pin.revision)?;
    let path = root.join("sdk-inputs.json");
    let bytes = read(&path, 1048576)?;
    if digest(&bytes) != pin.manifest_sha256 {
        return Err("external SDK manifest digest refused".into());
    }
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != 1
        || manifest.repository != pin.repository
        || manifest.files.is_empty()
        || manifest.files.len() > 128
        || manifest.mesh_source.is_empty()
        || !manifest.generator.is_object()
    {
        return Err("external SDK manifest shape refused".into());
    }
    let mut pins = vec![(path, pin.manifest_sha256.clone())];
    pins.extend(super::immutable_sdk_checkout::files(
        &root,
        "sdk-inputs.json",
        &manifest.files,
        128,
        4 * 1048576,
    )?);
    Ok(Source { root, pins })
}
pub(crate) fn admit(mesh: &Path) -> DynResult<Source> {
    let record = source_path(mesh, "ci/required-sdk-python/sdk-source.json");
    let bytes = read(&record, 65536)?;
    if bytes
        != include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../ci/required-sdk-python/sdk-source.json"
        ))
    {
        return Err("external SDK pin differs from compiled controller".into());
    }
    let pin: Pin = serde_json::from_slice(&bytes)?;
    let root = std::env::var_os("MESH_PYTHON_SDK_SOURCE")
        .ok_or("prepare the pinned external Python SDK source first")?;
    let mut source = admit_pin(Path::new(&root), &pin)?;
    source.pins.push((record, digest(&bytes)));
    Ok(source)
}
pub(crate) fn run(mesh: &Path, args: &[String]) -> DynResult<()> {
    let values = args.iter().map(String::as_str).collect::<Vec<_>>();
    let ["--kind", kind] = values.as_slice() else {
        return Err("sdk-source requires --kind embedding|compatibility|root".into());
    };
    let source = admit(mesh)?;
    let path = match *kind {
        "embedding" => source_path(&source.root, "ci/canary-python"),
        "compatibility" => source_path(&source.root, "ci/required-sdk-python"),
        "root" => source.root,
        _ => return Err("SDK source kind refused".into()),
    };
    use std::io::Write as _;
    writeln!(crate::cli_output::stdout(), "{}", path.display())?;
    Ok(())
}

#[cfg(all(test, unix))]
#[path = "python_sdk_source/tests.rs"]
mod tests;

#[cfg(all(test, windows))]
#[test]
fn external_sdk_relative_paths_preserve_native_verbatim_root() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("ci").join("canary-python")).unwrap();
    let project = source_path(&root, "ci/canary-python");
    assert_eq!(
        project.canonicalize().unwrap(),
        root.join("ci").join("canary-python")
    );
    assert!(!project.to_str().unwrap().contains('/'));
}

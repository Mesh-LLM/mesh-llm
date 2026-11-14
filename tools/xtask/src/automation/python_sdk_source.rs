//! In-tree Python SDK source admission before dependency or SDK execution.
use super::hf_certify::admission::{digest, read};
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

const SDK_ROOT: &str = "mesh/sdk/python";
const MANIFEST: &str = "sdk-inputs.json";
const MAX_FILES: usize = 128;
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    source_checkpoint: String,
    mesh_source: String,
    generator: serde_json::Value,
    files: BTreeMap<String, String>,
}

pub(crate) struct Source {
    pub root: PathBuf,
    pub pins: Vec<(PathBuf, String)>,
}

fn source_path(root: &Path, relative: &str) -> DynResult<PathBuf> {
    let relative = Path::new(relative);
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err("Python SDK source path refused".into());
    }
    Ok(root.join(relative))
}

fn ignored_build_output(path: &Path) -> bool {
    if path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(".venv" | "__pycache__" | "build" | "dist")
        )
    }) {
        return true;
    }
    matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("pyc" | "dylib" | "so" | "dll")
    )
}

fn collect_source_files(
    root: &Path,
    directory: &Path,
    files: &mut BTreeSet<String>,
) -> DynResult<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if relative == MANIFEST || ignored_build_output(Path::new(&relative)) {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err("Python SDK source symlink refused".into());
        }
        if kind.is_dir() {
            collect_source_files(root, &path, files)?;
        } else if kind.is_file() {
            files.insert(relative);
        } else {
            return Err("Python SDK source entry type refused".into());
        }
    }
    Ok(())
}

fn admit_root(root: &Path) -> DynResult<Source> {
    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err("in-tree Python SDK directory required".into());
    }
    let manifest_path = root.join(MANIFEST);
    let manifest_bytes = read(&manifest_path, 1024 * 1024)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema_version != 1
        || !is_hex(&manifest.source_checkpoint, 40)
        || !is_hex(&manifest.mesh_source, 40)
        || !manifest.generator.is_object()
        || manifest.files.is_empty()
        || manifest.files.len() > MAX_FILES
        || manifest.files.values().any(|hash| !is_hex(hash, 64))
    {
        return Err("in-tree Python SDK manifest shape refused".into());
    }

    let mut actual = BTreeSet::new();
    collect_source_files(&root, &root, &mut actual)?;
    let expected = manifest.files.keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err("in-tree Python SDK source roster refused".into());
    }

    let mut pins = vec![(manifest_path, digest(&manifest_bytes))];
    for (relative, expected_hash) in manifest.files {
        let path = source_path(&root, &relative)?;
        if path.canonicalize()? != path {
            return Err("in-tree Python SDK source path substitution refused".into());
        }
        let bytes = read(&path, MAX_FILE_BYTES)?;
        if digest(&bytes) != expected_hash {
            return Err(format!("in-tree Python SDK source checksum refused: {relative}").into());
        }
        pins.push((path, expected_hash));
    }
    Ok(Source { root, pins })
}

fn is_hex(value: &str, size: usize) -> bool {
    value.len() == size && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn admit(mesh: &Path) -> DynResult<Source> {
    if !mesh.is_absolute() {
        return Err("Mesh repository root must be absolute".into());
    }
    admit_root(&mesh.join(SDK_ROOT))
}

pub(crate) fn run(mesh: &Path, args: &[String]) -> DynResult<()> {
    let values = args.iter().map(String::as_str).collect::<Vec<_>>();
    let ["--kind", kind] = values.as_slice() else {
        return Err("sdk-source requires --kind embedding|compatibility|root".into());
    };
    let source = admit(mesh)?;
    let path = match *kind {
        "embedding" => source_path(&source.root, "ci/canary-python")?,
        "compatibility" => source_path(&source.root, "ci/required-sdk-python")?,
        "root" => source.root,
        _ => return Err("SDK source kind refused".into()),
    };
    use std::io::Write as _;
    writeln!(crate::cli_output::stdout(), "{}", path.display())?;
    Ok(())
}

#[cfg(test)]
#[path = "python_sdk_source/tests.rs"]
mod tests;

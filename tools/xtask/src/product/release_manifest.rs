//! `product runtime-release-manifest <out> <repo> <tag> <tmp-root>
//! <archive>...`: the port of the inline Python in
//! `scripts/generate-native-runtime-release-manifest.sh`. Each archive is
//! hashed, safely extracted below `<tmp-root>/archive-<index>`, and must
//! hold exactly one `manifest.json` whose runtime carries every required
//! field, targets the tag's version, and agrees with the other archives on
//! MeshLLM version and Skippy ABI. The sorted artifact list is written as
//! `json.dump(..., indent=2, sort_keys=True)` plus a newline.
//!
//! A `SystemExit` message prints as-is and an uncaught exception prints its
//! traceback's last line; both exit 1. Nothing is ever rebuilt: a missing
//! or unsafe archive fails.

use super::digest::file_sha256;
use super::manifest_load::load;
use super::posix_path::{abspath, dirname, join};
use super::pure_path::PurePath;
use super::python_object::{display, dumps, equal, is_dict, type_name};
use super::release_manifest_order::{manifest_paths, sort_by_id};
use crate::artifact::tar_extract::safe_extract;
use crate::artifact::zip_extract::os_error_line;
use crate::ci_plan::document::Json;
use crate::repository::check_report::CheckReport;
use std::path::Path;

const REQUIRED: [&str; 7] = [
    "backend",
    "files",
    "id",
    "libraries",
    "mesh_version",
    "platform",
    "skippy_abi",
];

/// The streams a failing run leaves: extractor output, then the exit line.
struct Failure {
    stderr: String,
}

impl From<String> for Failure {
    fn from(line: String) -> Self {
        Self { stderr: line }
    }
}

pub(super) fn run(args: &[String]) -> CheckReport {
    match generate(args) {
        Ok(()) => CheckReport::default(),
        Err(failure) => CheckReport::failure(String::new(), format!("{}\n", failure.stderr)),
    }
}

/// The versions every archive must share, fixed by the first archive.
#[derive(Default)]
struct Shared {
    mesh_version: Option<String>,
    skippy_abi: Option<Json>,
}

fn generate(args: &[String]) -> Result<(), Failure> {
    let [out, repo, tag, tmp_root, archives @ ..] = args else {
        return Err(format!(
            "ValueError: not enough values to unpack (expected at least 5, got {})",
            args.len() + 1
        )
        .into());
    };
    let release_version = tag.strip_prefix('v').unwrap_or(tag);
    if release_version.is_empty() {
        return Err("release tag must contain a version".to_owned().into());
    }
    let mut shared = Shared::default();
    let mut artifacts = Vec::with_capacity(archives.len());
    for (index, archive) in archives.iter().enumerate() {
        let archive = abspath(archive)?;
        let sha256 = file_sha256(Path::new(&archive))
            .map_err(|failure| os_error_line(&failure.error, &archive))?;
        let runtime = inspect(&archive, index, tmp_root, tag, release_version, &mut shared)?;
        let name = archive.rsplit('/').next().unwrap_or("");
        let mut artifact = runtime;
        set(
            &mut artifact,
            "url",
            format!("https://github.com/{repo}/releases/download/{tag}/{name}"),
        );
        set(&mut artifact, "sha256", sha256);
        artifacts.push(artifact);
    }
    let Some(mesh_version) = shared.mesh_version else {
        return Err("no native runtime artifacts supplied".to_owned().into());
    };
    sort_by_id(&mut artifacts)?;
    let manifest = Json::Object(vec![
        ("mesh_version".to_owned(), Json::String(mesh_version)),
        (
            "skippy_abi".to_owned(),
            shared.skippy_abi.unwrap_or(Json::Null),
        ),
        ("artifacts".to_owned(), Json::Array(artifacts)),
    ]);
    write(out, &format!("{}\n", dumps(&manifest)))?;
    Ok(())
}

/// `artifact[key] = value` on a dict copy.
fn set(artifact: &mut Json, key: &str, value: String) {
    if let Json::Object(entries) = artifact {
        match entries.iter_mut().find(|(name, _)| name == key) {
            Some(slot) => slot.1 = Json::String(value),
            None => entries.push((key.to_owned(), Json::String(value))),
        }
    }
}

/// Extract one archive and return its validated runtime object.
fn inspect(
    archive: &str,
    index: usize,
    tmp_root: &str,
    tag: &str,
    release_version: &str,
    shared: &mut Shared,
) -> Result<Json, Failure> {
    let extract_dir = join(tmp_root, &format!("archive-{index}"));
    if let Err(message) = safe_extract(archive, &PurePath::new(&extract_dir).display()) {
        return Err(Failure {
            stderr: format!(
                "unsafe or invalid tar archive: {message}\n\
                 unsafe or invalid native runtime archive: {archive}"
            ),
        });
    }
    let found = manifest_paths(&extract_dir);
    let [manifest_path] = found.as_slice() else {
        return Err(format!(
            "expected exactly one manifest.json in {archive}, found {}",
            found.len()
        )
        .into());
    };
    let manifest = load(Path::new(manifest_path), manifest_path)?;
    let runtime = runtime_object(&manifest, archive)?;
    let version = runtime_version(&runtime, archive, tag, release_version)?;
    agree(shared, &runtime, version)?;
    Ok(runtime)
}

/// `manifest.get("runtime")` must be a dict holding every required field.
fn runtime_object(manifest: &Json, archive: &str) -> Result<Json, String> {
    if !is_dict(manifest) {
        return Err(format!(
            "AttributeError: '{}' object has no attribute 'get'",
            type_name(manifest)
        ));
    }
    let runtime = manifest.get("runtime").filter(|value| is_dict(value));
    let Some(runtime) = runtime else {
        return Err(format!("{archive} is missing runtime manifest"));
    };
    let missing: Vec<&str> = REQUIRED
        .into_iter()
        .filter(|key| runtime.get(key).is_none())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{archive} is missing native runtime field(s): {}",
            missing.join(", ")
        ));
    }
    Ok(runtime.clone())
}

fn runtime_version(
    runtime: &Json,
    archive: &str,
    tag: &str,
    release_version: &str,
) -> Result<String, String> {
    let value = runtime.get("mesh_version").unwrap_or(&Json::Null);
    let Json::String(version) = value else {
        return Err(format!(
            "AttributeError: '{}' object has no attribute 'startswith'",
            type_name(value)
        ));
    };
    if version.strip_prefix('v').unwrap_or(version) != release_version {
        return Err(format!(
            "{archive} mesh_version {version} does not match release tag {tag}"
        ));
    }
    Ok(version.clone())
}

fn agree(shared: &mut Shared, runtime: &Json, version: String) -> Result<(), String> {
    match &shared.mesh_version {
        None => shared.mesh_version = Some(version),
        Some(first) if *first != version => {
            return Err(format!(
                "mixed mesh versions in native runtime artifacts: {version} != {first}"
            ));
        }
        Some(_) => {}
    }
    let abi = runtime.get("skippy_abi").unwrap_or(&Json::Null);
    match &shared.skippy_abi {
        None => shared.skippy_abi = Some(abi.clone()),
        Some(first) if !equal(abi, first) => {
            return Err(format!(
                "mixed Skippy ABI versions in native runtime artifacts: {} != {}",
                display(abi),
                display(first)
            ));
        }
        Some(_) => {}
    }
    Ok(())
}

/// `os.makedirs(dirname(abspath(out)), exist_ok=True)` then the write.
fn write(out: &str, text: &str) -> Result<(), String> {
    let parent = dirname(&abspath(out)?);
    if let Err(error) = std::fs::create_dir_all(&parent) {
        let exists = Path::new(&parent).exists();
        let error = if exists {
            std::io::Error::from_raw_os_error(17)
        } else {
            error
        };
        return Err(os_error_line(&error, &parent));
    }
    std::fs::write(out, text).map_err(|error| os_error_line(&error, out))
}

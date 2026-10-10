//! `product runtime-release-manifest <out> <repo> <tag> <runtime-version> <tmp-root>
//! <archive>...`: the port of the inline Python in
//! `scripts/generate-native-runtime-release-manifest.sh`. Each archive is
//! hashed, safely extracted below `<tmp-root>/archive-<index>`, and must
//! hold exactly one `manifest.json` whose runtime carries every required
//! field, matches the requested independent runtime release, and agrees with
//! the other archives on runtime release and Skippy ABI. The artifact list is
//! written as sorted, indented JSON plus a newline.
//!
//! Domain validation failures exit 1. Nothing is rebuilt; missing or
//! unsafe archives fail before publication.

use super::digest::file_sha256;
use super::json_object::{display, dumps, equal, is_dict, type_name};
use super::manifest_load::load;
use super::posix_path::{abspath, dirname, join};
use super::pure_path::PurePath;
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
    "release_version",
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
    release_version: Option<String>,
    skippy_abi: Option<Json>,
}

fn generate(args: &[String]) -> Result<(), Failure> {
    let [out, repo, tag, requested_version, tmp_root, archives @ ..] = args else {
        return Err(
            "usage: product runtime-release-manifest <out> <repo> <tag> <runtime-version> <tmp-root> <archive>..."
                .to_owned().into(),
        );
    };
    let release_version = requested_version
        .strip_prefix('v')
        .unwrap_or(requested_version);
    if release_version.is_empty() {
        return Err("requested runtime release must contain a version"
            .to_owned()
            .into());
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
    let Some(release_version) = shared.release_version else {
        return Err("no native runtime artifacts supplied".to_owned().into());
    };
    sort_by_id(&mut artifacts)?;
    let manifest = Json::Object(vec![
        ("schema_version".to_owned(), Json::Number(2.into())),
        ("release_version".to_owned(), Json::String(release_version)),
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
    if !is_dict(&manifest) {
        return Err(format!(
            "{archive} native runtime manifest must be a JSON object, found {}",
            type_name(&manifest)
        )
        .into());
    }
    if manifest.get("schema_version").and_then(Json::as_int) != Some(2) {
        return Err(format!(
            "{archive} requires native runtime schema_version 2; import legacy caches explicitly"
        )
        .into());
    }
    let runtime = runtime_object(&manifest, archive)?;
    let version = runtime_version(&runtime, archive, tag, release_version)?;
    agree(shared, &runtime, version)?;
    Ok(runtime)
}

/// Admit an object manifest whose runtime object carries every required field.
fn runtime_object(manifest: &Json, archive: &str) -> Result<Json, String> {
    if !is_dict(manifest) {
        return Err(format!(
            "{archive} native runtime manifest must be a JSON object, found {}",
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
    _tag: &str,
    release_version: &str,
) -> Result<String, String> {
    let value = runtime.get("release_version").unwrap_or(&Json::Null);
    let Json::String(version) = value else {
        return Err(format!(
            "{archive} native runtime release_version must be a string, found {}",
            type_name(value)
        ));
    };
    if version.strip_prefix('v').unwrap_or(version) != release_version {
        return Err(format!(
            "{archive} release_version {version} does not match requested runtime release {release_version}"
        ));
    }
    Ok(version.clone())
}

fn agree(shared: &mut Shared, runtime: &Json, version: String) -> Result<(), String> {
    match &shared.release_version {
        None => shared.release_version = Some(version),
        Some(first) if *first != version => {
            return Err(format!(
                "mixed runtime releases in native runtime artifacts: {version} != {first}"
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_release_is_independent_of_product_publication_tag() {
        let runtime = Json::parse(br#"{"release_version":"9.0.0"}"#).unwrap();
        assert_eq!(
            runtime_version(&runtime, "producer.tar.gz", "v2.0.0", "9.0.0").unwrap(),
            "9.0.0"
        );
        assert!(
            runtime_version(&runtime, "producer.tar.gz", "v2.0.0", "2.0.0")
                .unwrap_err()
                .contains("requested runtime release")
        );
        let legacy = Json::parse(br#"{"mesh_version":"9.0.0"}"#).unwrap();
        assert!(runtime_version(&legacy, "producer.tar.gz", "v2.0.0", "9.0.0").is_err());
    }
    #[test]
    fn release_manifest_bad_arguments_cannot_create_outputs_or_overwrite_publication() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("release.json");
        let scratch = directory.path().join("extraction");
        let arguments = [
            output.to_string_lossy().into_owned(),
            "Fixture/runtime".into(),
            "v0.68.0".into(),
            "9.0.0".into(),
            scratch.to_string_lossy().into_owned(),
        ];
        for count in 0..=arguments.len() {
            let report = run(&arguments[..count]);
            assert_ne!(report.code, 0);
            assert!(report.stdout.is_empty());
            assert!(!output.exists());
            assert!(!scratch.exists());
            if count < arguments.len() {
                assert!(
                    report
                        .stderr
                        .contains("usage: product runtime-release-manifest")
                );
            }
            std::fs::write(&output, b"previous immutable release publication").unwrap();
            let report = run(&arguments[..count]);
            assert_ne!(report.code, 0);
            assert!(report.stdout.is_empty());
            assert_eq!(
                std::fs::read(&output).unwrap(),
                b"previous immutable release publication"
            );
            assert!(!scratch.exists());
            std::fs::remove_file(&output).unwrap();
        }
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }
    fn archive_with_manifest(
        directory: &Path,
        index: usize,
        manifest: &serde_json::Value,
    ) -> std::path::PathBuf {
        let source = directory.join(format!("source-{index}.json"));
        std::fs::write(&source, serde_json::to_vec(manifest).unwrap()).unwrap();
        let archive = directory.join(format!("native-{index}.tar.gz"));
        super::super::archive_tar::write(
            std::fs::File::create(&archive).unwrap(),
            &[super::super::archive::Entry {
                name: "runtime/manifest.json".into(),
                path: source.clone(),
                directory: false,
                size: std::fs::metadata(&source).unwrap().len(),
                mode: 0o644,
                sha256: Some(
                    file_sha256(&source)
                        .map_err(|failure| failure.error)
                        .unwrap(),
                ),
            }],
        )
        .unwrap();
        archive
    }
    fn valid_manifest() -> serde_json::Value {
        serde_json::json!({"schema_version":2,"runtime":{"backend":{},"files":[],"id":"native-fixture","libraries":[],"release_version":"v9.0.0","platform":{},"skippy_abi":7}})
    }
    fn schema_failure_preserves_publication(manifest: &serde_json::Value, diagnostic: &str) {
        for existing in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let output = directory.path().join("publication/release.json");
            if existing {
                std::fs::create_dir_all(output.parent().unwrap()).unwrap();
                std::fs::write(&output, b"previous immutable release publication").unwrap();
            }
            // Admit a complete valid first archive before failing the second: no
            // partially accumulated artifact list can publish or replace output.
            let first = archive_with_manifest(directory.path(), 0, &valid_manifest());
            let bad = archive_with_manifest(directory.path(), 1, manifest);
            let arguments = [
                output.to_string_lossy().into_owned(),
                "Fixture/runtime".into(),
                "v1.2.3".into(),
                "9.0.0".into(),
                directory
                    .path()
                    .join("extract")
                    .to_string_lossy()
                    .into_owned(),
                first.to_string_lossy().into_owned(),
                bad.to_string_lossy().into_owned(),
            ];
            let report = run(&arguments);
            assert_ne!(report.code, 0);
            assert!(report.stdout.is_empty());
            assert!(report.stderr.contains(diagnostic), "{}", report.stderr);
            assert!(
                report.stderr.contains("native-1.tar.gz"),
                "{}",
                report.stderr
            );
            if existing {
                assert_eq!(
                    std::fs::read(&output).unwrap(),
                    b"previous immutable release publication"
                );
            } else {
                assert!(!output.exists());
                assert!(!output.parent().unwrap().exists());
            }
        }
    }
    #[test]
    fn independent_runtime_catalog_retains_publication_tag_and_schema() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("release.json");
        let archive = archive_with_manifest(directory.path(), 0, &valid_manifest());
        let arguments = [
            output.to_string_lossy().into_owned(),
            "Fixture/runtime".into(),
            "v1.2.3".into(),
            "9.0.0".into(),
            directory
                .path()
                .join("extract")
                .to_string_lossy()
                .into_owned(),
            archive.to_string_lossy().into_owned(),
        ];
        let report = run(&arguments);
        assert_eq!(report.code, 0, "{}", report.stderr);
        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert_eq!(document["schema_version"], 2);
        assert_eq!(document["release_version"], "v9.0.0");
        assert!(
            document["artifacts"][0]["url"]
                .as_str()
                .unwrap()
                .contains("/v1.2.3/native-0.tar.gz")
        );
    }
    #[test]
    fn legacy_and_noninteger_schemas_preserve_existing_publication() {
        for schema in [
            serde_json::json!(1),
            serde_json::json!(true),
            serde_json::json!(2.0),
            serde_json::Value::Null,
        ] {
            let mut manifest = valid_manifest();
            manifest["schema_version"] = schema;
            schema_failure_preserves_publication(
                &manifest,
                "requires native runtime schema_version 2",
            );
        }
    }
    #[test]
    fn malformed_native_runtime_manifest_root_rejects_before_release_publication() {
        for manifest in [
            serde_json::Value::Null,
            serde_json::json!(false),
            serde_json::json!(42),
            serde_json::json!("not an object"),
            serde_json::json!([]),
        ] {
            schema_failure_preserves_publication(
                &manifest,
                "native runtime manifest must be a JSON object",
            );
        }
    }
    #[test]
    fn malformed_native_runtime_version_rejects_before_release_publication() {
        for version in [
            serde_json::Value::Null,
            serde_json::json!(false),
            serde_json::json!(42),
            serde_json::json!([]),
            serde_json::json!({}),
        ] {
            let mut manifest = valid_manifest();
            manifest["runtime"]["release_version"] = version;
            schema_failure_preserves_publication(
                &manifest,
                "native runtime release_version must be a string",
            );
        }
    }
}

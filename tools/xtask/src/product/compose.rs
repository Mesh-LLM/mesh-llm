//! `product compose`: the port of `scripts/compose-product-bundle.py`.
//!
//! Validation and digests run in the legacy order, so the first failure is
//! the one Python would have raised. Every failure is an uncaught exception
//! in the legacy script; the port prints its traceback's last line and
//! exits with status 1. Success prints nothing and writes (or with
//! `--check`, compares) `product-manifest.json`.

use super::compose_argv::{Args, parse};
use super::digest::{IoFailure, file_sha256, tree_sha256};
use super::manifest_load::load;
use super::pure_path::PurePath;
use super::python_object::{display, dumps, equal, get, item, require_hashable};
use crate::artifact::zip_extract::os_error_line;
use crate::ci_plan::document::Json;
use crate::repository::check_report::CheckReport;
use std::path::Path;

const MANIFEST: &str = "product-manifest.json";

pub(super) fn run(args: &[String]) -> CheckReport {
    let args = match parse(args) {
        Ok(args) => args,
        Err(report) => return report,
    };
    match compose(&args) {
        Ok(()) => CheckReport::default(),
        Err(line) => CheckReport::failure(String::new(), format!("{line}\n")),
    }
}

fn io(failure: IoFailure) -> String {
    let shown = failure.path.to_string_lossy().into_owned();
    os_error_line(&failure.error, &shown)
}

fn value_error(message: String) -> String {
    format!("ValueError: {message}")
}

/// `BACKEND_KIND_ALIASES.get(backend, backend)` for a string backend.
fn expected_kind(backend: &str) -> &str {
    match backend {
        "cuda-blackwell" => "cuda",
        "hip" => "rocm",
        other => other,
    }
}

/// The same lookup for a parsed value: only the two alias strings map.
fn expected_kind_of(value: &Json) -> Result<Json, String> {
    require_hashable(value)?;
    Ok(match value {
        Json::String(text) => Json::String(expected_kind(text).to_owned()),
        other => other.clone(),
    })
}

fn compose(args: &Args) -> Result<(), String> {
    let bundle = PurePath::new(&args.bundle);
    let manifest = compose_manifest(args, &bundle)?;
    let manifest_path = bundle.join(MANIFEST).display();
    if args.check {
        let existing = load(Path::new(&manifest_path), &manifest_path)?;
        if !equal(&existing, &manifest) {
            return Err(value_error(format!(
                "product manifest does not match composed bytes: {manifest_path}"
            )));
        }
        return Ok(());
    }
    let bytes = format!("{}\n", dumps(&manifest));
    std::fs::write(&manifest_path, bytes).map_err(|error| os_error_line(&error, &manifest_path))
}

fn compose_manifest(args: &Args, bundle: &PurePath) -> Result<Json, String> {
    let version = args.version.strip_prefix('v').unwrap_or(&args.version);
    let host = PurePath::new(&args.host);
    let runtime = PurePath::new(&args.runtime);
    let runtime_manifest_path = runtime.join("manifest.json").display();
    let runtime_manifest = load(Path::new(&runtime_manifest_path), &runtime_manifest_path)?;
    if runtime_manifest
        .get("schema_version")
        .and_then(Json::as_int)
        != Some(2)
    {
        return Err(value_error(
            "native runtime manifest requires schema_version 2; import legacy caches explicitly"
                .to_owned(),
        ));
    }
    let runtime_data = item(&runtime_manifest, "runtime")?;
    let runtime_id = item(runtime_data, "id")?;
    validate_backend(runtime_id, &runtime_manifest, runtime_data, &args.backend)?;
    let contract = super::host_contract::read(Path::new(&host.display()))?;
    super::host_contract::validate(&contract, runtime_data, runtime_id, version)?;
    let host_path = host.relative_to(bundle).map_err(value_error)?;
    let host_sha = file_sha256(Path::new(&host.display())).map_err(io)?;
    let runtime_path = runtime.relative_to(bundle).map_err(value_error)?;
    let runtime_sha = tree_sha256(Path::new(&runtime.display())).map_err(io)?;
    let manifest_sha = file_sha256(Path::new(&runtime_manifest_path)).map_err(io)?;
    let text = |value: &str| Json::String(value.to_owned());
    Ok(Json::Object(vec![
        ("schema_version".to_owned(), Json::Number(2.into())),
        ("contract".to_owned(), text("mesh-llm-product-v2")),
        ("mesh_version".to_owned(), text(version)),
        ("backend".to_owned(), text(&args.backend)),
        (
            "host".to_owned(),
            Json::Object(vec![
                ("path".to_owned(), text(&host_path)),
                ("sha256".to_owned(), text(&host_sha)),
                (
                    "required_skippy_abi".to_owned(),
                    item(&contract, "skippy_abi")?.clone(),
                ),
            ]),
        ),
        (
            "runtime".to_owned(),
            Json::Object(vec![
                ("id".to_owned(), runtime_id.clone()),
                (
                    "release_version".to_owned(),
                    item(runtime_data, "release_version")?.clone(),
                ),
                (
                    "skippy_abi".to_owned(),
                    item(runtime_data, "skippy_abi")?.clone(),
                ),
                ("path".to_owned(), text(&runtime_path)),
                ("sha256".to_owned(), text(&runtime_sha)),
                ("manifest_sha256".to_owned(), text(&manifest_sha)),
            ]),
        ),
    ]))
}

/// `validate_runtime_backend`: the runtime family, then any build backend,
/// must match the requested backend's family.
fn validate_backend(
    runtime_id: &Json,
    runtime_manifest: &Json,
    runtime_data: &Json,
    requested: &str,
) -> Result<(), String> {
    let expected = expected_kind(requested);
    let runtime_kind = item(item(runtime_data, "backend")?, "kind")?;
    if runtime_kind.as_str() != Some(expected) {
        return Err(value_error(format!(
            "native runtime {} backend mismatch: found {}, expected {expected} for requested \
             backend {requested}",
            display(runtime_id),
            display(runtime_kind)
        )));
    }
    let Some(build) = get(runtime_manifest, "build").filter(|value| **value != Json::Null) else {
        return Ok(());
    };
    let build_backend = item(build, "backend")?;
    let build_kind = expected_kind_of(build_backend)?;
    if build_kind.as_str() != Some(expected) {
        return Err(value_error(format!(
            "native runtime {} build backend mismatch: found {}, expected runtime family \
             {expected} for requested backend {requested}",
            display(runtime_id),
            display(build_backend)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_runtime_rejected_before_product_write() {
        let temp = tempfile::tempdir().unwrap();
        let runtime = temp.path().join("runtime");
        std::fs::create_dir(&runtime).unwrap();
        std::fs::write(
            runtime.join("manifest.json"),
            br#"{"schema_version":1,"runtime":{"mesh_version":"2.0.0"}}"#,
        )
        .unwrap();
        let args = [
            "--bundle",
            temp.path().to_str().unwrap(),
            "--host",
            "nonexistent-host",
            "--runtime",
            runtime.to_str().unwrap(),
            "--version",
            "2.0.0",
            "--backend",
            "cpu",
        ]
        .map(str::to_owned);
        let args = parse(&args).unwrap_or_else(|_| panic!("valid arguments"));
        assert!(compose(&args).unwrap_err().contains("schema_version 2"));
        assert!(!temp.path().join(MANIFEST).exists());
    }

    #[test]
    fn migration_product_backend_aliases() {
        assert_eq!(expected_kind("cuda-blackwell"), "cuda");
        assert_eq!(expected_kind("hip"), "rocm");
        assert_eq!(expected_kind("metal"), "metal");
        let list = Json::Array(Vec::new());
        assert_eq!(
            expected_kind_of(&list).err().as_deref(),
            Some("backend must be a scalar value")
        );
    }
}

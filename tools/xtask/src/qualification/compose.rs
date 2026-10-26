//! `qualify compose`: compose the exact standalone CLI and native runtime CI
//! produced into a qualification product. The composer verifies and copies
//! exact producer bytes; it never compiles, links, re-stamps, or substitutes
//! an input.

use super::cli_input;
use super::contract::{self, Json};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

const USAGE: &str = "qualify compose --cli-input <dir> --runtime-input <dir> --output <dir> --source-sha <sha> --target <target> --backend <backend>";

const TARGETS: [(&str, &str, &str); 4] = [
    ("x86_64-unknown-linux-gnu", "linux", "x86_64"),
    ("aarch64-unknown-linux-gnu", "linux", "aarch64"),
    ("aarch64-apple-darwin", "macos", "aarch64"),
    ("x86_64-pc-windows-msvc", "windows", "x86_64"),
];

const BACKENDS: [&str; 5] = ["cpu", "cuda", "rocm", "vulkan", "metal"];

pub(crate) fn target_platform(target: &str) -> Option<(&'static str, &'static str)> {
    TARGETS
        .iter()
        .find(|(name, ..)| *name == target)
        .map(|(_, os, arch)| (*os, *arch))
}

/// `validate_pair`: the runtime manifest and CLI build contract must describe
/// the selected row's target and backend, the selected source, and a matching
/// release/ABI pair.
pub(crate) fn validate_pair(
    cli_contract: &Json,
    runtime_manifest: &Json,
    source_sha: &str,
    target: &str,
    backend: &str,
) -> Result<Json, String> {
    let (expected_os, expected_arch) = target_platform(target)
        .ok_or_else(|| format!("unsupported standalone target: {target}"))?;
    contract::require(
        BACKENDS.contains(&backend),
        &format!("unsupported standalone backend: {backend}"),
    )?;
    contract::require(
        runtime_manifest
            .get("schema_version")
            .and_then(Json::as_int)
            == Some(2),
        "native runtime manifest must use schema 2",
    )?;
    let runtime = runtime_manifest
        .get("runtime")
        .filter(|runtime| runtime.as_object().is_some())
        .ok_or("native runtime manifest is missing runtime")?;
    let platform = runtime.get("platform");
    let selected = runtime.get("backend");
    contract::require(
        platform
            .and_then(|platform| platform.get("target"))
            .and_then(Json::as_str)
            == Some(target),
        "native runtime target differs from selected row",
    )?;
    contract::require(
        selected
            .and_then(|backend| backend.get("kind"))
            .and_then(Json::as_str)
            == Some(backend),
        "native runtime backend differs from selected row",
    )?;
    contract::require(
        platform
            .and_then(|platform| platform.get("os"))
            .and_then(Json::as_str)
            == Some(expected_os)
            && platform
                .and_then(|platform| platform.get("arch"))
                .and_then(Json::as_str)
                == Some(expected_arch),
        "native runtime OS or architecture differs from selected row",
    )?;
    contract::require(
        cli_contract.get("source_sha").and_then(Json::as_str) == Some(source_sha),
        "Skippy CLI source differs from selected source",
    )?;
    contract::require(
        cli_contract.get("os").and_then(Json::as_str) == Some(expected_os)
            && cli_contract.get("architecture").and_then(Json::as_str) == Some(expected_arch),
        "Skippy CLI target differs from native runtime target",
    )?;
    contract::require(
        cli_contract.get("runtime_release").and_then(Json::as_str)
            == runtime.get("release_version").and_then(Json::as_str),
        "native runtime release differs from Skippy CLI requirement",
    )?;
    contract::require(
        cli_contract.get("skippy_abi").and_then(Json::as_str)
            == runtime.get("skippy_abi").and_then(Json::as_str),
        "native runtime ABI differs from Skippy CLI requirement",
    )?;
    if let Some(build) = runtime_manifest
        .get("build")
        .filter(|build| build.as_object().is_some())
    {
        let accepted = if backend == "rocm" {
            &[backend, "hip"][..]
        } else {
            &[backend][..]
        };
        contract::require(
            build
                .get("backend")
                .and_then(Json::as_str)
                .is_some_and(|build_backend| accepted.contains(&build_backend)),
            "native runtime build backend differs from selected row",
        )?;
    }
    Ok(runtime.clone())
}

fn verify_runtime_source(runtime_input: &Path, source_sha: &str) -> Result<(), String> {
    let provenance = contract::load_json(&runtime_input.join("ci-source.json"))?;
    let expected = Json::Object(vec![(
        "source_sha".to_owned(),
        Json::String(source_sha.to_owned()),
    )]);
    contract::require(
        provenance == expected,
        "native runtime producer source differs from selected source",
    )
}

/// `verify_discovery`: the composed CLI must discover the exact extracted
/// runtime through its own runtime list.
pub(crate) fn verify_discovery(
    binary: &Path,
    runtime_dir: &Path,
    runtime: &Json,
) -> Result<(), String> {
    let cache = tempfile::Builder::new()
        .prefix("skippy-ci-runtime-cache-")
        .tempdir()
        .map_err(|error| error.to_string())?;
    let release_version = runtime
        .get("release_version")
        .and_then(Json::as_str)
        .ok_or("native runtime lacks a release version")?;
    let runtime_id = runtime
        .get("id")
        .and_then(Json::as_str)
        .ok_or("native runtime lacks an id")?;
    let output = super::serve::run_capture(
        binary,
        &[
            "--runtime-bundle".to_owned(),
            runtime_dir.to_string_lossy().into_owned(),
            "--runtime-cache".to_owned(),
            cache.path().to_string_lossy().into_owned(),
            "--runtime-release".to_owned(),
            release_version.to_owned(),
            "--output".to_owned(),
            "json".to_owned(),
            "runtime".to_owned(),
            "list".to_owned(),
            "--installed".to_owned(),
        ],
        std::time::Duration::from_secs(30),
    )?;
    let installed = contract::parse_strict(output.as_bytes())?;
    let entries = installed.as_array().unwrap_or(&[]);
    let discovered = entries.iter().any(|entry| {
        entry.get("native_runtime_id").and_then(Json::as_str) == Some(runtime_id)
            && entry.get("release_version").and_then(Json::as_str) == Some(release_version)
            && entry
                .get("path")
                .and_then(Json::as_str)
                .is_some_and(|path| same_file(Path::new(path), runtime_dir))
    });
    contract::require(
        discovered,
        "composed Skippy CLI did not discover the exact native runtime",
    )
}

fn same_file(left: &Path, right: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        let left = std::fs::metadata(left);
        let right = std::fs::metadata(right);
        match (left, right) {
            (Ok(left), Ok(right)) => left.dev() == right.dev() && left.ino() == right.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        match (left.canonicalize(), right.canonicalize()) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    }
}

fn find_runtime_archive(runtime_input: &Path) -> Result<PathBuf, String> {
    let mut archives = Vec::new();
    let mut sidecars = Vec::new();
    collect_matching(runtime_input, &mut archives, &mut sidecars)?;
    archives.sort();
    sidecars.sort();
    contract::require(
        archives.len() == 1
            && sidecars.len() == 1
            && sidecars[0].as_os_str().to_string_lossy()
                == format!("{}.sha256", archives[0].as_os_str().to_string_lossy()),
        "standalone product requires exactly one runtime archive and sidecar",
    )?;
    Ok(archives.remove(0))
}

fn collect_matching(
    directory: &Path,
    archives: &mut Vec<PathBuf>,
    sidecars: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            collect_matching(&path, archives, sidecars)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("gz")
            && path.to_string_lossy().ends_with(".tar.gz")
        {
            archives.push(path);
        } else if path.to_string_lossy().ends_with(".tar.gz.sha256") {
            sidecars.push(path);
        }
    }
    Ok(())
}

fn verify_runtime_package(arguments: &[String]) -> Result<(), String> {
    let report = crate::native_policy::verify_runtime_package(arguments);
    if report.code == 0 {
        Ok(())
    } else {
        Err(format!(
            "native runtime package verification failed: {}{}",
            report.stderr, report.stdout
        ))
    }
}

fn write_product_archive(output: &Path, archive: &Path) -> Result<(), String> {
    let mut members = Vec::new();
    collect_archive_members(output, output, &mut members)?;
    members.sort();
    let file = std::fs::File::create(archive)
        .map_err(|error| format!("{}: {error}", archive.display()))?;
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut builder = tar::Builder::new(encoder);
    for (relative, path) in members {
        if path.is_dir() {
            builder
                .append_dir(&relative, &path)
                .map_err(|error| error.to_string())?;
        } else {
            builder
                .append_path_with_name(&path, &relative)
                .map_err(|error| error.to_string())?;
        }
    }
    let encoder = builder.into_inner().map_err(|error| error.to_string())?;
    encoder
        .finish()
        .map_err(|error| error.to_string())
        .map(|_| ())
}

fn collect_archive_members(
    root: &Path,
    directory: &Path,
    members: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(directory)
        .map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "product member escapes its root".to_owned())?
            .iter()
            .map(|part| part.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        if path.is_dir() {
            collect_archive_members(root, &path, members)?;
        }
        members.push((relative, path));
    }
    Ok(())
}

fn compose(
    cli_input_dir: &Path,
    runtime_input: &Path,
    output: &Path,
    source_sha: &str,
    target: &str,
    backend: &str,
) -> Result<PathBuf, String> {
    let cli_input_dir = cli_input_dir
        .canonicalize()
        .map_err(|error| format!("{}: {error}", cli_input_dir.display()))?;
    let runtime_input = runtime_input
        .canonicalize()
        .map_err(|error| format!("{}: {error}", runtime_input.display()))?;
    let output_absolute =
        std::path::absolute(output).map_err(|error| format!("{}: {error}", output.display()))?;
    for producer in [&cli_input_dir, &runtime_input] {
        contract::require(
            output_absolute != *producer
                && !output_absolute.starts_with(producer)
                && !producer.starts_with(&output_absolute),
            "standalone product output overlaps a producer input",
        )?;
    }
    let cli_contract = cli_input::verify(&cli_input_dir, source_sha)?;
    verify_runtime_source(&runtime_input, source_sha)?;
    let archive = find_runtime_archive(&runtime_input)?;
    verify_runtime_package(&[archive.to_string_lossy().into_owned()])?;
    std::fs::create_dir_all(&output_absolute)
        .map_err(|error| format!("{}: {error}", output_absolute.display()))?;
    contract::require(
        std::fs::read_dir(&output_absolute)
            .map_err(|error| error.to_string())?
            .next()
            .is_none(),
        "standalone product output must be empty",
    )?;
    let runtime_root = output_absolute.join("native-runtimes");
    std::fs::create_dir(&runtime_root).map_err(|error| error.to_string())?;
    crate::artifact::tar_extract::safe_extract(
        &archive.to_string_lossy(),
        &runtime_root.to_string_lossy(),
    )?;
    let manifests = runtime_manifests(&runtime_root)?;
    contract::require(
        manifests.len() == 1,
        "runtime archive must contain exactly one runtime manifest",
    )?;
    let manifest_path = manifests.into_iter().next().unwrap_or_default();
    let runtime_dir = manifest_path
        .parent()
        .ok_or("runtime manifest has no parent")?
        .to_path_buf();
    verify_runtime_package(&[runtime_dir.to_string_lossy().into_owned()])?;
    let runtime_manifest = contract::load_json(&manifest_path)?;
    let runtime = validate_pair(
        &cli_contract,
        &runtime_manifest,
        source_sha,
        target,
        backend,
    )?;
    let binary_name = if target.starts_with("x86_64-pc-windows") {
        "skippy.exe"
    } else {
        "skippy"
    };
    let binary = output_absolute.join(binary_name);
    copy(&cli_input_dir.join(binary_name), &binary)?;
    copy(
        &cli_input_dir.join("host-imports.json"),
        &output_absolute.join("host-imports.json"),
    )?;
    copy(
        &cli_input_dir.join("build-contract.json"),
        &output_absolute.join("build-contract.json"),
    )?;
    verify_discovery(&binary, &runtime_dir, &runtime)?;
    let product = serde_json::json!({
        "schema_version": 1,
        "contract": "skippy-product-v1",
        "source_sha": source_sha,
        "target": target,
        "backend": backend,
        "cli": {
            "path": binary_name,
            "sha256": contract::digest(&binary)?,
            "build_contract_sha256": contract::digest(&output_absolute.join("build-contract.json"))?,
            "host_imports_sha256": contract::digest(&output_absolute.join("host-imports.json"))?,
        },
        "runtime": {
            "id": runtime.get("id").and_then(Json::as_str).unwrap_or_default(),
            "path": runtime_dir
                .strip_prefix(&output_absolute)
                .map_err(|_| "runtime escapes product".to_owned())?
                .iter()
                .map(|part| part.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/"),
            "sha256": contract::tree_digest(&runtime_dir)?,
            "manifest_sha256": contract::digest(&manifest_path)?,
        },
    });
    let manifest_out = output_absolute.join("product-manifest.json");
    std::fs::write(&manifest_out, contract::render(&product)?)
        .map_err(|error| format!("{}: {error}", manifest_out.display()))?;
    let product_archive = output_absolute.with_extension("tar.gz");
    write_product_archive(&output_absolute, &product_archive)?;
    Ok(product_archive)
}

fn runtime_manifests(runtime_root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut manifests = Vec::new();
    let entries = std::fs::read_dir(runtime_root)
        .map_err(|error| format!("{}: {error}", runtime_root.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        let manifest = path.join("manifest.json");
        if path.is_dir() && manifest.is_file() {
            manifests.push(manifest);
        }
    }
    manifests.sort();
    Ok(manifests)
}

fn copy(source: &Path, destination: &Path) -> Result<(), String> {
    std::fs::copy(source, destination)
        .map(|_| ())
        .map_err(|error| format!("{} -> {}: {error}", source.display(), destination.display()))
}

pub(super) fn run(_root: &Path, args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--cli-input",
            "--runtime-input",
            "--output",
            "--source-sha",
            "--target",
            "--backend",
        ],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| parsed.last(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() || !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    match compose(
        Path::new(value("--cli-input")),
        Path::new(value("--runtime-input")),
        Path::new(value("--output")),
        value("--source-sha"),
        value("--target"),
        value("--backend"),
    ) {
        Ok(archive) => CheckReport::success(format!("{}\n", archive.display())),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Skippy product composition failed: {error}\n"),
        ),
    }
}

#[cfg(test)]
mod tests;

//! `qualify verify-cli-input`: verify the standalone Skippy CLI producer bytes
//! and embedded build contract. Re-executing the binary proves the recorded
//! contract is the executable's own, not a swapped sidecar.

use super::contract::{self};
use crate::ci_plan::document::Json;
use crate::process::{self, Cancellation, Completion, Limits, RawCaptureOptions, Readiness, Value};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

const USAGE: &str = "qualify verify-cli-input <directory> --source-sha <sha>";

/// Run `binary` with `arguments`, bounding execution and captured output.
fn probe(binary: &Path, arguments: &[&str]) -> Result<String, String> {
    let spec = process::ProcessSpec {
        executable: binary
            .canonicalize()
            .map_err(|error| format!("{}: {error}", binary.display()))?,
        arguments: arguments
            .iter()
            .map(|value| Value::Public((*value).into()))
            .collect(),
        cwd: std::env::current_dir().map_err(|error| error.to_string())?,
        environment: ["PATH", "SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect(),
    };
    let limits = Limits {
        execution: Duration::from_secs(30),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .map_err(|error| format!("Skippy CLI probe failed: {error:?}"))?;
    if !report.process.success() {
        return Err(format!("Skippy CLI probe failed: {:?}", report.process));
    }
    let bytes = report.stdout.ok_or("Skippy CLI probe output missing")?;
    Ok(String::from_utf8_lossy(bytes.as_bytes()).into_owned())
}

/// Verify the CLI producer directory, returning the build contract the
/// executable itself reports.
pub(crate) fn verify(directory: &Path, source_sha: &str) -> Result<Json, String> {
    contract::require(
        contract::is_git_sha(source_sha),
        "source SHA must be a full lowercase Git commit ID",
    )?;
    let names: Vec<&str> = ["skippy", "skippy.exe"]
        .into_iter()
        .filter(|name| directory.join(name).is_file())
        .collect();
    contract::require(
        names.len() == 1,
        "CLI input must contain exactly one Skippy executable",
    )?;
    let name = names[0];
    let binary = directory.join(name);
    let digest = contract::digest(&binary)?;
    let sidecar = std::fs::read_to_string(directory.join(format!("{name}.sha256")))
        .map_err(|error| error.to_string())?;
    contract::require(
        sidecar.trim() == format!("{digest}  {name}"),
        "Skippy CLI checksum sidecar does not match executable",
    )?;
    // Artifact download does not preserve the POSIX executable mode.
    #[cfg(unix)]
    if name == "skippy" {
        use std::os::unix::fs::PermissionsExt as _;
        let metadata = std::fs::metadata(&binary).map_err(|error| error.to_string())?;
        std::fs::set_permissions(
            &binary,
            std::fs::Permissions::from_mode(metadata.permissions().mode() | 0o100),
        )
        .map_err(|error| error.to_string())?;
    }
    let imports = contract::load_json(&directory.join("host-imports.json"))?;
    verify_imports(&imports, name, &digest)?;
    let recorded_raw =
        std::fs::read(directory.join("build-contract.json")).map_err(|error| error.to_string())?;
    let recorded = contract::parse_strict(strip_bom(&recorded_raw))?;
    let actual = contract::parse_strict(probe(&binary, &["build-contract"])?.as_bytes())?;
    contract::require(
        recorded == actual,
        "recorded Skippy build contract differs from executable",
    )?;
    contract::require(
        actual.get("schema_version").and_then(Json::as_int) == Some(1)
            && actual.get("product").and_then(Json::as_str) == Some("skippy"),
        "unsupported Skippy build contract",
    )?;
    contract::require(
        actual.get("source_sha").and_then(Json::as_str) == Some(source_sha),
        "Skippy CLI was built from a different source commit",
    )?;
    let os = actual.get("os").and_then(Json::as_str).unwrap_or("");
    let expected_format = match os {
        "linux" => Some("elf"),
        "macos" => Some("macho"),
        "windows" => Some("pe"),
        _ => None,
    };
    let format = imports.get("format").and_then(Json::as_str).unwrap_or("");
    contract::require(
        Some(format) == expected_format
            && name.ends_with(".exe") == (expected_format == Some("pe")),
        "Skippy CLI import format differs from executable target",
    )?;
    for field in [
        "product_version",
        "runtime_release",
        "skippy_abi",
        "os",
        "architecture",
    ] {
        contract::require(
            actual
                .get(field)
                .and_then(Json::as_str)
                .is_some_and(|value| !value.is_empty()),
            &format!("Skippy build contract is missing {field}"),
        )?;
    }
    let version = probe(&binary, &["--version"])?;
    let product_version = actual
        .get("product_version")
        .and_then(Json::as_str)
        .unwrap_or("");
    contract::require(
        version.trim() == format!("skippy {product_version}"),
        "Skippy CLI version differs from embedded build contract",
    )?;
    Ok(actual)
}

fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(bytes)
}

fn verify_imports(imports: &Json, name: &str, digest: &str) -> Result<(), String> {
    let valid = imports.get("binary").and_then(Json::as_str) == Some(name)
        && imports.get("binary_sha256").and_then(Json::as_str) == Some(digest)
        && imports.get("policy").and_then(Json::as_str) == Some("mesh-llm-dynamic-host-v2")
        && imports
            .get("rejected_imports")
            .and_then(Json::as_array)
            .is_some_and(|rejected| rejected.is_empty())
        && imports
            .get("imports")
            .and_then(Json::as_array)
            .is_some_and(|items| items.iter().all(|item| item.as_str().is_some()))
        && imports
            .get("format")
            .and_then(Json::as_str)
            .is_some_and(|format| matches!(format, "elf" | "macho" | "pe"));
    contract::require(
        valid,
        "Skippy CLI import-policy report is missing, foreign, or rejected",
    )
}

pub(super) fn run(args: &[String]) -> CheckReport {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &["--source-sha"],
        flags: &[],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let [directory] = parsed.positionals.as_slice() else {
        return GRAMMAR.error("expected exactly one CLI input directory");
    };
    let Some(source_sha) = parsed.last("--source-sha") else {
        return GRAMMAR.error("the following arguments are required: --source-sha");
    };
    match verify(Path::new(directory), source_sha) {
        Ok(_) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Skippy CLI input verification failed: {error}\n"),
        ),
    }
}

#[cfg(test)]
mod tests;

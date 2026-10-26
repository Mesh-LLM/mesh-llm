//! `release await-packaging`: require the terminal, identity-bound
//! mesh-packaging release receipt. Polls `gh api` for the correlated
//! `images-release.yml` repository-dispatch run, downloads its single
//! `packaging-readiness` artifact, and validates the receipt against the run
//! and the release identities before writing it.

use crate::process::{self, Cancellation, Completion, Limits, RawCaptureOptions, Readiness, Value};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

const USAGE: &str = "release await-packaging --repository <owner/name> --correlation-id <id> --upstream-repository <owner/name> --upstream-ref <ref> --upstream-sha <40hex> --manifest-sha256 <64hex> --out <file> [--timeout-seconds 10800]";

const REPOSITORY: &str = "Mesh-LLM/mesh-packaging";
const UPSTREAM_REPOSITORY: &str = "Mesh-LLM/mesh-llm";
const WORKFLOW: &str = "images-release.yml";
const ARTIFACT_NAME: &str = "packaging-readiness";
const MEMBER: &str = "packaging-readiness.json";
const POLL_INTERVAL: Duration = Duration::from_secs(30);
const GH_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024;
const MAX_PAGES: u32 = 5;

pub(crate) struct Options {
    correlation_id: String,
    upstream_ref: String,
    upstream_sha: String,
    manifest_sha256: String,
    out: std::path::PathBuf,
    timeout: Duration,
}

/// The `gh` executable; tests substitute a fixture through the environment.
fn gh_program() -> OsString {
    std::env::var_os("MESH_LLM_TEST_GH").unwrap_or_else(|| OsString::from("gh"))
}

fn gh_api(path: &str) -> Result<Vec<u8>, String> {
    let spec = process::ProcessSpec {
        executable: gh_program().into(),
        arguments: ["api", "--method", "GET", path]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: std::env::current_dir().map_err(|error| error.to_string())?,
        environment: [
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "HOME",
        ]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
        .collect(),
    };
    let limits = Limits {
        execution: GH_TIMEOUT,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 4 * 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(4 * 1024 * 1024),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .map_err(|error| format!("gh api {path} failed: {error:?}"))?;
    if !report.process.success() {
        return Err(format!("gh api {path} failed: {:?}", report.process));
    }
    Ok(report
        .stdout
        .ok_or("gh api output missing")?
        .as_bytes()
        .to_vec())
}

fn json(bytes: &[u8]) -> Result<serde_json::Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid gh api JSON: {error}"))
}

/// Find the unique correlated packaging run, or `None` while it is absent.
fn find_run(options: &Options) -> Result<Option<serde_json::Value>, String> {
    let mut matching = Vec::new();
    for page in 1..=MAX_PAGES {
        let payload = json(&gh_api(&format!(
            "repos/{REPOSITORY}/actions/workflows/{WORKFLOW}/runs?event=repository_dispatch&per_page=100&page={page}"
        ))?)?;
        let runs = payload
            .get("workflow_runs")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        matching.extend(
            runs.iter()
                .filter(|run| {
                    run.get("display_title").and_then(serde_json::Value::as_str)
                        == Some(format!("Packaging · {}", options.correlation_id).as_str())
                })
                .cloned(),
        );
        if runs.len() < 100 {
            break;
        }
        if page == MAX_PAGES {
            return Err("packaging run search exceeded the bounded history window".to_owned());
        }
    }
    if matching.len() > 1 {
        return Err("more than one packaging run has the release correlation ID".to_owned());
    }
    Ok(matching.into_iter().next())
}

fn load_receipt(run: &serde_json::Value) -> Result<serde_json::Value, String> {
    let run_id = run
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .ok_or("packaging run lacks an id")?;
    let mut artifacts = Vec::new();
    for page in 1..=MAX_PAGES {
        let payload = json(&gh_api(&format!(
            "repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        ))?)?;
        let current = payload
            .get("artifacts")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        artifacts.extend(
            current
                .iter()
                .filter(|artifact| {
                    artifact.get("name").and_then(serde_json::Value::as_str) == Some(ARTIFACT_NAME)
                        && artifact.get("expired").and_then(serde_json::Value::as_bool)
                            != Some(true)
                })
                .cloned(),
        );
        if current.len() < 100 {
            break;
        }
        if page == MAX_PAGES {
            return Err(
                "packaging artifact search exceeded the bounded artifact window".to_owned(),
            );
        }
    }
    if artifacts.len() != 1 {
        return Err(format!(
            "packaging run {run_id} has {} readiness artifacts",
            artifacts.len()
        ));
    }
    let artifact = &artifacts[0];
    if artifact
        .get("size_in_bytes")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        > MAX_ARTIFACT_BYTES
    {
        return Err("packaging readiness artifact is unexpectedly large".to_owned());
    }
    let artifact_id = artifact
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .ok_or("packaging readiness artifact lacks an id")?;
    let data = gh_api(&format!(
        "repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
    ))?;
    let bytes = crate::artifact::zip_read_single_file(data, MEMBER)?;
    json(&bytes)
}

fn validate_receipt(
    receipt: &serde_json::Value,
    run: &serde_json::Value,
    options: &Options,
) -> Result<(), String> {
    if receipt.get("schema").and_then(serde_json::Value::as_str)
        != Some("mesh-packaging-readiness-v1")
    {
        return Err("unsupported packaging readiness schema".to_owned());
    }
    if receipt
        .get("correlation_id")
        .and_then(serde_json::Value::as_str)
        != Some(options.correlation_id.as_str())
    {
        return Err("packaging receipt correlation ID mismatch".to_owned());
    }
    if receipt
        .get("packaging_run_id")
        .and_then(serde_json::Value::as_u64)
        != run.get("id").and_then(serde_json::Value::as_u64)
        || receipt
            .get("packaging_run_attempt")
            .and_then(serde_json::Value::as_u64)
            != run.get("run_attempt").and_then(serde_json::Value::as_u64)
    {
        return Err("packaging receipt run identity mismatch".to_owned());
    }
    let expected_upstream = serde_json::json!({
        "repository": UPSTREAM_REPOSITORY,
        "ref": options.upstream_ref,
        "sha": options.upstream_sha,
        "manifest_sha256": options.manifest_sha256,
    });
    if receipt.get("upstream") != Some(&expected_upstream) {
        return Err("packaging receipt upstream release identity mismatch".to_owned());
    }
    let expected_requested = serde_json::json!({
        "publish_images": "true",
        "publish_release_assets": "true",
        "publish_npm": "true",
    });
    if receipt.get("requested") != Some(&expected_requested) {
        return Err("packaging receipt did not select all required release channels".to_owned());
    }
    if receipt.get("status").and_then(serde_json::Value::as_str) != Some("success")
        || run.get("conclusion").and_then(serde_json::Value::as_str) != Some("success")
    {
        return Err(format!(
            "packaging did not complete successfully: {}",
            receipt.get("results").unwrap_or(&serde_json::Value::Null)
        ));
    }
    Ok(())
}

fn valid_correlation_id(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse(args: &[String]) -> Result<Options, CheckReport> {
    const GRAMMAR: Grammar = Grammar {
        usage: USAGE,
        values: &[
            "--repository",
            "--correlation-id",
            "--upstream-repository",
            "--upstream-ref",
            "--upstream-sha",
            "--manifest-sha256",
            "--out",
            "--timeout-seconds",
        ],
        flags: &[],
    };
    let parsed = GRAMMAR.parse(args)?;
    let missing: Vec<&str> = GRAMMAR
        .values
        .iter()
        .filter(|name| **name != "--timeout-seconds" && parsed.last(name).is_none())
        .copied()
        .collect();
    if !missing.is_empty() || !parsed.positionals.is_empty() {
        return Err(GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        )));
    }
    let value = |name: &str| parsed.last(name).unwrap_or_default();
    if value("--repository") != REPOSITORY || value("--upstream-repository") != UPSTREAM_REPOSITORY
    {
        return Err(GRAMMAR.error("unexpected release repository"));
    }
    if !valid_correlation_id(value("--correlation-id")) {
        return Err(GRAMMAR.error("invalid correlation ID"));
    }
    if !hex(value("--upstream-sha"), 40) || !hex(value("--manifest-sha256"), 64) {
        return Err(GRAMMAR.error("invalid release digest"));
    }
    let timeout = match value("--timeout-seconds") {
        "" => 3 * 60 * 60,
        text => text
            .parse::<u64>()
            .map_err(|_| GRAMMAR.error("invalid --timeout-seconds"))?,
    };
    Ok(Options {
        correlation_id: value("--correlation-id").to_owned(),
        upstream_ref: value("--upstream-ref").to_owned(),
        upstream_sha: value("--upstream-sha").to_owned(),
        manifest_sha256: value("--manifest-sha256").to_owned(),
        out: std::path::PathBuf::from(value("--out")),
        timeout: Duration::from_secs(timeout),
    })
}

fn await_packaging(options: &Options) -> Result<String, String> {
    let deadline = Instant::now() + options.timeout;
    while Instant::now() < deadline {
        if let Some(run) = find_run(options)?
            && run.get("status").and_then(serde_json::Value::as_str) == Some("completed")
        {
            let receipt = load_receipt(&run)?;
            let mut text = serde_json::to_string_pretty(&receipt).map_err(|e| e.to_string())?;
            text.push('\n');
            std::fs::write(&options.out, text)
                .map_err(|error| format!("{}: {error}", options.out.display()))?;
            validate_receipt(&receipt, &run, options)?;
            return Ok(format!(
                "Verified packaging readiness from run {}/{}",
                run.get("id")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default(),
                run.get("run_attempt")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_default(),
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    Err("correlated packaging run did not reach terminal readiness before the deadline".to_owned())
}

pub(super) fn run(args: &[String]) -> CheckReport {
    let options = match parse(args) {
        Ok(options) => options,
        Err(report) => return report,
    };
    match await_packaging(&options) {
        Ok(message) => CheckReport::success(format!("{message}\n")),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Packaging readiness failed: {error}\n"),
        ),
    }
}

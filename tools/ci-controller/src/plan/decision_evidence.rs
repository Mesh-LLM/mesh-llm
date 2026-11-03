//! Bounded, non-authoritative join of a selector trace and supplied context.
//!
//! This command cannot authenticate its caller, bind a selector job to a
//! consumer job, observe a cache backend, or retain an artifact. Its output
//! keeps supplied context explicitly unverified and does not establish an
//! executing runner, effective cache mode, protected policy, or qualification.

use crate::DynResult;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

const MAX_INPUT: u64 = 16 * 1024;
const SUPPORTED_ROLES: &[&str] = &[
    "gpu-nvidia",
    "linux-build-4",
    "linux-build-8",
    "linux-native-16",
    "linux-plan",
    "macos-build",
    "platform-build",
    "windows-build",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    selection_trace: String,
    #[serde(default)]
    context: Context,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Context {
    role: Option<String>,
    run_id: Option<u64>,
    run_attempt: Option<u32>,
    job_id: Option<String>,
    source_sha: Option<String>,
    policy_sha: Option<String>,
    started_at: Option<u64>,
    runner: Option<RunnerInput>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerInput {
    name: Option<String>,
    os: Option<String>,
    arch: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema_version: u8,
    kind: String,
    event: String,
    declared_origin: String,
    ref_kind: String,
    #[serde(deserialize_with = "required_optional")]
    declared_head_sha: Option<String>,
    repository_matches: bool,
    head_repository_matches: bool,
    main_gate: bool,
    pr_gate: bool,
    exception_active: bool,
    manual_depot_request: bool,
    force_hosted: bool,
    selected_provider: String,
    selected_labels: Labels,
    reason: String,
    allow_depot_remote_cache: bool,
    allow_native_github_cache: bool,
    allow_trusted_sccache_seed: bool,
    role: (),
    policy_revision: (),
    actual_runner: (),
    effective_cache_mode: (),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Labels {
    linux: String,
    linux_4: String,
    linux_8: String,
    linux_16: String,
    linux_arm: String,
    linux_arm_4: String,
    linux_arm_8: String,
    linux_arm_16: String,
    macos: String,
    windows: String,
}

#[derive(Serialize)]
struct Record {
    schema_version: u8,
    kind: &'static str,
    authenticity: &'static str,
    selection_to_job_binding: &'static str,
    selection_sha256: String,
    selection: Selection,
    claimed_context: SafeContext,
    verified_policy_revision: (),
    actual_runner: (),
    effective_cache_mode: (),
    retention: &'static str,
    authorizes_execution: bool,
}

#[derive(Serialize)]
struct SafeContext {
    role: Option<String>,
    run_id: Option<u64>,
    run_attempt: Option<u32>,
    job_id_sha256: Option<String>,
    source_sha: Option<String>,
    policy_sha: Option<String>,
    started_at: Option<u64>,
    runner: Option<SafeRunner>,
}

#[derive(Serialize)]
struct SafeRunner {
    name_sha256: Option<String>,
    os: Option<String>,
    arch: Option<String>,
}

fn required_optional<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte))
}

fn role(value: &str) -> bool {
    SUPPORTED_ROLES.contains(&value)
}

fn valid_labels(labels: &Labels) -> bool {
    let linux = [
        &labels.linux,
        &labels.linux_4,
        &labels.linux_8,
        &labels.linux_16,
        &labels.linux_arm,
        &labels.linux_arm_4,
        &labels.linux_arm_8,
        &labels.linux_arm_16,
    ];
    let allowed = [
        "ubuntu-24.04",
        "ubuntu-24.04-arm",
        "depot-ubuntu-24.04",
        "depot-ubuntu-24.04-4",
        "depot-ubuntu-24.04-8",
        "depot-ubuntu-24.04-16",
        "depot-ubuntu-24.04-arm",
        "depot-ubuntu-24.04-arm-4",
        "depot-ubuntu-24.04-arm-8",
        "depot-ubuntu-24.04-arm-16",
    ];
    linux.iter().all(|value| allowed.contains(&value.as_str()))
        && ["macos-15", "depot-macos-15"].contains(&labels.macos.as_str())
        && ["windows-2022", "depot-windows-2022"].contains(&labels.windows.as_str())
}

fn valid_selection(selection: &Selection) -> bool {
    selection.schema_version == 1
        && selection.kind == "selection_only"
        && [
            "pull_request",
            "pull_request_target",
            "push",
            "workflow_dispatch",
            "other",
        ]
        .contains(&selection.event.as_str())
        && [
            "pull_request",
            "pull_request_target",
            "push",
            "workflow_dispatch",
            "unknown",
        ]
        .contains(&selection.declared_origin.as_str())
        && ["main", "pull_request_merge", "other"].contains(&selection.ref_kind.as_str())
        && selection.declared_head_sha.as_deref().is_none_or(sha)
        && ["github_hosted", "depot"].contains(&selection.selected_provider.as_str())
        && [
            "not_eligible_or_disabled",
            "eligible_depot_selection",
            "forced_hosted",
        ]
        .contains(&selection.reason.as_str())
        && valid_labels(&selection.selected_labels)
}

fn valid_context(context: &Context) -> bool {
    context.role.as_deref().is_none_or(role)
        && context.job_id.as_deref().is_none_or(token)
        && context.source_sha.as_deref().is_none_or(sha)
        && context.policy_sha.as_deref().is_none_or(sha)
        && context.run_id.is_none_or(|value| value > 0)
        && context.run_attempt.is_none_or(|value| value > 0)
        && context.started_at.is_none_or(|value| value > 0)
        && context.runner.as_ref().is_none_or(|runner| {
            runner.name.as_ref().is_none_or(|name| {
                !name.is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
            }) && runner
                .os
                .as_deref()
                .is_none_or(|os| ["Linux", "macOS", "Windows"].contains(&os))
                && runner
                    .arch
                    .as_deref()
                    .is_none_or(|arch| ["X64", "ARM64"].contains(&arch))
        })
}

/// Render an unverified claim, preserving unknowns and refusing unsafe input.
///
/// The selection digest binds the exact supplied trace bytes; it does not prove
/// that the trace came from a protected selector or from the claimed run.
pub fn render(input: &[u8]) -> Result<String, &'static str> {
    if input.len() as u64 > MAX_INPUT {
        return Err("invalid or oversized execution-record input");
    }
    let request: Request =
        serde_json::from_slice(input).map_err(|_| "invalid or oversized execution-record input")?;
    let selection: Selection = serde_json::from_str(&request.selection_trace)
        .map_err(|_| "invalid or oversized execution-record input")?;
    if !valid_selection(&selection) || !valid_context(&request.context) {
        return Err("invalid or oversized execution-record input");
    }
    let context = request.context;
    let runner = context.runner.map(|runner| SafeRunner {
        name_sha256: runner.name.map(|name| digest(name.as_bytes())),
        os: runner.os,
        arch: runner.arch,
    });
    let record = Record {
        schema_version: 1,
        kind: "execution_context_claim",
        authenticity: "unverified",
        selection_to_job_binding: "unverified",
        selection_sha256: digest(request.selection_trace.as_bytes()),
        selection,
        claimed_context: SafeContext {
            role: context.role,
            run_id: context.run_id,
            run_attempt: context.run_attempt,
            job_id_sha256: context.job_id.map(|value| digest(value.as_bytes())),
            source_sha: context.source_sha,
            policy_sha: context.policy_sha,
            started_at: context.started_at,
            runner,
        },
        verified_policy_revision: (),
        actual_runner: (),
        effective_cache_mode: (),
        retention: "not_recorded",
        authorizes_execution: false,
    };
    serde_json::to_string(&record).map_err(|_| "unable to render execution record")
}

/// Read one bounded request from stdin and write one non-authoritative record.
pub fn run(args: &[String]) -> DynResult<()> {
    if !args.is_empty() {
        return Err("usage: ci execution-record < request.json".into());
    }
    let mut input = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_INPUT + 1)
        .read_to_end(&mut input)?;
    let record = render(&input)?;
    let mut output = std::io::stdout().lock();
    writeln!(output, "{record}")?;
    Ok(())
}

#[cfg(test)]
#[path = "decision_evidence_tests.rs"]
mod tests;

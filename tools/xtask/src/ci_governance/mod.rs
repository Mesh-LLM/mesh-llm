//! Local policy checks and explicitly activated, protected-source GitHub adapters.
pub(crate) mod activation;
pub(crate) mod app_auth;
pub(crate) mod approval;
pub(crate) mod collector;
mod live_inspection;
pub(crate) mod operations;
pub(crate) mod ownership;
pub(crate) mod policy;
pub(crate) mod publisher;
pub(crate) mod scope;

#[cfg(test)]
mod approval_tests;
#[cfg(test)]
mod ownership_tests;

use crate::command::{DynResult, print_json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args.is_empty() || args == ["--help"] {
        writeln!(
            crate::cli_output::stdout(),
            "ci-governance validate-pr --policy PATH --snapshot PATH\nci-governance validate-ownership --policy PATH --codeowners PATH\nci-governance validate-change --policy PATH --expected PATH --record PATH --observed PATH\nci-governance check-drift --policy PATH --expected PATH --observed PATH\nci-governance inspect-live-controls\nci-governance activation-status\nci-governance collect-pr <positive PR number>\nci-governance publish-pr <positive PR number>\nci-governance resolve-publisher-event\nci-governance publish-event --expected-pr N\nci-governance prepare-administration --record-id UUID --record-sha256 SHA256\nci-governance inspect-administration-approval --record-id UUID --record-sha256 SHA256\nLocal snapshot checks only establish consistency; the read-only inspector reports partial unverified observations; activated enforcement requires protected activation and explicit App credentials."
        )?;
        return Ok(());
    }
    let command = args[0].as_str();
    match command {
        "inspect-live-controls" => return print_json(&live_inspection::run(root, &args[1..])?),
        "activation-status" => return publisher::activation_status(root, &args[1..]),
        "publish-pr" => return publisher::run(root, &args[1..]),
        "collect-pr" => return publisher::collect_pr(root, &args[1..]),
        "resolve-publisher-event" => return publisher::resolve_event(root, &args[1..]),
        "publish-event" => return publisher::publish_event(root, &args[1..]),
        "prepare-administration" => {
            return print_json(&operations::prepare_administration(root, &args[1..])?);
        }
        "inspect-administration-approval" => {
            print_json(&operations::inspect_administration_approval(
                root,
                &args[1..],
            )?)?;
            return Err("administration authorization blocked: environment history cannot prove exact run attempt and packet digest; no operation or credential release authorized".into());
        }
        _ => {}
    }
    let flags = parse_flags(&args[1..])?;
    let policy_path = flag_path(root, &flags, "--policy")?;
    let policy_bytes = read_bounded(&policy_path)?;
    let policy = policy::Policy::from_bytes(&policy_bytes)?;
    match command {
        "validate-pr" => {
            require_flags(&flags, &["--policy", "--snapshot"])?;
            let bytes = read_bounded(&flag_path(root, &flags, "--snapshot")?)?;
            let snapshot = serde_json::from_slice(&bytes)?;
            let digest = hex::encode(Sha256::digest(&policy_bytes));
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            print_json(&approval::evaluate_at(&policy, &snapshot, &digest, now)?)
        }
        "validate-ownership" => {
            require_flags(&flags, &["--policy", "--codeowners"])?;
            let bytes = read_bounded(&flag_path(root, &flags, "--codeowners")?)?;
            print_json(&ownership::validate(&policy, std::str::from_utf8(&bytes)?)?)
        }
        "validate-change" => {
            require_flags(
                &flags,
                &["--policy", "--expected", "--record", "--observed"],
            )?;
            print_json(&operations::validate_change(
                &policy,
                &flag_path(root, &flags, "--expected")?,
                &flag_path(root, &flags, "--record")?,
                &flag_path(root, &flags, "--observed")?,
            )?)
        }
        "check-drift" => {
            require_flags(&flags, &["--policy", "--expected", "--observed"])?;
            match operations::check_drift(
                &policy,
                &flag_path(root, &flags, "--expected")?,
                &flag_path(root, &flags, "--observed")?,
            )? {
                operations::DriftOutcome::Match(report) => print_json(&report),
                operations::DriftOutcome::Mismatch {
                    report,
                    differences,
                } => {
                    print_json(&report)?;
                    Err(format!("CI governance drift: {}", differences.join(", ")).into())
                }
            }
        }
        _ => Err(format!("unknown ci-governance command {command}").into()),
    }
}

fn parse_flags(args: &[String]) -> DynResult<BTreeMap<String, String>> {
    if !args.len().is_multiple_of(2) {
        return Err("ci-governance options require flag/value pairs".into());
    }
    let mut flags = BTreeMap::new();
    for pair in args.as_chunks::<2>().0 {
        if !pair[0].starts_with("--")
            || pair[1].is_empty()
            || flags.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("invalid or duplicate ci-governance option".into());
        }
    }
    Ok(flags)
}

fn require_flags(flags: &BTreeMap<String, String>, expected: &[&str]) -> DynResult<()> {
    if flags.len() != expected.len() || expected.iter().any(|flag| !flags.contains_key(*flag)) {
        return Err(format!("required options: {}", expected.join(" ")).into());
    }
    Ok(())
}

fn flag_path(root: &Path, flags: &BTreeMap<String, String>, flag: &str) -> DynResult<PathBuf> {
    let value = flags.get(flag).ok_or_else(|| format!("missing {flag}"))?;
    Ok(root.join(value))
}

fn read_bounded(path: &Path) -> DynResult<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(
            "governance input exceeds 8 MiB; incomplete/truncated data is forbidden".into(),
        );
    }
    Ok(bytes)
}

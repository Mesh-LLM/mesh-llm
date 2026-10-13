//! Worker receipts and publication text consume the same verified package.
use crate::{
    automation::canary_receipts::{
        Digest, Family, PackageVerification, ReceiptContext, WorkerOutcome, WorkerResult,
        verify_package, write_receipt,
    },
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    package: PathBuf,
    identity_sha256: Digest,
    run_id: String,
    run_attempt: String,
    controller_revision: String,
    #[serde(default)]
    selected_source: String,
    evidence: Option<PathBuf>,
    family: Option<Family>,
    outcome: Option<WorkerOutcome>,
    runner: Option<String>,
    repository: Option<String>,
}
pub(crate) fn run(args: &[String], publication: bool) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation canary-receipts {receipt|publication} --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(value) => value,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input =
        serde_json::from_slice(&fs::read(parsed.last("--input").ok_or("missing --input")?)?)?;
    execute(&input, publication)?;
    CheckReport::success(
        if publication {
            "wrote verified publication summary\n"
        } else {
            "wrote verified worker receipt\n"
        }
        .into(),
    )
    .emit()
}
fn execute(input: &Input, publication: bool) -> DynResult<()> {
    if !input.package.is_absolute()
        || !fs::symlink_metadata(&input.package)?.file_type().is_dir()
        || input.controller_revision.len() != 40
        || !input
            .controller_revision
            .bytes()
            .all(|byte| matches!(byte,b'0'..=b'9'|b'a'..=b'f'))
    {
        return Err("handoff requires an absolute package and frozen controller revision".into());
    }
    let package = verify_package(
        &input.package,
        PackageVerification {
            expected_identity_sha256: input.identity_sha256.clone(),
            current_run_id: input.run_id.clone(),
            current_run_attempt: input.run_attempt.clone(),
            controller_revision: Some(input.controller_revision.clone()),
            selected_source: input.selected_source.clone(),
        },
    )?;
    if publication {
        if input.evidence.is_some()
            || input.family.is_some()
            || input.outcome.is_some()
            || input.runner.is_some()
        {
            return Err("publication does not accept worker receipt fields".into());
        }
        let (identity, count, bundle) = package.publication_context();
        let body = publication_body(
            identity,
            count,
            bundle,
            input
                .repository
                .as_deref()
                .ok_or("publication requires repository")?,
            &fs::read_to_string(input.package.join("upstream-summary.md"))?,
        )?;
        let path = input.package.join("pr-body.md");
        if let Ok(metadata) = fs::symlink_metadata(&path)
            && !metadata.file_type().is_file()
        {
            return Err("publication summary destination must be a regular file".into());
        }
        fs::write(path, body)?;
    } else {
        if input.repository.is_some() {
            return Err("receipt does not accept publication repository".into());
        }
        let evidence = input.evidence.as_ref().ok_or("receipt requires evidence")?;
        if !evidence.is_absolute() {
            return Err("receipt evidence must be absolute".into());
        }
        write_receipt(
            &ReceiptContext::from_verified_package(package),
            evidence,
            WorkerResult {
                family: input.family.clone().ok_or("receipt requires family")?,
                outcome: input.outcome.ok_or("receipt requires outcome")?,
                runner: input.runner.clone(),
            },
        )?;
    }
    Ok(())
}
fn publication_body(
    identity: &crate::automation::canary_receipts::ProducerIdentity,
    count: usize,
    bundle: bool,
    repository: &str,
    summary: &str,
) -> DynResult<String> {
    let parts: Vec<_> = repository.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || [".", ".."].contains(part)
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        })
    {
        return Err("publication repository must be OWNER/NAME".into());
    }
    if !identity.pass_id.starts_with("verify-") || !bundle {
        return Err("publication requires the independent verifier package".into());
    }
    Ok(format!(
        "Update the llama.cpp pin and its patch queue to the independently certified candidate.\n\nCandidate: `{}`. Both complete per-family passes succeeded on this exact tree. Each pass independently rebuilt the native/Rust binaries and ran the complete roster, including single-step, chain, state-handoff, native draft requirements, applicable multimodal smokes, and class-specific workload smoke/oracle lanes for the non-chat families.\n\nEvidence: https://github.com/{repository}/actions/runs/{} ({}; {count} families).\n\n{summary}",
        identity.candidate, identity.run_id, identity.pass_id
    ))
}

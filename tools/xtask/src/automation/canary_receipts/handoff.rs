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
    #[serde(default)]
    retry_recovered: String,
    evidence: Option<PathBuf>,
    family: Option<Family>,
    outcome: Option<WorkerOutcome>,
    runner: Option<String>,
    repository: Option<String>,
    source_root: Option<PathBuf>,
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
        let selected = !input.selected_source.is_empty() && !bundle;
        let recovered = input.retry_recovered == "true";
        if !matches!(input.retry_recovered.as_str(), "" | "true" | "false") {
            return Err("publication retry recovery marker must be true, false or absent".into());
        }
        if selected {
            super::canary_package_closure::selected_publication::prepare(
                input
                    .source_root
                    .as_deref()
                    .ok_or("selected publication requires source root")?,
                &input.package,
                &input.controller_revision,
                identity,
                recovered,
            )?;
        } else if input.source_root.is_some() {
            return Err("ordinary publication does not accept selected source root".into());
        }
        let body = publication_body(
            identity,
            count,
            bundle || selected,
            recovered,
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
        if input.repository.is_some() || input.source_root.is_some() {
            return Err("receipt does not accept publication repository".into());
        }
        if !input.retry_recovered.is_empty() {
            return Err("receipt does not accept the publication recovery marker".into());
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
    recovered: bool,
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
    let admitted = match super::canary_receipts::pass_identity::PassId::parse(&identity.pass_id) {
        Ok(pass) => {
            if recovered {
                pass.is_repair()
            } else {
                !pass.is_repair()
            }
        }
        Err(_) => false,
    };
    if !admitted || !bundle {
        return Err("publication requires the independent verifier package".into());
    }
    let (lead, evidence) = if recovered {
        (
            "Publish the certified llama.cpp candidate and its MeshLLM changes.",
            "The family pass certified this exact tree green; every infrastructure retry was recovered by trusted reconciliation against the immutable producer package, so no second independent pass was required. Certification ran the complete roster, including single-step, chain, state-handoff, native draft requirements, applicable multimodal smokes, and class-specific workload smoke/oracle lanes for the non-chat families.",
        )
    } else {
        (
            "Publish the independently certified llama.cpp candidate and its MeshLLM changes.",
            "Both complete per-family passes succeeded on this exact tree. Each pass independently rebuilt the native/Rust binaries and ran the complete roster, including single-step, chain, state-handoff, native draft requirements, applicable multimodal smokes, and class-specific workload smoke/oracle lanes for the non-chat families.",
        )
    };
    Ok(format!(
        "{lead}\n\nCandidate: `{}`. {evidence}\n\nEvidence: https://github.com/{repository}/actions/runs/{} ({}; {count} families).\n\n{summary}",
        identity.candidate, identity.run_id, identity.pass_id
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(pass_id: &str) -> crate::automation::canary_receipts::ProducerIdentity {
        crate::automation::canary_receipts::ProducerIdentity {
            candidate: "a".repeat(40),
            branch: "llama-canary/repair-1-a4d4f2fd4e".into(),
            pass_id: pass_id.into(),
            run_id: "42678880338".into(),
            run_attempt: crate::automation::canary_receipts::RunAttempt::try_from("1".to_owned())
                .unwrap(),
        }
    }

    #[test]
    fn verified_package_publishes_independent_evidence_text() {
        let body = publication_body(
            &identity("verify-2"),
            95,
            true,
            false,
            "Mesh-LLM/mesh-llm",
            "sum",
        )
        .unwrap();
        assert!(body.contains("verify-2"));
        assert!(body.contains("42678880338"));
        assert!(body.contains(&"a".repeat(40)));
    }

    #[test]
    fn recovered_repair_package_publishes_reconciliation_evidence_text() {
        let body = publication_body(
            &identity("repair-2"),
            95,
            true,
            true,
            "Mesh-LLM/mesh-llm",
            "sum",
        )
        .unwrap();
        assert!(!body.contains("independently certified"));
        assert!(body.contains("repair-2"));
    }

    #[test]
    fn unrecovered_repair_package_cannot_publish() {
        assert!(
            publication_body(
                &identity("repair-2"),
                95,
                true,
                false,
                "Mesh-LLM/mesh-llm",
                "sum"
            )
            .is_err()
        );
        assert!(
            publication_body(
                &identity("repair-2"),
                95,
                false,
                true,
                "Mesh-LLM/mesh-llm",
                "sum"
            )
            .is_err()
        );
    }

    #[test]
    fn recovered_marker_cannot_launder_a_verify_package() {
        assert!(
            publication_body(
                &identity("verify-1"),
                95,
                true,
                true,
                "Mesh-LLM/mesh-llm",
                "sum"
            )
            .is_err()
        );
    }

    #[test]
    fn malformed_repository_cannot_publish() {
        assert!(
            publication_body(&identity("verify-1"), 95, true, false, "mesh-llm", "sum").is_err()
        );
    }

    #[test]
    fn malformed_pass_identity_cannot_publish() {
        for pass_id in ["repair-x", "verify-", "repair-4", "verification"] {
            assert!(
                publication_body(
                    &identity(pass_id),
                    95,
                    true,
                    false,
                    "Mesh-LLM/mesh-llm",
                    "sum"
                )
                .is_err(),
                "{pass_id}"
            );
        }
    }

    const PLAN: &[u8] =
        include_bytes!("../../../tests/migration_canary_receipts/fixtures/plan.json");

    fn write_package(
        root: &std::path::Path,
        pass_id: &str,
        mesh_source: Option<&str>,
        bundle: bool,
    ) -> (std::path::PathBuf, Digest) {
        let package = root.join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(package.join("plan.json"), PLAN).unwrap();
        let artifacts: [(&str, &[u8]); 4] = [
            ("binaries.tar", b"binaries"),
            ("workload-oracles.tar", b"oracles"),
            ("llama-source.bundle", b"llama bundle"),
            ("llama-source.json", b"llama provenance"),
        ];
        for (name, bytes) in artifacts {
            fs::write(package.join(name), bytes).unwrap();
        }
        if bundle {
            fs::write(package.join("candidate.bundle"), b"candidate bundle").unwrap();
        }
        let source = "a".repeat(40);
        let mut identity = serde_json::json!({
            "schema": 3,
            "platform": "macos-arm64-metal",
            "candidate": source,
            "base": source,
            "branch": format!("llama-canary/{pass_id}-aaaaaaaaaa"),
            "pass_id": pass_id,
            "run_id": "123",
            "run_attempt": "1",
            "controller": "b".repeat(40),
            "plan_sha256": Digest::of_bytes(PLAN),
            "binaries_sha256": Digest::of_bytes(b"binaries"),
            "workload_oracles_sha256": Digest::of_bytes(b"oracles"),
            "llama_bundle_sha256": Digest::of_bytes(b"llama bundle"),
            "llama_provenance_sha256": Digest::of_bytes(b"llama provenance"),
        });
        if bundle {
            identity["bundle_sha256"] =
                serde_json::to_value(Digest::of_bytes(b"candidate bundle")).unwrap();
        }
        if let Some(mesh_source) = mesh_source {
            identity["mesh_source"] = mesh_source.into();
        }
        let identity = serde_json::to_vec(&identity).unwrap();
        fs::write(package.join("identity.json"), &identity).unwrap();
        (package, Digest::of_bytes(&identity))
    }

    fn publication_input(
        package: &std::path::Path,
        identity_sha256: &Digest,
        retry_recovered: &str,
        selected_source: &str,
    ) -> Input {
        Input {
            package: package.to_path_buf(),
            identity_sha256: identity_sha256.clone(),
            run_id: "123".into(),
            run_attempt: "1".into(),
            controller_revision: "b".repeat(40),
            selected_source: selected_source.into(),
            retry_recovered: retry_recovered.into(),
            evidence: None,
            family: None,
            outcome: None,
            runner: None,
            repository: Some("Mesh-LLM/mesh-llm".into()),
            source_root: None,
        }
    }

    #[test]
    fn recovered_repair_package_publishes_through_execute() {
        let root = tempfile::tempdir().unwrap();
        let (package, identity_sha256) = write_package(root.path(), "repair-1", None, true);
        fs::write(package.join("upstream-summary.md"), "summary").unwrap();
        let input = publication_input(&package, &identity_sha256, "true", "");
        execute(&input, true).unwrap();
        let body = fs::read_to_string(package.join("pr-body.md")).unwrap();
        assert!(body.contains("repair-1"));
        assert!(!body.contains("independently certified"));
    }

    #[test]
    fn execute_rejects_malformed_recovery_marker() {
        let root = tempfile::tempdir().unwrap();
        let (package, identity_sha256) = write_package(root.path(), "repair-1", None, true);
        let input = publication_input(&package, &identity_sha256, "yes", "");
        let error = execute(&input, true).unwrap_err();
        assert!(error.to_string().contains("must be true, false or absent"));
    }

    #[test]
    fn recovered_selected_source_requires_controller_checkout_for_export() {
        let root = tempfile::tempdir().unwrap();
        let source = "a".repeat(40);
        let (package, identity_sha256) =
            write_package(root.path(), "repair-1", Some(&source), false);
        let input = publication_input(&package, &identity_sha256, "true", &source);
        let error = execute(&input, true).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("selected publication requires source root")
        );
    }

    #[test]
    fn receipt_rejects_recovery_marker() {
        let root = tempfile::tempdir().unwrap();
        let (package, identity_sha256) = write_package(root.path(), "repair-1", None, true);
        let mut input = publication_input(&package, &identity_sha256, "true", "");
        input.repository = None;
        let error = execute(&input, false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("receipt does not accept the publication recovery marker")
        );
    }
}

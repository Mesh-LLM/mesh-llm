//! The protected catalogs with real Cargo: every routing profile the
//! protected callers use, and the inert PR manifest root against the
//! protected default. The manifest root only relocates the two catalogs;
//! Cargo discovery and the reverse-dependency closure stay in the protected
//! checkout, so the plans must be byte-identical.

use super::Run;
use super::action::real_action_outputs;
use super::manifest_root::{CATALOGS, entries};
use super::process::{Captured, process_difference};
use super::projection::{action_outputs, outputs_difference};
use super::report::Outcome;
use super::stage::Tools;
use crate::command::DynResult;
use std::path::Path;

/// One planner input, serialized in the action's `jq -cn` key order.
struct Input<'a> {
    label: &'a str,
    profile: &'a str,
    event: &'a str,
    changed: &'a [&'a str],
    affected: Option<&'a [&'a str]>,
}

fn json_list(items: &[&str]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_owned())
}

impl Input<'_> {
    fn bytes(&self, sha: &str) -> Vec<u8> {
        let base = if self.event == "pull_request" {
            sha
        } else {
            ""
        };
        let mut text = format!(
            r#"{{"profile":"{}","event_name":"{}","source_sha":"{sha}","base_sha":"{base}","changed_files":{}"#,
            self.profile,
            self.event,
            json_list(self.changed)
        );
        if let Some(affected) = self.affected.filter(|affected| !affected.is_empty()) {
            text.push_str(&format!(r#","affected_crates":{}"#, json_list(affected)));
        }
        text.push('}');
        text.into_bytes()
    }
}

const PULL_REQUESTS: [Input<'static>; 6] = [
    Input {
        label: "pr-ready-runtime",
        profile: "pr-ready",
        event: "pull_request",
        changed: &["crates/mesh-llm-host-runtime/src/lib.rs"],
        affected: Some(&["mesh-llm-host-runtime", "mesh-llm"]),
    },
    Input {
        label: "pr-ready-reverse-dependency",
        profile: "pr-ready",
        event: "pull_request",
        changed: &["crates/skippy-metrics/src/lib.rs"],
        affected: None,
    },
    Input {
        label: "pr-ready-control",
        profile: "pr-ready",
        event: "pull_request",
        changed: &[".github/workflows/pr_linux.yml"],
        affected: None,
    },
    Input {
        label: "pr-ready-native-pin",
        profile: "pr-ready",
        event: "pull_request",
        changed: &["third_party/llama.cpp/upstream.txt"],
        affected: None,
    },
    Input {
        label: "pr-draft-docs",
        profile: "pr-draft",
        event: "pull_request",
        changed: &["CONTRIBUTING.md"],
        affected: None,
    },
    Input {
        label: "pr-draft-docs-ci-control",
        profile: "pr-draft",
        event: "pull_request",
        changed: &["CONTRIBUTING.md", ".github/README.md"],
        affected: None,
    },
];

const TRUSTED: [Input<'static>; 2] = [
    Input {
        label: "main",
        profile: "main",
        event: "push",
        changed: &["crates/mesh-llm/src/lib.rs"],
        affected: None,
    },
    Input {
        label: "manual-full",
        profile: "manual-full",
        event: "workflow_dispatch",
        changed: &["__force_all__"],
        affected: None,
    },
];

fn plan(
    run: &Run<'_>,
    manifest_root: Option<&str>,
    stdin: &[u8],
) -> DynResult<(Captured, Vec<String>)> {
    let args = manifest_root
        .map(|root| vec!["--manifest-root".to_owned(), root.to_owned()])
        .unwrap_or_default();
    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let captured = run.rust_plan(&refs, &run.stage.search_path(Tools::Real), stdin)?;
    Ok((captured, args))
}

/// The inert PR root against the protected default, plus its contents.
pub(super) fn manifest_root(run: &mut Run<'_>, inert: &Path, sha: &str) -> DynResult<()> {
    let found = entries(inert)?;
    let difference = (found != CATALOGS).then(|| format!("inert root holds {found:?}"));
    run.ledger.record("manifest-root", "entries", difference);
    let inert_arg = inert.to_str().ok_or("non-UTF8 scratch path")?;
    let default_arg = run
        .root
        .to_str()
        .ok_or("non-UTF8 checkout path")?
        .to_owned();
    for input in &PULL_REQUESTS {
        let stdin = input.bytes(sha);
        let (pr, _) = plan(run, Some(inert_arg), &stdin)?;
        let (protected, _) = plan(run, Some(&default_arg), &stdin)?;
        let (implicit, _) = plan(run, None, &stdin)?;
        let failed = (pr.code != Some(0)).then(|| String::from_utf8_lossy(&pr.stderr).into_owned());
        let difference = failed
            .or_else(|| process_difference(&protected, &pr))
            .or_else(|| process_difference(&implicit, &pr));
        run.ledger.record("manifest-root", input.label, difference);
        run.keep(
            &format!("manifest-root/{}.plan.json", input.label),
            &pr.stdout,
        )?;
        let args = ["--manifest-root", inert_arg];
        let search_path = run.stage.search_path(Tools::Real);
        run.compare_legacy(
            "manifest-root",
            input.label,
            &pr,
            (&args, &search_path, &stdin),
        )?;
    }
    Ok(())
}

/// Every profile with real catalogs and Cargo; legacy side by side, then
/// the real action derivation over legacy stdout against Rust's projection.
pub(super) fn profiles(run: &mut Run<'_>, sha: &str) -> DynResult<()> {
    for input in TRUSTED.iter().chain(&PULL_REQUESTS) {
        let stdin = input.bytes(sha);
        let (rust, args) = plan(run, None, &stdin)?;
        run.keep(&format!("profiles/{}.plan.json", input.label), &rust.stdout)?;
        let projected = action_outputs(&rust.stdout);
        let label = format!("profile {}", input.label);
        let Ok(projected) = projected else {
            let detail = format!("{:?}: {}", rust.code, String::from_utf8_lossy(&rust.stderr));
            run.ledger
                .push("protected-profiles", label, Outcome::Different, detail);
            continue;
        };
        run.keep(
            &format!("profiles/{}.outputs.txt", input.label),
            projected.as_bytes(),
        )?;
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        let search_path = run.stage.search_path(Tools::Real);
        let legacy = run.compare_legacy(
            "protected-profiles",
            &label,
            &rust,
            (&refs, &search_path, &stdin),
        )?;
        match legacy {
            None => run.ledger.push(
                "protected-profiles",
                label,
                Outcome::RustOnly,
                "legacy not requested",
            ),
            Some(legacy) if legacy.code == Some(0) => {
                let actual = real_action_outputs(run, input.label, &legacy.stdout)?;
                let difference = outputs_difference(&projected, &actual);
                run.ledger.record(
                    "action-outputs",
                    format!("{label} legacy-action-vs-rust"),
                    difference,
                );
            }
            Some(_) => {}
        }
    }
    Ok(())
}

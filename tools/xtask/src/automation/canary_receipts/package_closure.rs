#[path = "package_closure/admission.rs"]
mod admission;
#[path = "package_closure/archive.rs"]
mod archive;
#[path = "package_closure/archive_extract.rs"]
mod archive_extract;
#[path = "package_closure/archive_write.rs"]
mod archive_write;
#[path = "package_closure/candidate_plan.rs"]
mod candidate_plan;
#[path = "package_closure/candidate_view.rs"]
mod candidate_view;
#[path = "package_closure/executable.rs"]
mod executable;
#[path = "package_closure/packing.rs"]
mod packing;
#[path = "package_closure/parity_inventory.rs"]
mod parity_inventory;
#[path = "package_closure/prepared_source.rs"]
pub(crate) mod prepared_source;
#[path = "package_closure/process.rs"]
mod process;
#[path = "package_closure/producer_receipt.rs"]
mod producer_receipt;
#[path = "package_closure/restore_transaction.rs"]
mod restore_transaction;
#[path = "package_closure/restoring.rs"]
mod restoring;
#[path = "package_closure/runtime_slice.rs"]
mod runtime_slice;
#[path = "package_closure/source.rs"]
mod source;
#[path = "package_closure/split_roster.rs"]
mod split_roster;
#[cfg(test)]
#[path = "package_closure/tests.rs"]
mod tests;
#[path = "package_closure/workload.rs"]
mod workload;

use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkloadInput {
    root: PathBuf,
    closure: PathBuf,
    source_diff_capture: PathBuf,
    candidate: Option<String>,
}

pub(crate) fn prepared_native_head(root: &std::path::Path) -> DynResult<String> {
    process::operation(|| Ok(source::prepared(root)?.head))
}

pub(crate) fn verified_workload_test(
    root: &std::path::Path,
    binary: &std::path::Path,
    native: &std::path::Path,
    manifest: &std::path::Path,
) -> DynResult<PathBuf> {
    process::operation(|| workload::production::verified_test(root, binary, native, manifest))
}

pub(crate) fn run(args: &[String], workload: bool) -> DynResult<()> {
    if workload && args.first().is_some_and(|arg| !arg.starts_with('-')) {
        return self::workload::production::run(args);
    }
    let grammar = Grammar {
        usage: if workload {
            "cargo xtool automation canary-receipts workload-manifest --input PATH"
        } else {
            "cargo xtool automation canary-receipts verify-package-closure --input PATH"
        },
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match grammar.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", grammar.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return grammar.error("unexpected positional arguments").emit();
    }
    let bytes = fs::read(parsed.last("--input").ok_or("missing --input")?)?;
    let output = process::operation(|| {
        if workload {
            workload_document(&bytes)
        } else {
            admission::verify_document(&bytes)
        }
    })?;
    CheckReport::success(format!("{}\n", std::str::from_utf8(&output)?)).emit()
}

fn workload_document(bytes: &[u8]) -> DynResult<Vec<u8>> {
    let input: WorkloadInput = serde_json::from_slice(bytes)?;
    if !input.root.is_absolute() || !input.closure.is_absolute() {
        return Err("workload source/closure must be absolute".into());
    }
    let prepared = source::prepared(&input.root)?;
    let proof = workload::verify_producer(
        &input.root,
        &input.closure,
        &input.source_diff_capture,
        &prepared.head,
    )?;
    match input.candidate {
        Some(candidate) => workload::seal(&input.root, &input.closure, &proof, &candidate),
        None => Ok(serde_json::to_vec(
            &serde_json::json!({"producer_sha256": proof.digest(), "native_head": prepared.head}),
        )?),
    }
}

/// Production package transaction verbs share one signal/cancellation scope.
pub(crate) fn transaction(args: &[String], verb: &str) -> DynResult<()> {
    let grammar = Grammar {
        usage: "cargo xtool automation canary-receipts <producer-receipt|candidate-plan|pack|restore> --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match grammar.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", grammar.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return grammar.error("unexpected positional arguments").emit();
    }
    let bytes = fs::read(parsed.last("--input").ok_or("missing --input")?)?;
    let result = process::operation(|| match verb {
        "producer-receipt" => Ok(
            serde_json::json!({"producer_receipt_sha256":producer_receipt::write(&serde_json::from_slice(&bytes)?)?}),
        ),
        "candidate-plan" => Ok(
            serde_json::json!({"admitted_identity_sha256":candidate_plan::admit(&serde_json::from_slice(&bytes)?)?}),
        ),
        "pack" => packing::pack(&serde_json::from_slice(&bytes)?),
        "restore" => restoring::restore(&serde_json::from_slice(&bytes)?),
        "manifest-policy" => manifest_policy::execute(&serde_json::from_slice(&bytes)?),
        "parity-inventory" => parity_inventory::execute(&serde_json::from_slice(&bytes)?),
        "split-roster" => split_roster::execute(&serde_json::from_slice(&bytes)?),
        "certify" => certification::execute(&serde_json::from_slice(&bytes)?),
        _ => Err("unknown package transaction".into()),
    });
    match result {
        Ok(output) => {
            if verb == "pack" {
                workflow_outputs(&output)?;
            }
            CheckReport::success(format!("{}\n", serde_json::to_string(&output)?)).emit()
        }
        Err(error) => {
            CheckReport::failure(String::new(), format!("canary {verb} rejected: {error}\n")).emit()
        }
    }
}

fn workflow_outputs(value: &serde_json::Value) -> DynResult<()> {
    use std::io::Write;
    if let Some(path) = std::env::var_os("GITHUB_OUTPUT") {
        let mut file = fs::OpenOptions::new().append(true).open(path)?;
        for key in ["matrix", "identity_sha256", "candidate", "branch"] {
            let field = &value[key];
            let output = if let Some(text) = field.as_str() {
                text.to_owned()
            } else {
                serde_json::to_string(field)?
            };
            if output.contains(['\n', '\r']) {
                return Err("invalid workflow output line".into());
            }
            writeln!(file, "{key}={output}")?;
        }
        file.sync_all()?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "package_closure/pipeline_tests.rs"]
mod pipeline_tests;

#[cfg(test)]
#[path = "package_closure/fixture_scope.rs"]
mod fixture_scope;

#[path = "package_closure/certification.rs"]
mod certification;

#[path = "package_closure/manifest_policy.rs"]
mod manifest_policy;
#[path = "package_closure/model_boundaries.rs"]
mod model_boundaries;

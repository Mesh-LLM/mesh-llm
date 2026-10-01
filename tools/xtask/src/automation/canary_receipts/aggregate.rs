use super::receipt::ReceiptProvenance;
use super::storage::{RECEIPT_LIMIT, RESULTS_LIMIT, read_bounded};
use super::{
    Digest, Error, ErrorKind, Family, ReceiptContext, RunAttempt, WorkerOutcome, WorkerReceipt,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize)]
pub(crate) struct ReceiptFailure {
    pub(crate) family: String,
    pub(crate) error: Error,
}

#[derive(Debug, Serialize)]
pub(crate) struct AggregateReport {
    pub(crate) candidate: String,
    pub(crate) branch: String,
    pub(crate) pass_id: String,
    pub(crate) planned_count: usize,
    pub(crate) passed: Vec<Family>,
    pub(crate) selected_attempts: BTreeMap<Family, RunAttempt>,
    pub(crate) failures: Vec<ReceiptFailure>,
}

struct SelectedReceipt {
    path: PathBuf,
    provenance: ReceiptProvenance,
    bytes: Vec<u8>,
}

impl AggregateReport {
    pub(crate) fn is_green(&self) -> bool {
        self.failures.is_empty()
    }

    pub(crate) fn summary(&self) -> String {
        let mut text = format!(
            "Canary {}: {}/{} family receipts passed for {}\n",
            self.pass_id,
            self.passed.len(),
            self.planned_count,
            self.candidate
        );
        if !self.failures.is_empty() {
            text.push('\n');
            for failure in &self.failures {
                text.push_str("- ");
                if !failure.family.is_empty() {
                    text.push_str(&failure.family);
                    text.push_str(": ");
                }
                text.push_str(&failure.error.message);
                text.push('\n');
            }
        }
        text
    }

    /// Receipt success alone does not replace the workflow's family job-result gate.
    pub(crate) fn github_outputs(&self) -> Option<String> {
        self.is_green().then(|| {
            format!(
                "green=true\ncandidate={}\nbranch={}\n",
                self.candidate, self.branch
            )
        })
    }
}

pub(crate) fn aggregate(
    context: &ReceiptContext,
    evidence: &Path,
) -> Result<AggregateReport, Error> {
    let identity = &context.package.identity;
    let mut report = AggregateReport {
        candidate: identity.candidate.clone(),
        branch: identity.branch.clone(),
        pass_id: identity.pass_id.clone(),
        planned_count: context.package.plan.models.len(),
        passed: Vec::new(),
        selected_attempts: BTreeMap::new(),
        failures: Vec::new(),
    };
    let mut latest: BTreeMap<Family, SelectedReceipt> = BTreeMap::new();
    let mut attempts = BTreeSet::new();
    for path in receipt_paths(evidence)? {
        let mut family = path
            .parent()
            .and_then(Path::file_name)
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
        let result = (|| {
            let bytes = read_bounded(&path, RECEIPT_LIMIT)?;
            let receipt = ReceiptProvenance::parse(&bytes)?;
            family = receipt.family.to_string();
            receipt.validate_identity(context)?;
            if !attempts.insert((receipt.family.clone(), receipt.run_attempt.clone())) {
                return Err(Error::new(
                    ErrorKind::DuplicateReceipt,
                    "duplicate family receipt in workflow attempt",
                ));
            }
            let replace = latest
                .get(&receipt.family)
                .is_none_or(|previous| receipt.run_attempt > previous.provenance.run_attempt);
            if replace {
                latest.insert(
                    receipt.family.clone(),
                    SelectedReceipt {
                        path: path.clone(),
                        provenance: receipt,
                        bytes,
                    },
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            report.failures.push(ReceiptFailure { family, error });
        }
    }
    for (family, receipt) in latest {
        report
            .selected_attempts
            .insert(family.clone(), receipt.provenance.run_attempt.clone());
        match check_selected(context, &receipt) {
            Ok(()) => report.passed.push(family),
            Err(error) => report.failures.push(ReceiptFailure {
                family: family.to_string(),
                error,
            }),
        }
    }
    let missing: Vec<_> = context
        .package
        .plan
        .models
        .keys()
        .filter(|family| !report.selected_attempts.contains_key(*family))
        .map(|family| format!("'{family}'"))
        .collect();
    if !missing.is_empty() {
        report.failures.push(ReceiptFailure {
            family: String::new(),
            error: Error::new(
                ErrorKind::MissingReceipts,
                format!("missing family receipts: [{}]", missing.join(", ")),
            ),
        });
    }
    Ok(report)
}

fn receipt_paths(evidence: &Path) -> Result<Vec<PathBuf>, Error> {
    let entries = match fs::read_dir(evidence) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path().join("receipt.json");
        match fs::symlink_metadata(&path) {
            Ok(_) => paths.push(path),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) => {}
            Err(error) => return Err(error.into()),
        }
    }
    paths.sort();
    Ok(paths)
}

fn check_selected(context: &ReceiptContext, selected: &SelectedReceipt) -> Result<(), Error> {
    let receipt: WorkerReceipt = serde_json::from_slice(&selected.bytes)?;
    match receipt.outcome {
        WorkerOutcome::Success => {}
        WorkerOutcome::Failure | WorkerOutcome::Cancelled | WorkerOutcome::Skipped => {
            return Err(Error::new(
                ErrorKind::WorkerOutcome,
                format!(
                    "failed or mismatched worker receipt (runner={}, outcome={})",
                    receipt.runner,
                    receipt.outcome.as_str()
                ),
            ));
        }
    }
    let results = read_bounded(
        &selected.path.with_file_name("results.jsonl"),
        RESULTS_LIMIT,
    )?;
    if Some(&Digest::of_bytes(&results)) != receipt.results_sha256.as_ref() {
        return Err(Error::new(
            ErrorKind::ResultsDigest,
            "worker results digest mismatch",
        ));
    }
    let family = &receipt.provenance.family;
    super::validate_results(&results, family, context.model(family)?)
}

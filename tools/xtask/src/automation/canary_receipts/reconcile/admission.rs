//! Snapshot the complete retry artifact closure before choosing newest receipts.
use super::super::feedback::{FamilyEvidence, evidence::Snapshot};
use super::super::storage::{RECEIPT_LIMIT, RESULTS_LIMIT, read_bounded};
use super::super::{
    Digest, Error, ErrorKind, Family, ReceiptContext, RunAttempt, WorkerOutcome, WorkerReceipt,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub(super) struct RetryReceipt {
    pub(super) attempt: RunAttempt,
    directory: PathBuf,
    receipt: WorkerReceipt,
    receipt_digest: Digest,
}
pub(super) struct Selected {
    pub(super) receipts: BTreeMap<Family, RetryReceipt>,
    // Keep the descriptor-admitted immutable bytes alive through certification.
    _snapshot: Option<Snapshot>,
}

pub(super) fn collect(
    context: &ReceiptContext,
    root: &Path,
    allowed: &BTreeSet<Family>,
) -> Result<Selected, Error> {
    match fs::symlink_metadata(root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Selected {
                receipts: BTreeMap::new(),
                _snapshot: None,
            });
        }
        Err(error) => return Err(error.into()),
        Ok(_) => (),
    }
    let snapshot = Snapshot::capture(root)?;
    if snapshot.files().keys().any(|path| !path.contains('/')) {
        return Err(contract("retry artifact root contains undeclared files"));
    }
    let directories: BTreeSet<_> = snapshot
        .directories()
        .iter()
        .map(|path| path.split('/').next().expect("nonempty snapshot path"))
        .collect();
    let mut receipts = BTreeMap::new();
    let mut attempts = BTreeSet::new();
    for directory in directories {
        let member = snapshot.root().join(directory);
        let bytes = read_bounded(&member.join("receipt.json"), RECEIPT_LIMIT)?;
        let receipt: WorkerReceipt = serde_json::from_slice(&bytes)?;
        receipt.provenance.validate_identity(context)?;
        let family = receipt.provenance.family.clone();
        if !allowed.contains(&family) {
            return Err(contract(
                "retry receipt is outside the prior infrastructure set",
            ));
        }
        let attempt = receipt.provenance.run_attempt.clone();
        if !attempts.insert((family.clone(), attempt.clone())) {
            return Err(Error::new(
                ErrorKind::DuplicateReceipt,
                "duplicate retry family receipt in workflow attempt",
            ));
        }
        let replace = receipts
            .get(&family)
            .is_none_or(|previous: &RetryReceipt| attempt > previous.attempt);
        if replace {
            receipts.insert(
                family,
                RetryReceipt {
                    attempt,
                    directory: member,
                    receipt,
                    receipt_digest: Digest::of_bytes(&bytes),
                },
            );
        }
    }
    Ok(Selected {
        receipts,
        _snapshot: Some(snapshot),
    })
}

pub(super) fn check(
    context: &ReceiptContext,
    family: &Family,
    selected: &RetryReceipt,
) -> Result<Option<FamilyEvidence>, Error> {
    if selected.receipt.outcome == WorkerOutcome::Success {
        let results = read_bounded(&selected.directory.join("results.jsonl"), RESULTS_LIMIT)?;
        if selected.receipt.results_sha256.as_ref() != Some(&Digest::of_bytes(&results)) {
            return Err(Error::new(
                ErrorKind::ResultsDigest,
                "retry result digest mismatch",
            ));
        }
        if super::super::validate_results(&results, family, context.model(family)?).is_ok() {
            return Ok(None);
        }
    }
    let admitted = FamilyEvidence::admit(context, family, &selected.directory)?;
    if !admitted.matches_selection(&selected.attempt, &selected.receipt_digest) {
        return Err(contract("retry evidence changed after receipt selection"));
    }
    Ok(Some(admitted))
}
fn contract(message: &str) -> Error {
    Error::new(ErrorKind::ReceiptIdentity, message)
}

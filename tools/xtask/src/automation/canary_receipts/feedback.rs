//! Digest-bound failed-family feedback. Admission never authorizes publication.
mod contract;
pub(super) mod evidence;
mod export;
mod summary;
#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests;

use super::failure_classification::{self, FailureClass};
use super::{Digest, Error, ErrorKind, Family, ReceiptContext, WorkerOutcome, WorkerReceipt};
use contract::Payload;
use evidence::Snapshot;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FeedbackState {
    CandidateRepairable,
    InfrastructureRetryable,
}
impl FeedbackState {
    const fn failure_class(self) -> &'static str {
        match self {
            Self::CandidateRepairable => "candidate",
            Self::InfrastructureRetryable => "infrastructure",
        }
    }
}

/// A classified receipt and an owned immutable copy of its exact evidence tree.
/// The fields cannot be constructed from unchecked caller paths or JSON.
pub(crate) struct FamilyEvidence {
    family: Family,
    class: FailureClass,
    identity: Digest,
    receipt_attempt: super::RunAttempt,
    receipt_digest: Digest,
    snapshot: Snapshot,
}
impl FamilyEvidence {
    pub(crate) fn admit(
        context: &ReceiptContext,
        family: &Family,
        directory: &Path,
    ) -> Result<Self, Error> {
        contract::safe_family(family)?;
        context.model(family)?;
        let snapshot = Snapshot::capture(directory)?;
        let receipt = admitted_receipt(context, family, &snapshot)?;
        if receipt.outcome == WorkerOutcome::Success {
            let bytes = super::storage::read_bounded(
                &snapshot.root().join("results.jsonl"),
                super::storage::RESULTS_LIMIT,
            )?;
            if super::validate_results(&bytes, family, context.model(family)?).is_ok() {
                return Err(contract_error(
                    "successful family evidence is not failure feedback",
                ));
            }
        }
        let class = failure_classification::classify(&receipt, snapshot.root());
        if class == FailureClass::Contract {
            return Err(contract_error(
                "family evidence cannot establish repair or retry custody",
            ));
        }
        Ok(Self {
            family: family.clone(),
            class,
            identity: context.package.identity_sha256.clone(),
            receipt_attempt: receipt.provenance.run_attempt.clone(),
            receipt_digest: snapshot.files["receipt.json"].clone(),
            snapshot,
        })
    }
    pub(crate) fn matches_selection(
        &self,
        attempt: &super::RunAttempt,
        receipt_digest: &Digest,
    ) -> bool {
        &self.receipt_attempt == attempt && &self.receipt_digest == receipt_digest
    }
    pub(crate) fn family(&self) -> &Family {
        &self.family
    }
    pub(crate) fn is_candidate(&self) -> bool {
        self.class == FailureClass::Candidate
    }
}

fn admitted_receipt(
    context: &ReceiptContext,
    family: &Family,
    snapshot: &Snapshot,
) -> Result<WorkerReceipt, Error> {
    let receipt: WorkerReceipt =
        serde_json::from_slice(&snapshot.read_metadata(Path::new("receipt.json"))?)?;
    receipt.provenance.validate_identity(context)?;
    if &receipt.provenance.family != family {
        return Err(contract_error("feedback receipt family mismatch"));
    }
    if matches!(
        receipt.outcome,
        WorkerOutcome::Success | WorkerOutcome::Failure
    ) && snapshot.files.get("results.jsonl") != receipt.results_sha256.as_ref()
    {
        return Err(contract_error("feedback worker result digest mismatch"));
    }
    Ok(receipt)
}

/// Prepared from admitted evidence and the aggregate's explicit missing set.
pub(crate) struct FeedbackDraft {
    payload: Payload,
    evidence: BTreeMap<Family, FamilyEvidence>,
}
impl FeedbackDraft {
    pub(crate) fn new(
        context: &ReceiptContext,
        state: FeedbackState,
        evidence: Vec<FamilyEvidence>,
        missing_infrastructure: BTreeSet<Family>,
        errors: Vec<String>,
    ) -> Result<Self, Error> {
        let mut admitted = BTreeMap::new();
        let mut candidate = BTreeSet::new();
        let mut infrastructure = missing_infrastructure;
        let mut bytes = 0u64;
        let mut files = 0usize;
        for item in evidence {
            if item.identity != context.package.identity_sha256
                || admitted.contains_key(&item.family)
                || infrastructure.contains(&item.family)
            {
                return Err(contract_error(
                    "feedback evidence identity, duplicate or missing-family mismatch",
                ));
            }
            bytes = bytes
                .checked_add(item.snapshot.bytes)
                .ok_or_else(|| contract_error("feedback byte count overflow"))?;
            files = files
                .checked_add(item.snapshot.files.len() + item.snapshot.directories.len())
                .ok_or_else(|| contract_error("feedback entry count overflow"))?;
            if bytes > evidence::MAXIMUM_BYTES || files > evidence::MAXIMUM_FILES {
                return Err(Error::new(
                    ErrorKind::InputLimit,
                    "feedback export evidence budget exceeded",
                ));
            }
            if item.is_candidate() {
                candidate.insert(item.family.clone());
            } else {
                infrastructure.insert(item.family.clone());
            }
            admitted.insert(item.family.clone(), item);
        }
        let producer = &context.package.identity;
        let payload = Payload {
            schema: 2,
            identity_sha256: context.package.identity_sha256.clone(),
            candidate: producer.candidate.clone(),
            source_pass: producer.pass_id.clone(),
            run_id: producer.run_id.clone(),
            run_attempt: context.current.run_attempt.clone(),
            state,
            repairable: state == FeedbackState::CandidateRepairable,
            failure_class: state.failure_class().to_owned(),
            failure_stage: "family-certification".into(),
            candidate_failures: candidate.iter().cloned().collect(),
            infrastructure_failures: infrastructure.iter().cloned().collect(),
            failed_families: candidate.union(&infrastructure).cloned().collect(),
            evidence_sha256: admitted
                .iter()
                .map(|(family, item)| (family.clone(), item.snapshot.files.clone()))
                .collect(),
            errors,
        };
        payload.validate(context, state)?;
        Ok(Self {
            payload,
            evidence: admitted,
        })
    }
    pub(crate) fn publish(
        self,
        context: &ReceiptContext,
        destination: &Path,
    ) -> Result<PublishedFeedback, Error> {
        export::publish(context, self, destination)
    }
}

pub(crate) struct PublishedFeedback {
    directory: PathBuf,
    state: FeedbackState,
}
impl PublishedFeedback {
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }
    pub(crate) fn state(&self) -> FeedbackState {
        self.state
    }
}

pub(crate) struct VerifiedFeedback {
    payload: Payload,
    snapshot: Snapshot,
}
impl VerifiedFeedback {
    /// Render a bounded repair index from this admitted, owned evidence snapshot.
    pub(crate) fn summary(&self) -> Result<String, Error> {
        summary::render(self)
    }
    /// The owned snapshot must outlive consumers of this immutable directory.
    pub(crate) fn directory(&self) -> &Path {
        self.snapshot.root()
    }
    pub(crate) fn state(&self) -> FeedbackState {
        self.payload.state
    }
    pub(crate) fn candidate_failures(&self) -> &[Family] {
        &self.payload.candidate_failures
    }
    pub(crate) fn infrastructure_failures(&self) -> &[Family] {
        &self.payload.infrastructure_failures
    }
    pub(crate) fn missing_infrastructure(&self) -> BTreeSet<Family> {
        self.payload
            .infrastructure_failures
            .iter()
            .filter(|family| !self.payload.evidence_sha256.contains_key(*family))
            .cloned()
            .collect()
    }
    pub(crate) fn candidate_evidence(
        &self,
        context: &ReceiptContext,
    ) -> Result<Vec<FamilyEvidence>, Error> {
        self.payload.validate(context, self.payload.state)?;
        self.payload
            .candidate_failures
            .iter()
            .map(|family| {
                FamilyEvidence::admit(context, family, &self.snapshot.root().join(family.as_str()))
            })
            .collect()
    }
}

pub(crate) fn verify(
    context: &ReceiptContext,
    directory: &Path,
    expected: FeedbackState,
) -> Result<VerifiedFeedback, Error> {
    let snapshot = Snapshot::capture(directory)?;
    let payload: Payload =
        serde_json::from_slice(&snapshot.read_metadata(Path::new("feedback.json"))?)?;
    payload.validate(context, expected)?;
    verify_closure(context, &payload, &snapshot)?;
    Ok(VerifiedFeedback { payload, snapshot })
}

fn verify_closure(
    context: &ReceiptContext,
    payload: &Payload,
    snapshot: &Snapshot,
) -> Result<(), Error> {
    let mut expected_files = BTreeMap::from([(
        "feedback.json".to_owned(),
        snapshot
            .files
            .get("feedback.json")
            .cloned()
            .ok_or_else(|| contract_error("feedback metadata missing"))?,
    )]);
    let mut expected_directories = BTreeSet::new();
    for (family, manifest) in &payload.evidence_sha256 {
        if manifest.is_empty() {
            return Err(contract_error("feedback evidence tree must not be empty"));
        }
        expected_directories.insert(family.to_string());
        for (path, digest) in manifest {
            evidence::safe_relative(Path::new(path))?;
            let full = format!("{family}/{path}");
            let mut ancestor = Path::new(&full).parent();
            while let Some(path) = ancestor.filter(|path| !path.as_os_str().is_empty()) {
                expected_directories.insert(evidence::safe_relative(path)?);
                ancestor = path.parent();
            }
            expected_files.insert(full, digest.clone());
        }
        let admitted =
            FamilyEvidence::admit(context, family, &snapshot.root().join(family.as_str()))?;
        if admitted.snapshot.files != *manifest
            || admitted.is_candidate() != payload.candidate_failures.contains(family)
        {
            return Err(contract_error(
                "feedback evidence digest or failure class mismatch",
            ));
        }
        let receipt = admitted_receipt(context, family, &admitted.snapshot)?;
        if receipt.provenance.run_attempt > payload.run_attempt {
            return Err(contract_error(
                "feedback includes evidence from a later attempt",
            ));
        }
    }
    if snapshot.files != expected_files || snapshot.directories != expected_directories {
        return Err(contract_error(
            "feedback declared evidence closure mismatch",
        ));
    }
    Ok(())
}

fn contract_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ReceiptIdentity, message)
}

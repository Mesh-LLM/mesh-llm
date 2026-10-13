use super::super::{Digest, Error, Family, ReceiptContext, RunAttempt};
use super::{FeedbackState, contract_error};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Payload {
    pub(super) schema: u8,
    pub(super) identity_sha256: Digest,
    pub(super) candidate: String,
    pub(super) source_pass: String,
    pub(super) run_id: String,
    pub(super) run_attempt: RunAttempt,
    pub(super) state: FeedbackState,
    pub(super) repairable: bool,
    pub(super) failure_class: String,
    pub(super) failure_stage: String,
    pub(super) candidate_failures: Vec<Family>,
    pub(super) infrastructure_failures: Vec<Family>,
    pub(super) failed_families: Vec<Family>,
    pub(super) evidence_sha256: BTreeMap<Family, BTreeMap<String, Digest>>,
    pub(super) errors: Vec<String>,
}

pub(super) fn safe_family(family: &Family) -> Result<(), Error> {
    if matches!(family.as_str(), "." | "..") {
        return Err(contract_error("feedback family must name a direct child"));
    }
    Ok(())
}

impl Payload {
    pub(super) fn validate(
        &self,
        context: &ReceiptContext,
        expected: FeedbackState,
    ) -> Result<(), Error> {
        let producer = &context.package.identity;
        if self.schema != 2
            || self.identity_sha256 != context.package.identity_sha256
            || self.candidate != producer.candidate
            || self.source_pass != producer.pass_id
            || self.run_id != producer.run_id
            || self.failure_stage != "family-certification"
            || self.state != expected
        {
            return Err(contract_error("feedback producer or state mismatch"));
        }
        if self.run_attempt < producer.run_attempt || self.run_attempt > context.current.run_attempt
        {
            return Err(contract_error(
                "feedback attempt outside producer/current bounds",
            ));
        }
        self.failure_sets(context)?;
        let candidate: BTreeSet<_> = self.candidate_failures.iter().collect();
        let infrastructure: BTreeSet<_> = self.infrastructure_failures.iter().collect();
        if !candidate.is_disjoint(&infrastructure)
            || self.repairable != (self.state == FeedbackState::CandidateRepairable)
            || self.failure_class != self.state.failure_class()
        {
            return Err(contract_error("feedback failure classes disagree"));
        }
        match self.state {
            FeedbackState::CandidateRepairable
                if candidate.is_empty() || !infrastructure.is_empty() =>
            {
                return Err(contract_error(
                    "candidate feedback must contain only candidate failures",
                ));
            }
            FeedbackState::InfrastructureRetryable if infrastructure.is_empty() => {
                return Err(contract_error(
                    "infrastructure feedback has no retry families",
                ));
            }
            _ => {}
        }
        let union: BTreeSet<_> = candidate.union(&infrastructure).copied().collect();
        if self.failed_families.iter().collect::<Vec<_>>()
            != union.iter().copied().collect::<Vec<_>>()
        {
            return Err(contract_error("feedback failed-family union mismatch"));
        }
        let keys: BTreeSet<_> = self.evidence_sha256.keys().collect();
        if !candidate.is_subset(&keys) || !keys.is_subset(&union) {
            return Err(contract_error("feedback evidence family manifest mismatch"));
        }
        Ok(())
    }

    fn failure_sets(&self, context: &ReceiptContext) -> Result<(), Error> {
        for families in [&self.candidate_failures, &self.infrastructure_failures] {
            if families.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(contract_error(
                    "feedback families must be sorted and unique",
                ));
            }
            for family in families {
                safe_family(family)?;
                context.model(family)?;
            }
        }
        Ok(())
    }
}

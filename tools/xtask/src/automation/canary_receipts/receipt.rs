use super::storage::{RESULTS_LIMIT, read_bounded};
use super::{Digest, Error, ErrorKind, Family, ReceiptContext, RunAttempt};
use crate::ci_plan::plan_bytes::write_string;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WorkerOutcome {
    Success,
    Failure,
    Cancelled,
    Skipped,
}

impl WorkerOutcome {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Cancelled => "cancelled",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(remote = "Self")]
pub(crate) struct ReceiptProvenance {
    pub(crate) candidate: String,
    pub(crate) family: Family,
    pub(crate) identity_sha256: Digest,
    pub(crate) pass_id: String,
    pub(crate) run_attempt: RunAttempt,
    pub(crate) run_id: String,
}

impl<'de> Deserialize<'de> for ReceiptProvenance {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::boundary::ordered_object_last_wins(deserializer, Self::deserialize)
    }
}

#[derive(Debug, Deserialize)]
#[serde(remote = "Self")]
pub(crate) struct WorkerReceipt {
    #[serde(flatten)]
    pub(crate) provenance: ReceiptProvenance,
    pub(crate) outcome: WorkerOutcome,
    pub(crate) results_sha256: Option<Digest>,
    #[serde(default = "unknown_runner")]
    pub(crate) runner: String,
}

impl<'de> Deserialize<'de> for WorkerReceipt {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        super::boundary::ordered_object_last_wins(deserializer, Self::deserialize)
    }
}

fn unknown_runner() -> String {
    "unknown".to_owned()
}

pub(crate) struct WorkerResult {
    pub(crate) family: Family,
    pub(crate) outcome: WorkerOutcome,
    pub(crate) runner: Option<String>,
}

pub(crate) fn write_receipt(
    context: &ReceiptContext,
    evidence: &Path,
    worker: WorkerResult,
) -> Result<WorkerReceipt, Error> {
    context.model(&worker.family)?;
    fs::create_dir_all(evidence)?;
    let results = evidence.join("results.jsonl");
    let results_sha256 = if results.is_file() {
        Some(Digest::of_bytes(&read_bounded(&results, RESULTS_LIMIT)?))
    } else {
        None
    };
    let identity = &context.package.identity;
    let provenance = ReceiptProvenance {
        candidate: identity.candidate.clone(),
        family: worker.family,
        identity_sha256: context.package.identity_sha256.clone(),
        pass_id: identity.pass_id.clone(),
        run_attempt: context.current.run_attempt.clone(),
        run_id: context.current.run_id.clone(),
    };
    let receipt = WorkerReceipt {
        provenance,
        outcome: worker.outcome,
        results_sha256,
        runner: worker.runner.unwrap_or_else(unknown_runner),
    };
    fs::write(evidence.join("receipt.json"), receipt.render())?;
    Ok(receipt)
}

impl ReceiptProvenance {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, Error> {
        if bytes
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            != Some(b'{')
        {
            return Err(Error::new(
                ErrorKind::Json,
                "expected a worker receipt object",
            ));
        }
        Ok(serde_json::from_slice(bytes)?)
    }

    pub(super) fn validate_identity(&self, context: &ReceiptContext) -> Result<(), Error> {
        context.model(&self.family)?;
        let identity = &context.package.identity;
        if self.identity_sha256 != context.package.identity_sha256
            || self.candidate != identity.candidate
            || self.pass_id != identity.pass_id
            || self.run_id != identity.run_id
        {
            return Err(Error::new(
                ErrorKind::ReceiptIdentity,
                "mismatched worker receipt",
            ));
        }
        if self.run_attempt < identity.run_attempt || self.run_attempt > context.current.run_attempt
        {
            return Err(Error::new(
                ErrorKind::AttemptBounds,
                "worker attempt outside producer/current bounds",
            ));
        }
        Ok(())
    }
}

impl WorkerReceipt {
    pub(crate) fn render(&self) -> String {
        let identity = &self.provenance;
        let fields = [
            ("candidate", Some(identity.candidate.as_str())),
            ("family", Some(identity.family.as_str())),
            ("identity_sha256", Some(identity.identity_sha256.as_str())),
            ("outcome", Some(self.outcome.as_str())),
            ("pass_id", Some(identity.pass_id.as_str())),
            (
                "results_sha256",
                self.results_sha256.as_ref().map(Digest::as_str),
            ),
            ("run_attempt", Some(identity.run_attempt.as_str())),
            ("run_id", Some(identity.run_id.as_str())),
            ("runner", Some(self.runner.as_str())),
        ];
        let mut output = String::from("{");
        for (index, (name, value)) in fields.into_iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            output.push_str("\n  ");
            write_string(&mut output, name);
            output.push_str(": ");
            match value {
                Some(text) => write_string(&mut output, text),
                None => output.push_str("null"),
            }
        }
        output.push_str("\n}\n");
        output.replace('\u{7f}', "\\u007f")
    }
}

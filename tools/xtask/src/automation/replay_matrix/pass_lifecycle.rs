use crate::command::DynResult;
use crate::process::{
    Outcome,
    retained::{MemberId, Report},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize, Serialize)]
pub(super) struct Receipt {
    pub infrastructure_clean: bool,
    pub worker_status: Option<i32>,
}

pub(super) fn retain(report: &Report<String>, directory: &Path) -> DynResult<()> {
    let worker_status = report
        .members
        .iter()
        .find(|member| member.member == MemberId::WorkerOne)
        .and_then(|member| member.process.status.as_ref())
        .and_then(std::process::ExitStatus::code);
    let infrastructure_clean = report.failure.is_none()
        && report.rejection.is_none()
        && !matches!(
            report.outcome,
            Outcome::Deadline
                | Outcome::ReadinessDeadline
                | Outcome::Cancelled
                | Outcome::IoFailure
        )
        && report.members.len() == 2
        && report.members.iter().all(|member| {
            let process = &member.process;
            process.cleanup.complete
                && !process.cleanup.forced
                && !process.cleanup.graceful_signal_failed
                && process.cleanup.failure.is_none()
                && process.failure.is_none()
                && process.status.is_some()
                && !matches!(
                    process.outcome,
                    Outcome::Deadline
                        | Outcome::ReadinessDeadline
                        | Outcome::Cancelled
                        | Outcome::IoFailure
                )
                && (member.member != MemberId::Seed
                    || !matches!(
                        member.disposition,
                        crate::process::retained::Disposition::Failure(_)
                    ))
        });
    crate::command::write_json_file(
        &directory.join("lifecycle.json"),
        &Receipt {
            infrastructure_clean,
            worker_status,
        },
    )
}

pub(super) fn acceptance_exit(directory: &Path) -> DynResult<bool> {
    let path = directory.join("lifecycle.json");
    if !path.try_exists()? {
        return Ok(false);
    }
    let receipt: Receipt = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(receipt.infrastructure_clean && matches!(receipt.worker_status, Some(0 | 2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infrastructure_failure_never_becomes_acceptance_only() {
        let state = tempfile::tempdir().unwrap();
        crate::command::write_json_file(
            &state.path().join("lifecycle.json"),
            &Receipt {
                infrastructure_clean: false,
                worker_status: Some(2),
            },
        )
        .unwrap();
        assert!(!acceptance_exit(state.path()).unwrap());
    }

    #[test]
    fn unknown_worker_exit_never_becomes_acceptance_only() {
        let state = tempfile::tempdir().unwrap();
        crate::command::write_json_file(
            &state.path().join("lifecycle.json"),
            &Receipt {
                infrastructure_clean: true,
                worker_status: Some(7),
            },
        )
        .unwrap();
        assert!(!acceptance_exit(state.path()).unwrap());
    }
}

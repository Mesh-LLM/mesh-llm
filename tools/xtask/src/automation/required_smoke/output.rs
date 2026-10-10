use super::{Completed, Model, Stack};
use crate::process::ProcessReport;

pub(super) fn reject(error: &(dyn std::error::Error + 'static)) -> crate::command::DynResult<()> {
    use std::io::Write;
    #[derive(serde::Serialize)]
    struct Failed {
        schema: u32,
        owner: &'static str,
        status: &'static str,
        reason: String,
        processes: Vec<Process>,
    }
    let reason = reason(error);
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(
        &mut output,
        &Failed {
            schema: 1,
            owner: "required_smoke",
            status: "rejected",
            reason,
            processes: reports(error).into_iter().map(Process::new).collect(),
        },
    )?;
    writeln!(output)?;
    Ok(())
}

pub(super) fn reason(error: &(dyn std::error::Error + 'static)) -> String {
    if let Some(failure) = error.downcast_ref::<super::failure::Failure>() {
        return failure.reason();
    }
    match error.downcast_ref::<super::Rejected>() {
        Some(rejected) => rejected.reason.to_string(),
        None => match error.downcast_ref::<super::Rejection>() {
            Some(reason) => reason.to_string(),
            None => "orchestration_failed".into(),
        },
    }
}

pub(super) fn reports<'a>(error: &'a (dyn std::error::Error + 'static)) -> Vec<&'a ProcessReport> {
    if let Some(failure) = error.downcast_ref::<super::failure::Failure>() {
        return failure.reports();
    }
    match error.downcast_ref::<super::Rejected>() {
        Some(rejected) => {
            let mut reports = vec![&rejected.primary];
            reports.extend(rejected.headless.as_ref());
            reports
        }
        None => Vec::new(),
    }
}

#[derive(Debug, serde::Serialize)]
pub(super) struct Receipt {
    schema: u32,
    owner: &'static str,
    artifact_id: &'static str,
    model_class: &'static str,
    stack: &'static str,
    status: &'static str,
    processes: [Process; 2],
    #[serde(skip)]
    completed: Completed,
}

#[derive(Debug, serde::Serialize)]
struct Process {
    pid: u32,
    exit_code: Option<i32>,
    cleanup_complete: bool,
    forced: bool,
    probe_admitted: bool,
}

impl Receipt {
    pub(super) fn new(completed: Completed) -> Self {
        let variant = completed.variant();
        let (primary, headless) = completed.process_reports();
        Self {
            schema: 1,
            owner: "required_smoke",
            status: "passed",
            artifact_id: variant.model.artifact_id(),
            model_class: match variant.model {
                Model::Dense => "dense",
                Model::Recurrent => "recurrent",
            },
            stack: match variant.stack {
                Stack::Default => "default",
                Stack::Constrained => "constrained",
            },
            processes: [Process::new(primary), Process::new(headless)],
            completed,
        }
    }

    pub(super) fn reports(&self) -> Vec<&ProcessReport> {
        let (primary, headless) = self.completed.process_reports();
        vec![primary, headless]
    }
}

impl Process {
    fn new(report: &ProcessReport) -> Self {
        Self {
            pid: report.pid,
            exit_code: report.status.and_then(|status| status.code()),
            cleanup_complete: report.cleanup.complete,
            forced: report.cleanup.forced,
            probe_admitted: matches!(
                report.readiness_stop,
                crate::process::ReadinessStop::ProbeAdmitted { .. }
            ),
        }
    }
}

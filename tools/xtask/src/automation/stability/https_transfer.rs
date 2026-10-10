use super::{
    Curl, Failure, Reply, Request, command, failure,
    response::{Observer, Reader},
};
use crate::process::{
    Cancellation, GracefulRequest, Outcome, OutputFiles, Probe, ProcessReport, ReadinessStop,
    supervise_with_probe,
};
use std::{fs::File, io::Write};

pub(super) fn run(
    curl: &Curl,
    request: Request,
    cancellation: &Cancellation,
    local: &Cancellation,
) -> Result<Reply, Failure> {
    if cancellation.is_cancelled() || local.is_cancelled() {
        return Err(failure("stability operation cancelled", None));
    }
    let directory = tempfile::Builder::new()
        .prefix("mesh-stability-https-")
        .tempdir()
        .map_err(|_| failure("private HTTPS directory unavailable", None))?;
    let directory_path = directory
        .path()
        .canonicalize()
        .map_err(|_| failure("private HTTPS directory unavailable", None))?;
    let body = tempfile::NamedTempFile::new_in(&directory_path)
        .map_err(|_| failure("private HTTPS body unavailable", None))?;
    let headers = tempfile::NamedTempFile::new_in(&directory_path)
        .map_err(|_| failure("private HTTPS headers unavailable", None))?;
    let mut payload = request
        .body
        .as_ref()
        .map(|_| tempfile::NamedTempFile::new_in(&directory_path))
        .transpose()
        .map_err(|_| failure("private HTTPS payload unavailable", None))?;
    if let (Some(bytes), Some(file)) = (&request.body, &mut payload) {
        file.write_all(bytes)
            .map_err(|_| failure("HTTPS payload write failed", None))?;
        file.flush()
            .map_err(|_| failure("HTTPS payload flush failed", None))?;
    }
    let budget = request.timeout.saturating_sub(request.started.elapsed());
    if budget.is_zero() {
        return Err(failure("stability request deadline exceeded", None));
    }
    let spec = curl.specification(
        &crate::process::curl_https::Request {
            endpoint: request.endpoint.clone(),
            method: request.method.clone(),
            token: request.token,
        },
        command::Files {
            directory: &directory_path,
            body: body.path(),
            headers: headers.path(),
            payload: payload.as_ref().map(|file| file.path()),
        },
        budget,
    );
    let mut reader = Reader::start(
        File::open(headers.path()).map_err(|_| failure("HTTPS headers unreadable", None))?,
        File::open(body.path()).map_err(|_| failure("HTTPS body unreadable", None))?,
        &request,
    )?;
    let mut observer = Observer {
        observations: &reader.observations,
        cancellation,
    };
    let report = supervise_with_probe(
        &spec,
        &command::limits(budget),
        local,
        OutputFiles::default(),
        Probe {
            observer: &mut observer,
            deadline: budget,
        },
    );
    // The process owner stops and reaps before the final reader scan and join.
    // Private files outlive both owners.
    let reply = reader.finish();
    let report = report.map_err(|_| failure("stability HTTPS process failed", None))?;
    let status = reply
        .as_ref()
        .ok()
        .map(|value| value.status)
        .or_else(|| reply.as_ref().err().and_then(|error| error.status));
    if cancellation.is_cancelled() || local.is_cancelled() {
        return Err(failure("stability operation cancelled", status));
    }
    if !clean(&report.process) {
        return Err(failure("stability HTTPS cleanup failed", status));
    }
    if matches!(
        report.process.outcome,
        Outcome::Deadline | Outcome::ReadinessDeadline
    ) || report
        .process
        .status
        .is_some_and(|value| value.code() == Some(28))
    {
        return Err(failure("stability request deadline exceeded", status));
    }
    if let Some(rejection) = report.rejection {
        return Err(rejection);
    }
    if report
        .process
        .status
        .is_some_and(|value| value.code() == Some(63))
    {
        return Err(failure("stability response exceeds 16 MiB", status));
    }
    if report
        .process
        .status
        .is_some_and(|value| value.code() == Some(60))
    {
        return Err(failure("HTTPS certificate verification failed", status));
    }
    let exited = report.process.status.is_some_and(|value| value.success())
        && matches!(
            report.process.outcome,
            Outcome::Exited | Outcome::EarlyExit | Outcome::Ready
        );
    let stopped_on_done = request.stream
        && report.process.outcome == Outcome::Ready
        && matches!(
            report.process.readiness_stop,
            ReadinessStop::ProbeAdmitted {
                request: GracefulRequest::RequestedAfterLiveObservation,
                ..
            }
        );
    if !exited && !stopped_on_done {
        return Err(failure("stability HTTPS request failed", status));
    }
    reply
}

fn clean(report: &ProcessReport) -> bool {
    report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && report.failure.is_none()
}

use super::{
    Rejection, Session, args::Options, execution, failure::Failure, output, overlap::Overlap,
    workers::Workers,
};
use crate::{
    automation::{
        daemon_readiness::ports::Reservation,
        private_state::{FinishError, PrivateState},
    },
    command::DynResult,
    process::{
        Cancellation, Completion, Limits, Readiness,
        retained::{self, Launch, MemberId},
    },
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(super) fn execute(
    run: (&Path, &Options, &Cancellation),
    mut session: Session,
    started: Instant,
) -> DynResult<output::Receipt> {
    let (root, options, cancellation) = run;
    let primary_state = PrivateState::create(&options.parent, "required-smoke")?;
    let mut headless_state = None;
    let result = (|| {
        primary_state.prepare()?;
        primary_state.model_fit(options.batch_sizes.or(options.variant.model.batch_sizes()))?;
        headless_state = Some(PrivateState::create(&options.parent, "required-smoke")?);
        let state = headless_state.as_ref().ok_or(Rejection::Incomplete)?;
        state.prepare()?;
        state.model_fit(options.batch_sizes.or(options.variant.model.batch_sizes()))?;
        let primary_ports = match options.endpoints {
            Some(endpoints) => Reservation::acquire_at(endpoints[0])?,
            None => Reservation::acquire()?,
        };
        let headless_ports = match options.endpoints {
            Some(endpoints) => Reservation::acquire_at(endpoints[1])?,
            None => Reservation::acquire()?,
        };
        let deadline = options.readiness + Duration::from_secs(300);
        let primary_launch = Launch {
            member: MemberId::Seed,
            spec: execution::spec(root, options, (&primary_state, primary_ports.ports, false)),
            files: primary_state.output_files(),
            readiness_deadline: deadline,
        };
        let headless_launch = Launch {
            member: MemberId::WorkerOne,
            spec: execution::spec(root, options, (state, headless_ports.ports, true)),
            files: state.output_files(),
            readiness_deadline: options.readiness,
        };
        let report = std::thread::scope(|scope| -> DynResult<_> {
            let (primary, primary_workers) = Workers::start(
                scope,
                (options, primary_ports.ports, false, started),
                cancellation,
            )?;
            let headless = Workers::start(
                scope,
                (options, headless_ports.ports, true, started),
                cancellation,
            );
            let (headless, headless_workers) = match headless {
                Ok(workers) => workers,
                Err(source) => {
                    drop(primary);
                    let joined = primary_workers.join();
                    if joined.0 && joined.1 {
                        return Err(source);
                    }
                    return Err(Failure::CoordinatedWorkers {
                        preceding: Err(source),
                        joined: vec![joined],
                    }
                    .into());
                }
            };
            let mut owner = Overlap {
                session: &mut session,
                primary,
                headless,
                primary_launch: Some(primary_launch),
                headless_launch: Some(headless_launch),
                headless_clock: false,
                started,
            };
            drop(primary_ports);
            drop(headless_ports);
            let result = retained::run(
                &mut owner,
                &Limits {
                    execution: deadline + options.readiness,
                    graceful_shutdown: options.shutdown,
                    forced_shutdown: Duration::from_secs(5),
                    retained_bytes_per_stream: 65536,
                    readiness: Readiness::None,
                    completion: Completion::Exit,
                },
                cancellation,
            );
            drop(owner);
            let primary_joined = primary_workers.join();
            let headless_joined = headless_workers.join();
            finish_workers(result, vec![primary_joined, headless_joined])
        })?;
        finish_report(session, report, cancellation)
    })();
    if result.is_err() {
        super::diagnostics::native_logs(&primary_state);
        if let Some(state) = &headless_state {
            super::diagnostics::native_logs(state);
        }
    }
    let result = match headless_state {
        Some(state) => finish_state(state, result),
        None => result,
    };
    let result = finish_state(primary_state, result);
    if cancellation.is_cancelled() {
        return Err(Failure::Interrupt {
            reason: crate::automation::command_interrupt::Reason::Interrupted,
            preceding: result,
        }
        .into());
    }
    result
}

pub(super) fn finish_report(
    session: Session,
    report: retained::Report<Rejection>,
    cancellation: &Cancellation,
) -> DynResult<output::Receipt> {
    if report.failure.is_some()
        || report
            .members
            .iter()
            .filter(|member| member.member == MemberId::Seed)
            .count()
            != 1
        || report
            .members
            .iter()
            .any(|member| ![MemberId::Seed, MemberId::WorkerOne].contains(&member.member))
    {
        return Err(Failure::Coordinated { report }.into());
    }
    let result = if cancellation.is_cancelled() {
        Err(Rejection::Cancelled)
    } else {
        report.rejection.map_or_else(
            || match report.outcome {
                crate::process::Outcome::Ready => Ok(()),
                crate::process::Outcome::Exited | crate::process::Outcome::EarlyExit => {
                    Err(Rejection::EarlyExit)
                }
                crate::process::Outcome::Deadline | crate::process::Outcome::ReadinessDeadline => {
                    Err(Rejection::ProcessDeadline)
                }
                crate::process::Outcome::Cancelled => Err(Rejection::Cancelled),
                crate::process::Outcome::IoFailure
                    if session.completion().is_ok()
                        && report.members.iter().any(|member| {
                            member.process.cleanup.forced
                                || !member.process.cleanup.complete
                                || member.process.cleanup.graceful_signal_failed
                                || member.process.cleanup.failure.is_some()
                                || member
                                    .process
                                    .status
                                    .is_none_or(|status| status.code() != Some(0))
                        }) =>
                {
                    Err(Rejection::ProcessCleanup)
                }
                crate::process::Outcome::IoFailure
                | crate::process::Outcome::ObservationRejected => Err(Rejection::ProcessFailure),
            },
            Err,
        )
    };
    let mut primary = None;
    let mut headless = None;
    for member in report.members {
        match member.member {
            MemberId::Seed => primary = Some(member.process),
            MemberId::WorkerOne => headless = Some(member.process),
            _ => return Err(Rejection::ProcessIdentity.into()),
        }
    }
    let primary = primary.ok_or(Rejection::Incomplete)?;
    let completed = session
        .finish((primary, headless), result)
        .map_err(|error| -> Box<dyn std::error::Error> { error })?;
    Ok(output::Receipt::new(completed))
}

pub(super) fn finish_workers(
    result: Result<retained::Report<Rejection>, crate::process::Failure>,
    joined: Vec<(bool, bool)>,
) -> DynResult<retained::Report<Rejection>> {
    let preceding = result.map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) });
    if joined.iter().any(|(daemon, smoke)| !daemon || !smoke) {
        return Err(Failure::CoordinatedWorkers { preceding, joined }.into());
    }
    preceding
}

pub(super) fn finish_state(
    state: PrivateState,
    result: DynResult<output::Receipt>,
) -> DynResult<output::Receipt> {
    let mut retained = None;
    match state.finish(result.map(|receipt| retained = Some(receipt))) {
        Ok(()) => retained.ok_or_else(|| Rejection::Incomplete.into()),
        Err(FinishError::Prior(error)) => Err(error),
        Err(FinishError::Deletion {
            kind,
            code,
            preceding,
        }) => Err(Failure::CoordinatedState {
            kind,
            code,
            preceding: match preceding {
                Some(error) => Err(*error),
                None => retained.ok_or_else(|| Rejection::Incomplete.into()),
            },
        }
        .into()),
    }
}

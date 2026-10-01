use super::{Action, Context, Coordinator, Report, member::Member, shutdown};
use crate::process::probe::ProbeLines;
use crate::process::{
    Cancellation, Completion, Failure, Limits, Outcome, ProbeDecision, Readiness, control,
};
use std::time::Instant;
pub fn run<Observer: Coordinator>(
    observer: &mut Observer,
    limits: &Limits,
    cancellation: &Cancellation,
) -> Result<Report<Observer::Rejection>, Failure> {
    limits.validate()?;
    match (&limits.readiness, limits.completion) {
        (Readiness::None, Completion::Exit) => (),
        (Readiness::None, Completion::StopAfterReady)
        | (Readiness::Line { .. } | Readiness::ObservedLines { .. }, _) => {
            return Err(Failure::InvalidSpec(
                "retained session requires None/Exit limits",
            ));
        }
    }
    if cancellation.is_cancelled() {
        return Err(Failure::InvalidSpec("cancelled before retained session"));
    }
    let started = Instant::now();
    let mut members = Vec::<Member>::with_capacity(super::MAX_CONCURRENT_MEMBERS);
    let mut rejection = None;
    let mut failure = None;
    let mut outcome = loop {
        if let Some(outcome) = expired(started, limits, cancellation) {
            break outcome;
        }
        match observe(
            &mut members,
            observer,
            (started, limits, cancellation),
            &mut rejection,
        ) {
            Ok(Some(outcome)) => break outcome,
            Ok(None) => (),
            Err(error) => {
                failure = Some(error);
                break Outcome::IoFailure;
            }
        }
        let snapshots: Vec<_> = members.iter().map(Member::snapshot).collect();
        let elapsed = started.elapsed();
        let action = observer.tick(Context {
            elapsed,
            remaining: limits.execution.saturating_sub(elapsed),
            members: &snapshots,
        });
        if let Some(outcome) = expired(started, limits, cancellation) {
            break outcome;
        }
        if !matches!(action, Action::Complete) {
            match observe(
                &mut members,
                observer,
                (started, limits, cancellation),
                &mut rejection,
            ) {
                Ok(Some(outcome)) => break outcome,
                Ok(None) => (),
                Err(error) => {
                    failure = Some(error);
                    break Outcome::IoFailure;
                }
            }
        }
        match action {
            Action::Pending => std::thread::sleep(control::POLL),
            Action::Admit(id) => match admit(&mut members, id) {
                Ok(Some(outcome)) => break outcome,
                Ok(None) => (),
                Err(error) => {
                    failure = Some(error);
                    break Outcome::IoFailure;
                }
            },
            Action::Start(launch) => {
                if let Err(error) = start(&mut members, launch, limits) {
                    failure = Some(error);
                    break Outcome::IoFailure;
                }
            }
            Action::Stop(id) => match shutdown::stop(&mut members, id, limits, |survivors| {
                match expired(started, limits, cancellation) {
                    Some(outcome) => Ok(Some(outcome)),
                    None => observe(
                        survivors,
                        observer,
                        (started, limits, cancellation),
                        &mut rejection,
                    ),
                }
            }) {
                Ok(Some(outcome)) => break outcome,
                Ok(None) => (),
                Err(error) => {
                    failure = Some(error);
                    break Outcome::IoFailure;
                }
            },
            Action::Complete => {
                if members.is_empty() || members.iter().any(|member| member.admitted.is_none()) {
                    failure = Some(Failure::InvalidSpec("completion requires admitted members"));
                    break Outcome::IoFailure;
                }
                break Outcome::Ready;
            }
            Action::Reject(reason) => {
                rejection = Some(reason);
                break Outcome::ObservationRejected;
            }
        }
    };
    shutdown::finish(
        &mut members,
        (started, limits, cancellation),
        (&mut outcome, &mut failure),
    );
    Ok(Report {
        outcome,
        rejection,
        failure,
        members: members
            .into_iter()
            .filter_map(|member| member.report)
            .collect(),
    })
}
fn start(members: &mut Vec<Member>, launch: super::Launch, limits: &Limits) -> Result<(), Failure> {
    if members
        .iter()
        .filter(|member| member.report.is_none())
        .count()
        >= super::MAX_CONCURRENT_MEMBERS
    {
        return Err(Failure::RetainedMemberLimit {
            limit: super::MAX_CONCURRENT_MEMBERS,
        });
    }
    let previous = members
        .iter()
        .rev()
        .find(|member| member.id.name() == launch.member.name());
    match previous {
        Some(member)
            if member.report.as_ref().is_some_and(|report| {
                matches!(report.disposition, super::Disposition::IntentionalStop)
            }) && member.id.next_generation()? == launch.member => {}
        None if launch.member.generation() == 0 => (),
        Some(_) | None => {
            return Err(Failure::InvalidSpec(
                "duplicate or invalid retained generation",
            ));
        }
    }
    members.push(Member::spawn(launch, limits)?);
    Ok(())
}
fn admit(members: &mut [Member], id: super::MemberId) -> Result<Option<Outcome>, Failure> {
    let member = members
        .iter_mut()
        .find(|member| member.id == id && member.report.is_none() && member.admitted.is_none())
        .ok_or(Failure::InvalidSpec("admit requires a starting member"))?;
    let elapsed = member.started.elapsed();
    if elapsed >= member.deadline {
        return Ok(Some(Outcome::ReadinessDeadline));
    }
    member.admitted = Some(elapsed);
    Ok(None)
}
pub(super) fn expired(
    started: Instant,
    limits: &Limits,
    cancellation: &Cancellation,
) -> Option<Outcome> {
    if cancellation.is_cancelled() {
        Some(Outcome::Cancelled)
    } else if started.elapsed() >= limits.execution {
        Some(Outcome::Deadline)
    } else {
        None
    }
}
fn observe<Observer: Coordinator>(
    members: &mut [Member],
    observer: &mut Observer,
    context: (Instant, &Limits, &Cancellation),
    rejection: &mut Option<Observer::Rejection>,
) -> Result<Option<Outcome>, Failure> {
    let (started, limits, cancellation) = context;
    for member in members {
        let Some(output) = member.output.as_mut() else {
            continue;
        };
        if let Some(outcome) = expired(started, limits, cancellation) {
            return Ok(Some(outcome));
        }
        if member.admitted.is_none() && member.started.elapsed() >= member.deadline {
            return Ok(Some(Outcome::ReadinessDeadline));
        }
        let mut candidate = false;
        output.poll_probe(&mut |line| {
            if rejection.is_none()
                && (!candidate || member.admitted.is_some())
                && (member.admitted.is_some() || member.started.elapsed() < member.deadline)
                && expired(started, limits, cancellation).is_none()
            {
                match observer.line(member.id, line) {
                    ProbeDecision::Pending => (),
                    ProbeDecision::Candidate => candidate = true,
                    ProbeDecision::Rejected(reason) => *rejection = Some(reason),
                }
            }
        })?;
        if let Some(outcome) = expired(started, limits, cancellation) {
            return Ok(Some(outcome));
        }
        if member.admitted.is_none() && member.started.elapsed() >= member.deadline {
            return Ok(Some(Outcome::ReadinessDeadline));
        }
        if rejection.is_some() {
            return Ok(Some(Outcome::ObservationRejected));
        }
        let exited = member.child.exited()?;
        if let Some(outcome) = expired(started, limits, cancellation) {
            return Ok(Some(outcome));
        }
        let elapsed = member.started.elapsed();
        if member.admitted.is_none() && elapsed >= member.deadline {
            return Ok(Some(Outcome::ReadinessDeadline));
        }
        if exited {
            return Ok(Some(Outcome::EarlyExit));
        }
        if candidate && member.admitted.is_none() {
            member.admitted = Some(elapsed);
        }
    }
    Ok(None)
}

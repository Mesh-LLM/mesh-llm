use super::{Coordinator, Disposition, member::Member, owner::expired};
use crate::process::{Cancellation, Failure, Limits, Outcome, ProbeDecision, probe::ProbeLines};
use std::time::Instant;

pub(super) fn observe<Observer: Coordinator>(
    members: &mut [Member],
    observer: &mut Observer,
    context: (Instant, &Limits, &Cancellation),
    rejection: &mut Option<Observer::Rejection>,
) -> Result<Option<Outcome>, Failure> {
    let (started, limits, cancellation) = context;
    for index in 0..members.len() {
        let member = &mut members[index];
        if member.output.is_none() {
            continue;
        }
        if let Some(outcome) =
            expired(started, limits, cancellation).or_else(|| member_deadline(member))
        {
            return Ok(Some(outcome));
        }
        let output = member
            .output
            .as_mut()
            .ok_or(Failure::InvalidSpec("missing retained output"))?;
        let mut candidate = false;
        output.poll_probe(&mut |line| {
            if rejection.is_none()
                && (!candidate || member.admitted.is_some() || member.expected.is_some())
                && member_deadline_for_line(
                    member.admitted,
                    member.expected.as_ref(),
                    member.started,
                    member.deadline,
                )
                && expired(started, limits, cancellation).is_none()
            {
                match observer.line(member.id, line) {
                    ProbeDecision::Pending => (),
                    ProbeDecision::Candidate => candidate = true,
                    ProbeDecision::Rejected(reason) => *rejection = Some(reason),
                }
            }
        })?;
        if let Some(outcome) =
            expired(started, limits, cancellation).or_else(|| member_deadline(member))
        {
            return Ok(Some(outcome));
        }
        if rejection.is_some() {
            return Ok(Some(Outcome::ObservationRejected));
        }
        let exited = member.child.exited()?;
        if let Some(outcome) =
            expired(started, limits, cancellation).or_else(|| member_deadline(member))
        {
            return Ok(Some(outcome));
        }
        if exited {
            if member.expected.is_none() {
                return Ok(Some(Outcome::EarlyExit));
            }
            if let Some(outcome) = complete_expected(members, index, context, observer, rejection)?
            {
                return Ok(Some(outcome));
            }
        } else if candidate && member.admitted.is_none() && member.expected.is_none() {
            member.admitted = Some(member.started.elapsed());
        }
    }
    Ok(None)
}

fn member_deadline_for_line(
    admitted: Option<std::time::Duration>,
    expected: Option<&super::ExpectedExit>,
    started: Instant,
    readiness: std::time::Duration,
) -> bool {
    match expected {
        Some(policy) => started.elapsed() < policy.deadline(),
        None => admitted.is_some() || started.elapsed() < readiness,
    }
}

fn member_deadline(member: &Member) -> Option<Outcome> {
    match &member.expected {
        Some(policy) if member.started.elapsed() >= policy.deadline() => Some(Outcome::Deadline),
        None if member.admitted.is_none() && member.started.elapsed() >= member.deadline => {
            Some(Outcome::ReadinessDeadline)
        }
        Some(_) | None => None,
    }
}

fn complete_expected<Observer: Coordinator>(
    members: &mut [Member],
    index: usize,
    context: (Instant, &Limits, &Cancellation),
    observer: &mut Observer,
    rejection: &mut Option<Observer::Rejection>,
) -> Result<Option<Outcome>, Failure> {
    let (before, rest) = members.split_at_mut(index);
    let (target, after) = rest
        .split_first_mut()
        .ok_or(Failure::InvalidSpec("expected exit target missing"))?;
    let elapsed = target.started.elapsed();
    let mut terminal = Ok(None);
    target.finish(context.1, Disposition::ExpectedExit, || {
        for survivors in [&mut *before, &mut *after] {
            if matches!(terminal, Ok(None)) {
                terminal = observe(survivors, observer, context, rejection);
            }
        }
    });
    let report = target
        .report
        .as_mut()
        .ok_or(Failure::InvalidSpec("expected exit report missing"))?;
    let policy = target
        .expected
        .as_ref()
        .ok_or(Failure::InvalidSpec("expected exit policy missing"))?;
    let receipt = policy.receipt(elapsed, &report.process);
    let accepted = receipt.accepted(&report.process);
    report.completion = Some(receipt);
    if let Some(outcome) = terminal? {
        return Ok(Some(outcome));
    }
    if !accepted {
        let outcome = if report.process.cleanup.complete && report.process.failure.is_none() {
            Outcome::EarlyExit
        } else {
            Outcome::IoFailure
        };
        report.disposition = Disposition::Failure(outcome);
        return Ok(Some(outcome));
    }
    Ok(None)
}

use super::{Disposition, MemberId, member::Member, owner::expired};
use crate::process::{Cancellation, Failure, Limits, Outcome};
use std::time::Instant;

pub(super) fn stop<Observer: super::Coordinator>(
    members: &mut [Member],
    id: MemberId,
    limits: &Limits,
    observer: &mut Observer,
    mut observe: impl FnMut(&mut [Member], &mut Observer) -> Result<Option<Outcome>, Failure>,
) -> Result<Option<Outcome>, Failure> {
    let index = members
        .iter()
        .position(|member| member.id == id && member.report.is_none() && member.admitted.is_some())
        .ok_or(Failure::InvalidSpec("stop requires a live admitted member"))?;
    let (before, rest) = members.split_at_mut(index);
    let (target, after) = rest
        .split_first_mut()
        .ok_or(Failure::InvalidSpec("stop target missing"))?;
    let mut terminal = Ok(None);
    target.finish(limits, Disposition::IntentionalStop, observer, |observer| {
        for survivors in [&mut *before, &mut *after] {
            if matches!(terminal, Ok(None)) {
                terminal = observe(survivors, observer);
            } else {
                drain(survivors, observer);
            }
        }
    });
    match terminal? {
        Some(outcome) => Ok(Some(outcome)),
        None => Ok(target.report.as_ref().and_then(|report| {
            if matches!(report.disposition, Disposition::IntentionalStop)
                && report.process.failure.is_none()
            {
                None
            } else {
                Some(Outcome::IoFailure)
            }
        })),
    }
}

pub(super) fn finish<Observer: super::Coordinator>(
    members: &mut [Member],
    context: (Instant, &Limits, &Cancellation),
    terminal: (&mut Outcome, &mut Option<Failure>),
    observer: &mut Observer,
) {
    let (started, limits, cancellation) = context;
    let (outcome, failure) = terminal;
    for index in 0..members.len() {
        let observation = observe_cleanup(members, context, observer);
        if *outcome == Outcome::Ready {
            match observation {
                Ok(Some(observed)) => *outcome = observed,
                Ok(None) => (),
                Err(error) => {
                    *outcome = Outcome::IoFailure;
                    *failure = Some(error);
                }
            }
        }
        let (before, rest) = members.split_at_mut(index);
        let Some((target, after)) = rest.split_first_mut() else {
            continue;
        };
        let disposition = match *outcome {
            Outcome::Ready => Disposition::SessionCleanup,
            other => Disposition::Failure(other),
        };
        target.finish(limits, disposition, observer, |observer| {
            for survivors in [&mut *before, &mut *after] {
                let observation = observe_cleanup(survivors, context, observer);
                if *outcome == Outcome::Ready {
                    match observation {
                        Ok(Some(observed)) => *outcome = observed,
                        Ok(None) => (),
                        Err(error) => {
                            *outcome = Outcome::IoFailure;
                            *failure = Some(error);
                        }
                    }
                }
            }
        });
        if *outcome == Outcome::Ready
            && let Some(report) = &target.report
            && let Disposition::Failure(observed) = report.disposition
        {
            *outcome = observed;
        }
        if *outcome == Outcome::Ready
            && let Some(report) = &target.report
            && matches!(report.disposition, Disposition::SessionCleanup)
            && !report.process.success()
        {
            *outcome = Outcome::IoFailure;
        }
    }
    if *outcome == Outcome::Ready
        && let Some(observed) = expired(started, limits, cancellation)
    {
        *outcome = observed;
    }
}

pub(super) fn observe_cleanup<Observer: super::Coordinator>(
    survivors: &mut [Member],
    context: (Instant, &Limits, &Cancellation),
    observer: &mut Observer,
) -> Result<Option<Outcome>, Failure> {
    let (started, limits, cancellation) = context;
    let mut outcome = expired(started, limits, cancellation);
    for member in survivors {
        if let Some(output) = &mut member.output {
            if let Err(error) = output.poll_captured(&mut |line, _readiness_allowed| {
                observer.captured_line(member.id, line)
            }) {
                output.failure.get_or_insert(error);
            }
            if let Some(error) = output.failure.take() {
                return Err(error);
            }
            if member.child.exited()? && outcome.is_none() {
                outcome = Some(Outcome::EarlyExit);
            }
        }
    }
    Ok(outcome.or_else(|| expired(started, limits, cancellation)))
}

fn drain<Observer: super::Coordinator>(members: &mut [Member], observer: &mut Observer) {
    for member in members {
        if let Some(output) = &mut member.output {
            if let Err(error) = output.poll_captured(&mut |line, _readiness_allowed| {
                observer.captured_line(member.id, line)
            }) {
                output.failure.get_or_insert(error);
            }
            if let Err(error) = member.child.exited() {
                output.failure.get_or_insert(error);
            }
        }
    }
}

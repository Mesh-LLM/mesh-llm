use super::{Disposition, MemberId, member::Member, owner::expired};
use crate::process::{Cancellation, Failure, Limits, Outcome, Readiness};
use std::time::Instant;

pub(super) fn stop(
    members: &mut [Member],
    id: MemberId,
    limits: &Limits,
    mut observe: impl FnMut(&mut [Member]) -> Result<Option<Outcome>, Failure>,
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
    target.finish(limits, Disposition::IntentionalStop, || {
        for survivors in [&mut *before, &mut *after] {
            if matches!(terminal, Ok(None)) {
                terminal = observe(survivors);
            } else {
                drain(survivors);
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

pub(super) fn finish(
    members: &mut [Member],
    context: (Instant, &Limits, &Cancellation),
    terminal: (&mut Outcome, &mut Option<Failure>),
) {
    let (started, limits, cancellation) = context;
    let (outcome, failure) = terminal;
    for index in 0..members.len() {
        let observation = observe_cleanup(members, context);
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
        target.finish(limits, disposition, || {
            for survivors in [&mut *before, &mut *after] {
                let observation = observe_cleanup(survivors, context);
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

pub(super) fn observe_cleanup(
    survivors: &mut [Member],
    context: (Instant, &Limits, &Cancellation),
) -> Result<Option<Outcome>, Failure> {
    let (started, limits, cancellation) = context;
    let mut outcome = expired(started, limits, cancellation);
    for member in survivors {
        if let Some(output) = &mut member.output {
            output.poll(&Readiness::None);
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

fn drain(members: &mut [Member]) {
    for member in members {
        if let Some(output) = &mut member.output {
            output.poll(&Readiness::None);
            if let Err(error) = member.child.exited() {
                output.failure.get_or_insert(error);
            }
        }
    }
}

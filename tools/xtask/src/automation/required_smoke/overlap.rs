use super::{Rejection, Session, observer::Observer};
use crate::process::{
    ObservedLine, ProbeContext, ProbeDecision,
    retained::{Action, Context, Coordinator, Launch, MemberId, MemberState},
};
use std::time::{Duration, Instant};

pub(super) struct Overlap<'a> {
    pub session: &'a mut Session,
    pub primary: Observer,
    pub headless: Observer,
    pub primary_launch: Option<Launch>,
    pub headless_launch: Option<Launch>,
    pub headless_clock: bool,
    pub started: Instant,
}

impl Overlap<'_> {
    fn decide(&mut self, context: Context<'_>) -> Result<Action<Rejection>, Rejection> {
        if let Some(launch) = self.primary_launch.take() {
            return Ok(Action::Start(launch));
        }
        let primary = context
            .members
            .iter()
            .find(|member| member.member == MemberId::Seed)
            .ok_or(Rejection::ProcessIdentity)?;
        match primary.state {
            MemberState::Starting => {
                return Ok(
                    match self
                        .primary
                        .decision(probe(primary, context.remaining), self.session)?
                    {
                        ProbeDecision::Pending => Action::Pending,
                        ProbeDecision::Candidate => Action::Admit(MemberId::Seed),
                        ProbeDecision::Rejected(reason) => Action::Reject(reason),
                    },
                );
            }
            MemberState::Ready { .. } => (),
            MemberState::IntentionalStop => return Err(Rejection::EarlyExit),
        }
        if let Some(launch) = self.headless_launch.take() {
            return Ok(Action::Start(launch));
        }
        let headless = context
            .members
            .iter()
            .find(|member| member.member == MemberId::WorkerOne)
            .ok_or(Rejection::ProcessIdentity)?;
        if !self.headless_clock {
            self.session
                .headless_launched(headless.started.duration_since(self.started))?;
            self.headless_clock = true;
        }
        match headless.state {
            MemberState::Starting => Ok(
                match self
                    .headless
                    .decision(probe(headless, context.remaining), self.session)?
                {
                    ProbeDecision::Pending => Action::Pending,
                    ProbeDecision::Candidate => Action::Admit(MemberId::WorkerOne),
                    ProbeDecision::Rejected(reason) => Action::Reject(reason),
                },
            ),
            MemberState::Ready { .. } => Ok(Action::Complete),
            MemberState::IntentionalStop => Err(Rejection::EarlyExit),
        }
    }
}

fn probe(member: &crate::process::retained::Snapshot, remaining: Duration) -> ProbeContext {
    ProbeContext {
        pid: member.pid,
        elapsed: member.started.elapsed(),
        remaining,
    }
}

impl Coordinator for Overlap<'_> {
    type Rejection = Rejection;
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<Rejection> {
        match member {
            MemberId::Seed => self.primary.line(line),
            MemberId::WorkerOne => self.headless.line(line),
            _ => ProbeDecision::Rejected(Rejection::ProcessIdentity),
        }
    }
    fn tick(&mut self, context: Context<'_>) -> Action<Rejection> {
        self.decide(context).unwrap_or_else(Action::Reject)
    }
}

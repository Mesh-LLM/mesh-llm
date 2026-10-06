//! Retained static arms and mixed request worker; all telemetry callbacks are typed only.
use super::mixed_phase::Observation;
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision, Stream};
pub(super) struct Owner {
    pub downstream: Option<Launch>,
    pub upstream: Option<Launch>,
    pub worker: Option<Launch>,
    pub split: bool,
    pub policy: ExpectedExit,
    pub telemetry: Observation,
    pub stopping: u8,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member == MemberId::WorkerOne && line.stream == Stream::Stderr {
            self.telemetry.observe(line.bytes);
        }
    }
    fn line(&mut self, _member: MemberId, _line: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(error) = &self.telemetry.error {
            return Action::Reject(error.clone());
        }
        match self.stopping {
            1 => {
                self.stopping = 2;
                if self.split {
                    return Action::Stop(MemberId::Seed);
                }
                return Action::Complete;
            }
            2 => return Action::Complete,
            _ => {}
        }
        if let Some(launch) = self.downstream.take() {
            return Action::Start(launch);
        }
        for id in [MemberId::Seed, MemberId::WorkerOne] {
            if context
                .members
                .iter()
                .any(|m| m.member == id && m.state == MemberState::Starting)
            {
                return Action::Admit(id);
            }
        }
        if (!self.split
            || context.members.iter().any(|m| {
                m.member == MemberId::Seed && matches!(m.state, MemberState::Ready { .. })
            }))
            && let Some(launch) = self.upstream.take()
        {
            return Action::Start(launch);
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerOne && matches!(m.state, MemberState::Ready { .. })
        }) && let Some(launch) = self.worker.take()
        {
            return Action::StartExpected {
                launch,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerTwo && matches!(m.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopping = 1;
            return Action::Stop(MemberId::WorkerOne);
        }
        Action::Pending
    }
}

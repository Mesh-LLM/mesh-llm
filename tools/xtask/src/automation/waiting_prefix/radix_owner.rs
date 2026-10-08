//! One retained standalone OpenAI arm, with exact summary barriers between owned batches.
use super::radix_projection::Projection;
use crate::process::{
    ObservedLine, ProbeDecision,
    retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};
pub(super) struct Next {
    pub launch: Launch,
    pub summaries: usize,
    pub warmup: bool,
}
pub(super) struct Owner {
    pub server: Option<Launch>,
    pub batches: VecDeque<Next>,
    pub active: Option<(MemberId, usize, bool, Instant)>,
    pub projection: Projection,
    pub boundaries: Vec<(usize, usize, bool)>,
    pub consumed: usize,
    pub batch_budget: Duration,
    pub telemetry_budget: Duration,
    pub stopping: bool,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _member: MemberId, _line: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member == MemberId::Seed {
            self.projection.observe(line.bytes);
        }
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(error) = &self.projection.error {
            return Action::Reject(error.clone());
        }
        if self.stopping {
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            return Action::Start(server);
        }
        if context
            .members
            .iter()
            .any(|m| m.member == MemberId::Seed && m.state == MemberState::Starting)
        {
            return Action::Admit(MemberId::Seed);
        }
        if !context
            .members
            .iter()
            .any(|m| m.member == MemberId::Seed && matches!(m.state, MemberState::Ready { .. }))
        {
            return Action::Pending;
        }
        if let Some((id, count, warmup, started)) = self.active {
            let status = context
                .members
                .iter()
                .find(|m| m.member == id)
                .and_then(|m| match m.state {
                    MemberState::ExpectedExit { status, .. } => Some(status),
                    _ => None,
                });
            if let Some(status) = status {
                if status != 0 {
                    return Action::Reject(
                        "radix HTTP batch failed; partial receipt retained".into(),
                    );
                }
                let expected = self.consumed + count;
                if self.projection.rows.len() > expected {
                    return Action::Reject(
                        "radix unexpected extra generation summaries at batch boundary".into(),
                    );
                }
                if self.projection.rows.len() == expected {
                    self.boundaries.push((self.consumed, expected, warmup));
                    self.consumed = expected;
                    self.active = None;
                } else if started.elapsed() > self.batch_budget + self.telemetry_budget {
                    return Action::Reject(format!(
                        "radix delayed summaries missing: observed {}/{} events",
                        self.projection.rows.len() - self.consumed,
                        count
                    ));
                } else {
                    return Action::Pending;
                }
            } else {
                return Action::Pending;
            }
        }
        if let Some(next) = self.batches.pop_front() {
            let id = next.launch.member;
            self.active = Some((id, next.summaries, next.warmup, Instant::now()));
            match ExpectedExit::new(&[0, 1], self.batch_budget) {
                Ok(policy) => Action::StartExpected {
                    launch: next.launch,
                    policy,
                },
                Err(error) => Action::Reject(format!("radix worker policy invalid: {error}")),
            }
        } else {
            self.stopping = true;
            Action::Stop(MemberId::Seed)
        }
    }
}

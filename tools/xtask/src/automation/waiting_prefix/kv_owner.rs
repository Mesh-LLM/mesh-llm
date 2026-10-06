//! One retained default-startup server and its full-message cohort worker.
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision};
pub(super) struct Owner {
    pub server: Option<Launch>,
    pub worker: Option<Launch>,
    pub policy: ExpectedExit,
    pub stopped: bool,
    pub started: std::time::Instant,
    pub request_sha256: String,
    pub ready_observed: Option<std::time::Duration>,
    pub ready_ambiguous: bool,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _member: MemberId, _line: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member != MemberId::WorkerOne || line.stream != crate::process::Stream::Stdout {
            return;
        }
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(line.bytes) else {
            return;
        };
        if value["event"] != "kv_restart_model_ready" {
            return;
        }
        if value["request_sha256"] != self.request_sha256 || self.ready_observed.is_some() {
            self.ready_ambiguous = true;
            return;
        }
        self.ready_observed = Some(self.started.elapsed());
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopped {
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
        if context
            .members
            .iter()
            .any(|m| m.member == MemberId::Seed && matches!(m.state, MemberState::Ready { .. }))
            && let Some(worker) = self.worker.take()
        {
            return Action::StartExpected {
                launch: worker,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerOne && matches!(m.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopped = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

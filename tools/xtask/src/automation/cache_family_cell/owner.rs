//! One owned host, then separate HTTP-readiness and measurement members.
use super::contract::{Host, Input};
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision};
use std::time::Duration;
pub(super) struct Owner {
    pub server: Option<Launch>,
    pub readiness: Option<Launch>,
    pub measurement: Option<Launch>,
    pub input: Input,
    pub dialect: crate::automation::skippy_cli_admission::Dialect,
    pub marker: bool,
    pub stopping: bool,
}
impl Owner {
    pub(super) fn bound_session(&mut self, execution: Duration) {
        for launch in [&mut self.server, &mut self.readiness, &mut self.measurement]
            .into_iter()
            .flatten()
        {
            launch.readiness_deadline = launch.readiness_deadline.min(execution);
        }
    }
    fn listening(&self, bytes: &[u8]) -> bool {
        if bytes.len() > 16384 {
            return false;
        }
        let Ok(line) = std::str::from_utf8(bytes) else {
            return false;
        };
        if self.input.host == Host::NativeBaseline {
            let marker = format!(
                "srv  llama_server: listening on http://127.0.0.1:{}",
                self.input.port
            );
            return line.trim() == marker;
        }
        if matches!(
            self.dialect,
            crate::automation::skippy_cli_admission::Dialect::Current
        ) {
            let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
                return false;
            };
            return event["schema_version"].as_u64() == Some(1)
                && event["type"] == "status"
                && event["sequence"]
                    .as_u64()
                    .is_some_and(|sequence| sequence > 0)
                && event["data"]["message"].as_str()
                    == Some(&format!(
                        "skippy-serving listening: openai=127.0.0.1:{}",
                        self.input.port
                    ));
        }
        let marker = format!(
            "skippy-server listening: openai=127.0.0.1:{} model_id={} backend=",
            self.input.port, self.input.model_id
        );
        let Some(rest) = line.trim().strip_prefix(&marker) else {
            return false;
        };
        rest.split_whitespace()
            .any(|field| field == format!("generation_concurrency={}", self.input.lane_count))
    }
}
impl Coordinator for Owner {
    type Rejection = String;
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member == MemberId::Seed && self.listening(line.bytes) {
            self.marker = true;
        }
    }
    fn line(&mut self, _member: MemberId, _line: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopping {
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            return Action::Start(server);
        }
        if self.marker
            && context
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
            && let Some(worker) = self.readiness.take()
        {
            return match ExpectedExit::new(
                &[0, 1],
                Duration::from_secs(self.input.startup_timeout_secs)
                    .saturating_sub(context.elapsed)
                    .min(context.remaining),
            ) {
                Ok(policy) => Action::StartExpected {
                    launch: worker,
                    policy,
                },
                Err(_) => Action::Reject("invalid readiness completion budget".into()),
            };
        }
        if let Some(member) = context.members.iter().find(|m| {
            m.member == MemberId::WorkerOne && matches!(m.state, MemberState::ExpectedExit { .. })
        }) {
            if !matches!(member.state, MemberState::ExpectedExit { status: 0, .. }) {
                return Action::Reject("owned HTTP/model readiness refused".into());
            }
            if let Some(worker) = self.measurement.take() {
                return match ExpectedExit::new(&[0, 1], context.remaining) {
                    Ok(policy) => Action::StartExpected {
                        launch: worker,
                        policy,
                    },
                    Err(_) => Action::Reject("invalid measurement completion budget".into()),
                };
            }
        }
        if context.members.iter().any(|m| {
            m.member == MemberId::WorkerTwo && matches!(m.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopping = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

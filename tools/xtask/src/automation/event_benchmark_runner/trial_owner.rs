//! Retain one fresh host and its native readiness/measurement worker.
use super::health_log;
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision};
use std::time::Duration;

pub(super) struct Owner {
    pub server: Option<Launch>,
    pub worker: Option<Launch>,
    pub worker_policy: ExpectedExit,
    pub stopping: bool,
    pub setup_ms: Option<f64>,
    pub stop_started: Option<Duration>,
    pub shutdown_ms: Option<f64>,
    pub api_readiness: Option<(String, u16)>,
    pub listener_ready: bool,
    pub host_readiness_timeout: Duration,
    pub host_started: Option<Duration>,
    pub health_streams: [Option<health_log::Observation>; 2],
}

impl Coordinator for Owner {
    type Rejection = String;
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member == MemberId::Seed
            && let Some(observation) = health_log::line(line.bytes)
        {
            let index = usize::from(line.stream == crate::process::Stream::Stderr);
            self.health_streams[index] = Some(observation);
        }
    }
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<String> {
        // Typed health is observed by captured_line before sanitization, including
        // stop/cleanup drains; this readiness callback only classifies owned startup.
        if member == MemberId::Seed
            && !self.listener_ready
            && let Some((url, port)) = &self.api_readiness
            && let Ok(value) = serde_json::from_slice::<serde_json::Value>(line.bytes)
        {
            self.listener_ready = match value.get("event").and_then(serde_json::Value::as_str) {
                Some("api_ready") => {
                    value.get("url").and_then(serde_json::Value::as_str) == Some(url.as_str())
                }
                Some("ready") => {
                    value.get("api_url").and_then(serde_json::Value::as_str) == Some(url.as_str())
                        && value.get("api_port").and_then(serde_json::Value::as_u64)
                            == Some(u64::from(*port))
                }
                _ => false,
            };
        }
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopping {
            self.shutdown_ms = self
                .stop_started
                .map(|started| context.elapsed.saturating_sub(started).as_secs_f64() * 1000.0);
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            self.host_started = Some(context.elapsed);
            return Action::Start(server);
        }
        if !self.listener_ready
            && self.host_started.is_some_and(|started| {
                context.elapsed.saturating_sub(started) >= self.host_readiness_timeout
            })
        {
            return Action::Reject("owned host API readiness deadline expired".into());
        }
        if let Some(server) = context
            .members
            .iter()
            .find(|member| member.member == MemberId::Seed)
        {
            if matches!(server.state, MemberState::Starting) {
                self.setup_ms = Some(context.elapsed.as_secs_f64() * 1000.0);
                // Admission means ownership of the live host, not HTTP readiness.
                // The separate worker proves readiness without blocking tick.
                return Action::Admit(MemberId::Seed);
            }
            if self.listener_ready
                && matches!(server.state, MemberState::Ready { .. })
                && let Some(worker) = self.worker.take()
            {
                return Action::StartExpected {
                    launch: worker,
                    policy: self.worker_policy.clone(),
                };
            }
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && matches!(member.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopping = true;
            self.stop_started = Some(context.elapsed);
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

#[cfg(test)]
#[path = "trial_owner_tests.rs"]
mod tests;

impl Owner {
    pub(super) fn health_capture_complete(
        report: &crate::process::retained::Report<String>,
    ) -> bool {
        report.failure.is_none()
            && report
                .members
                .iter()
                .find(|member| member.member == MemberId::Seed)
                .is_some_and(|host| {
                    host.process.failure.is_none()
                        && host.process.cleanup.complete
                        && host.process.cleanup.failure.is_none()
                        && !host.process.cleanup.graceful_signal_failed
                        && [&host.process.stdout, &host.process.stderr]
                            .iter()
                            .all(|stream| stream.line_capture_complete)
                })
    }
    pub(super) fn final_health(
        &self,
        report: &crate::process::retained::Report<String>,
    ) -> crate::command::DynResult<health_log::Observation> {
        if !Self::health_capture_complete(report) {
            return Err(
                "host typed line observation incomplete (EOF, oversized record or capture failure)"
                    .into(),
            );
        }
        match (&self.health_streams[0], &self.health_streams[1]) {
            (Some(_), Some(_)) => {
                Err("final health chronology is ambiguous across stdout and stderr".into())
            }
            (Some(value), None) | (None, Some(value)) => Ok(value.clone()),
            (None, None) => Ok(health_log::Observation::default()),
        }
    }
}

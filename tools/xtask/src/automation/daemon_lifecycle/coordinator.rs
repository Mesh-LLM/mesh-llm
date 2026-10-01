use super::checks::Check;
use crate::process::{
    ObservedLine, ProbeDecision,
    retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
};
use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
    time::Duration,
};

pub(super) enum Phase {
    Version,
    Auth,
    Start,
    Ready,
    Models,
    Stop,
    FailFast,
    Intents,
    Activity,
    Complete,
}
pub(super) struct Owner {
    pub version: Option<Launch>,
    pub auth: Option<Launch>,
    pub launches: VecDeque<Launch>,
    pub phase: Phase,
    pub index: usize,
    pub current: Option<MemberId>,
    pub requests: SyncSender<Check>,
    pub responses: Receiver<Result<(), String>>,
    pub pending: bool,
    pub base: u16,
    pub wait: Duration,
    pub completed: Vec<&'static str>,
    pub prerequisites: Vec<&'static str>,
    pub owner_available: bool,
    pub usage_exit: bool,
}
impl Owner {
    fn request(&mut self, check: Check) -> Action<String> {
        if !self.pending {
            if self.requests.try_send(check).is_err() {
                return Action::Reject("daemon HTTP worker disconnected".into());
            }
            self.pending = true;
        }
        Action::Pending
    }
    fn expected(&self, launch: Launch, statuses: &[i32]) -> Action<String> {
        match ExpectedExit::new(statuses, self.wait) {
            Ok(policy) => Action::StartExpected { launch, policy },
            Err(error) => Action::Reject(error.to_string()),
        }
    }
    fn exited(context: &Context<'_>, name: &[u8]) -> bool {
        context.members.iter().any(|member| {
            member.member.name() == name && matches!(member.state, MemberState::ExpectedExit { .. })
        })
    }
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<String> {
        if member.name() == b"mode-on-demand" {
            let text = String::from_utf8_lossy(line.bytes).to_ascii_lowercase();
            if text.contains("help") || text.contains("usage") {
                self.usage_exit = true;
            }
        }
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        let done = if self.pending {
            match self.responses.try_recv() {
                Ok(Ok(())) => {
                    self.pending = false;
                    true
                }
                Ok(Err(reason)) => {
                    self.pending = false;
                    if self.index == 2
                        && self.usage_exit
                        && Self::exited(&context, b"mode-on-demand")
                    {
                        self.prerequisites.push("runtime_mode_on_demand");
                        self.index = 3;
                        self.phase = Phase::Start;
                        return Action::Pending;
                    }
                    if matches!(self.phase, Phase::Intents)
                        && reason == "owner-control prerequisite unavailable"
                    {
                        self.prerequisites.extend([
                            "runtime_load_model",
                            "runtime_unload_model",
                            "runtime_ensure_model",
                            "runtime_drain_model",
                        ]);
                        self.phase = Phase::Activity;
                        return Action::Pending;
                    }
                    return Action::Reject(reason);
                }
                Err(TryRecvError::Empty) => false,
                Err(TryRecvError::Disconnected) => {
                    return Action::Reject("daemon worker disconnected".into());
                }
            }
        } else {
            false
        };
        match self.phase {
            Phase::Version => {
                if let Some(launch) = self.version.take() {
                    return self.expected(launch, &[0]);
                }
                if Self::exited(&context, b"version") {
                    self.completed.push("prereq.current-binary");
                    self.phase = Phase::Auth;
                }
                Action::Pending
            }
            Phase::Auth => {
                if let Some(launch) = self.auth.take() {
                    return self.expected(launch, &(0..=255).collect::<Vec<_>>());
                }
                if let Some(member) = context
                    .members
                    .iter()
                    .find(|member| member.member.name() == b"owner-auth")
                    && let MemberState::ExpectedExit { status, .. } = member.state
                {
                    self.owner_available = status == 0;
                    if self.owner_available {
                        self.completed.push("prereq.owner-identity");
                    } else {
                        self.prerequisites.push("prereq.owner-identity");
                        for launch in &mut self.launches {
                            if let Some(index)=launch.spec.arguments.iter().position(|argument|matches!(argument,crate::process::Value::Public(value)if value=="--owner-key")) {
                                launch.spec.arguments.drain(index..index+2);
                            }
                        }
                    }
                    self.phase = Phase::Start;
                }
                Action::Pending
            }
            Phase::Start => {
                let Some(launch) = self.launches.pop_front() else {
                    return Action::Reject("daemon launch missing".into());
                };
                self.current = Some(launch.member);
                if self.index == 4 {
                    self.phase = Phase::FailFast;
                    self.expected(launch, &(1..=255).collect::<Vec<_>>())
                } else if self.index == 2 {
                    self.phase = Phase::Ready;
                    self.expected(launch, &(0..=255).collect::<Vec<_>>())
                } else {
                    self.phase = Phase::Ready;
                    Action::Start(launch)
                }
            }
            Phase::Ready => {
                if done {
                    self.completed.push(
                        [
                            "zero_model_serve_ready",
                            "runtime_mode_serve",
                            "runtime_mode_on_demand",
                            "best_effort_startup",
                        ][self.index],
                    );
                    self.phase = if self.index == 0 {
                        Phase::Models
                    } else {
                        Phase::Stop
                    };
                    match self.current {
                        Some(member) => Action::Admit(member),
                        None => Action::Reject("daemon identity missing".into()),
                    }
                } else {
                    self.request(Check::Ready(
                        self.base + u16::try_from(self.index).unwrap_or(4) * 4 + 1,
                    ))
                }
            }
            Phase::Models => {
                if done {
                    self.index = 1;
                    self.phase = Phase::Start;
                    Action::Pending
                } else {
                    self.request(Check::Models(self.base))
                }
            }
            Phase::Stop => {
                self.index += 1;
                self.phase = Phase::Start;
                match self.current {
                    Some(member) => Action::Stop(member),
                    None => Action::Reject("daemon stop identity missing".into()),
                }
            }
            Phase::FailFast => {
                if Self::exited(&context, b"fail-fast") {
                    self.completed.push("fail_fast_startup");
                    self.phase = Phase::Intents;
                }
                Action::Pending
            }
            Phase::Intents => {
                if !self.owner_available {
                    self.prerequisites.extend([
                        "runtime_load_model",
                        "runtime_unload_model",
                        "runtime_ensure_model",
                        "runtime_drain_model",
                    ]);
                    self.phase = Phase::Activity;
                    return Action::Pending;
                }
                if done {
                    self.completed.extend([
                        "runtime_load_model",
                        "runtime_unload_model",
                        "runtime_ensure_model",
                        "runtime_drain_model",
                    ]);
                    self.phase = Phase::Activity;
                    Action::Pending
                } else {
                    self.request(Check::Intents(self.base + 1))
                }
            }
            Phase::Activity => {
                if done {
                    self.completed
                        .extend(["activity_override", "privacy_no_raw_data"]);
                    self.phase = Phase::Complete;
                    Action::Pending
                } else {
                    self.request(Check::Activity(self.base + 1))
                }
            }
            Phase::Complete => Action::Complete,
        }
    }
}

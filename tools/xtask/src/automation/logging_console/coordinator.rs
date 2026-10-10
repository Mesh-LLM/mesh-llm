use super::checks::Check;
use crate::process::{
    ObservedLine, ProbeDecision,
    retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
};
use std::{
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
    time::Duration,
};

pub(super) struct Owner {
    pub initial: Option<Launch>,
    pub restarted: Option<Launch>,
    pub browser: Option<Launch>,
    pub requests: SyncSender<Check>,
    pub responses: Receiver<Result<String, String>>,
    pub stage: Stage,
    pub pending: bool,
    pub request_id: String,
    pub results: Vec<super::ResultRow>,
    pub wait: Duration,
}

pub(super) enum Stage {
    InitialReady,
    Persist,
    Stop,
    Restart,
    RestartReady,
    Detail,
    Access,
    Browser,
    BrowserWait,
    Replay,
    Complete,
}

impl Owner {
    fn request(&mut self, check: Check) -> Action<String> {
        if !self.pending {
            if self.requests.try_send(check).is_err() {
                return Action::Reject("logging HTTP worker disconnected".into());
            }
            self.pending = true;
        }
        Action::Pending
    }

    fn result(&mut self) -> Result<Option<String>, String> {
        if !self.pending {
            return Ok(None);
        }
        match self.responses.try_recv() {
            Ok(result) => {
                self.pending = false;
                result.map(Some)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("logging HTTP worker disconnected".into()),
        }
    }

    fn passed(&mut self, name: &'static str) {
        self.results.push(super::ResultRow {
            status: "PASS",
            name,
            message: "check completed".into(),
        });
    }
}

impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(launch) = self.initial.take() {
            return Action::Start(launch);
        }
        let result = match self.result() {
            Ok(result) => result,
            Err(reason) => return Action::Reject(reason),
        };
        match self.stage {
            Stage::InitialReady => match result {
                Some(_) => {
                    self.passed("real_embedded_console_bundle");
                    self.stage = Stage::Persist;
                    Action::Admit(MemberId::Seed)
                }
                None => self.request(Check::Ready),
            },
            Stage::Persist => match result {
                Some(id) => {
                    self.request_id = id;
                    self.passed("real_openai_lifecycle_and_detail");
                    self.stage = Stage::Stop;
                    Action::Pending
                }
                None => self.request(Check::Persist),
            },
            Stage::Stop => {
                self.stage = Stage::Restart;
                Action::Stop(MemberId::Seed)
            }
            Stage::Restart => {
                self.stage = Stage::RestartReady;
                match self.restarted.take() {
                    Some(launch) => Action::Start(launch),
                    None => Action::Reject("restart launch missing".into()),
                }
            }
            Stage::RestartReady => match result {
                Some(_) => {
                    self.stage = Stage::Detail;
                    match MemberId::Seed.next_generation() {
                        Ok(member) => Action::Admit(member),
                        Err(error) => Action::Reject(error.to_string()),
                    }
                }
                None => self.request(Check::Ready),
            },
            Stage::Detail => match result {
                Some(_) => {
                    self.passed("restart_persistence");
                    self.stage = Stage::Access;
                    Action::Pending
                }
                None => self.request(Check::Restart(self.request_id.clone())),
            },
            Stage::Access => match result {
                Some(_) => {
                    self.passed("trusted_local_rejection");
                    self.stage = Stage::Browser;
                    Action::Pending
                }
                None => self.request(Check::Access),
            },
            Stage::Browser => {
                let Some(mut launch) = self.browser.take() else {
                    return Action::Reject("browser launch missing".into());
                };
                launch.spec.environment.insert(
                    "MESH_LOGS_E2E_PERSISTED_REQUEST_ID".into(),
                    crate::process::Value::Public(self.request_id.clone().into()),
                );
                self.stage = Stage::BrowserWait;
                match ExpectedExit::new(&[0], self.wait) {
                    Ok(policy) => Action::StartExpected { launch, policy },
                    Err(error) => Action::Reject(error.to_string()),
                }
            }
            Stage::BrowserWait => {
                if context.members.iter().any(|member| {
                    member.member == MemberId::WorkerOne
                        && matches!(member.state, MemberState::ExpectedExit { status: 0, .. })
                }) {
                    self.passed("real_console_playwright");
                    self.stage = Stage::Replay;
                }
                Action::Pending
            }
            Stage::Replay => match result {
                Some(_) => {
                    self.passed("dedicated_sse_replay_gap_and_authoritative_hydration");
                    self.stage = Stage::Complete;
                    Action::Pending
                }
                None => self.request(Check::Replay),
            },
            Stage::Complete => Action::Complete,
        }
    }
}

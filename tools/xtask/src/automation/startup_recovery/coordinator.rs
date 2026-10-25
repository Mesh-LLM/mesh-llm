use super::{
    options::Options,
    worker::{Facts, RequestKind},
};
use crate::process::{
    ObservedLine, ProbeDecision,
    retained::{Action, Context, Coordinator, Launch, MemberId, MemberState, recovery::Stability},
};
use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

pub(super) enum Phase {
    Invite,
    Workers,
    Startup,
    StartupChat,
    Select,
    Recovery,
    RecoveryChat,
    Complete,
}
pub(super) struct Owner<'a> {
    pub options: &'a Options,
    pub seed: Option<Launch>,
    pub workers: VecDeque<Launch>,
    pub requests: SyncSender<RequestKind>,
    pub responses: Receiver<Result<Facts, String>>,
    pub phase: Phase,
    pub pending: bool,
    pub next: Instant,
    pub until: Instant,
    pub driver: usize,
    pub killed: String,
    pub old_run: String,
    pub stopped: usize,
    pub stability: Stability,
    pub completed: Vec<&'static str>,
}
impl Owner<'_> {
    fn request(&mut self, request: RequestKind) -> Action<String> {
        if !self.pending && Instant::now() >= self.next {
            if self.requests.try_send(request).is_err() {
                return Action::Reject("split HTTP worker disconnected".into());
            }
            self.pending = true;
        }
        Action::Pending
    }
    fn fact(&mut self) -> Result<Option<Facts>, String> {
        if !self.pending {
            return Ok(None);
        }
        match self.responses.try_recv() {
            Ok(result) => {
                self.pending = false;
                self.next = Instant::now() + Duration::from_secs(1);
                result.map(Some)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("split HTTP worker disconnected".into()),
        }
    }
}
impl Coordinator for Owner<'_> {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if Instant::now() >= self.until {
            return Action::Reject("split narrative phase deadline".into());
        }
        if let Some(launch) = self.seed.take() {
            return Action::Start(launch);
        }
        let facts = match self.fact() {
            Ok(facts) => facts,
            Err(reason) => return Action::Reject(reason),
        };
        match self.phase {
            Phase::Invite => match facts {
                Some(Facts::Invite(token)) => {
                    for launch in &mut self.workers {
                        launch
                            .spec
                            .arguments
                            .push(crate::process::Value::Public("--join".into()));
                        launch
                            .spec
                            .arguments
                            .push(crate::process::Value::Secret(token.clone().into()));
                    }
                    self.phase = Phase::Workers;
                    self.completed.push("seed-invite-token");
                    Action::Admit(MemberId::Seed)
                }
                Some(Facts::Pending) | None => self.request(RequestKind::Invite),
                Some(_) => Action::Reject("out-of-order invite evidence".into()),
            },
            Phase::Workers => {
                if let Some(launch) = self.workers.pop_front() {
                    return Action::Start(launch);
                }
                if let Some(member) = context
                    .members
                    .iter()
                    .find(|member| matches!(member.state, MemberState::Starting))
                {
                    return Action::Admit(member.member);
                }
                self.phase = Phase::Startup;
                self.until = Instant::now() + self.options.startup;
                Action::Pending
            }
            Phase::Startup => match facts {
                Some(Facts::Startup(driver)) => {
                    self.driver = driver;
                    self.completed.push("startup-split-topology-ready");
                    self.phase = if self.options.inference {
                        Phase::StartupChat
                    } else {
                        Phase::Select
                    };
                    Action::Pending
                }
                Some(Facts::Pending) | None => self.request(RequestKind::Startup),
                Some(_) => Action::Reject("out-of-order startup evidence".into()),
            },
            Phase::StartupChat => match facts {
                Some(Facts::Chat) => {
                    self.completed.push("startup-chat");
                    self.phase = Phase::Select;
                    Action::Pending
                }
                None => self.request(RequestKind::Chat(self.driver)),
                Some(_) => Action::Reject("out-of-order chat evidence".into()),
            },
            Phase::Select => match facts {
                Some(Facts::Selected {
                    member,
                    index,
                    killed,
                    old_run,
                }) => {
                    self.stopped = index;
                    self.killed = killed;
                    self.old_run = old_run;
                    self.phase = Phase::Recovery;
                    self.until = Instant::now() + self.options.recovery;
                    self.completed.push("active-downstream-worker-stage");
                    Action::Stop(member)
                }
                None => self.request(RequestKind::Select(self.driver)),
                Some(_) => Action::Reject("out-of-order downstream evidence".into()),
            },
            Phase::Recovery => match facts {
                Some(Facts::Recovery(observations)) => {
                    if self.stability.sweep(&observations).is_some() {
                        self.completed.push("split-recovery");
                        self.phase = if self.options.inference {
                            Phase::RecoveryChat
                        } else {
                            Phase::Complete
                        };
                    }
                    Action::Pending
                }
                None => self.request(RequestKind::Recovery {
                    killed: self.killed.clone(),
                    old_run: self.old_run.clone(),
                    stopped: self.stopped,
                }),
                Some(_) => Action::Reject("out-of-order recovery evidence".into()),
            },
            Phase::RecoveryChat => match facts {
                Some(Facts::Chat) => {
                    self.completed.push("recovery-chat");
                    self.phase = Phase::Complete;
                    Action::Pending
                }
                None => self.request(RequestKind::Chat(self.driver)),
                Some(_) => Action::Reject("out-of-order recovery chat".into()),
            },
            Phase::Complete => Action::Complete,
        }
    }
}

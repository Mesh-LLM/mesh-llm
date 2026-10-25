use super::checks::Check;
use crate::process::{
    ObservedLine, ProbeDecision,
    retained::{Action, Context, Coordinator, Launch, MemberId},
};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

pub(super) enum Phase {
    InitialReady,
    Persist,
    Stop,
    Restart,
    RestartReady,
    Detail,
    Access,
    Delete,
    Replay,
    FailOpenStart,
    FailOpenReady,
    FailOpen,
    Complete,
}
pub(super) struct Owner {
    pub initial: Option<Launch>,
    pub restarted: Option<Launch>,
    pub fail_open: Option<Launch>,
    pub phase: Phase,
    pub requests: SyncSender<Check>,
    pub responses: Receiver<Result<String, String>>,
    pub pending: bool,
    pub base: u16,
    pub id: String,
    pub model: Option<String>,
    pub completed: Vec<&'static str>,
    pub prerequisites: Vec<&'static str>,
}
impl Owner {
    fn request(&mut self, check: Check) -> Action<String> {
        if !self.pending {
            if self.requests.try_send(check).is_err() {
                return Action::Reject("logging recovery worker disconnected".into());
            }
            self.pending = true;
        }
        Action::Pending
    }
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, _: Context<'_>) -> Action<String> {
        if let Some(launch) = self.initial.take() {
            return Action::Start(launch);
        }
        let result = if self.pending {
            match self.responses.try_recv() {
                Ok(Ok(value)) => {
                    self.pending = false;
                    Some(value)
                }
                Ok(Err(reason)) => return Action::Reject(reason),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    return Action::Reject("logging recovery worker disconnected".into());
                }
            }
        } else {
            None
        };
        match self.phase {
            Phase::InitialReady => {
                if result.is_some() {
                    self.phase = Phase::Persist;
                    Action::Admit(MemberId::Seed)
                } else {
                    self.request(Check::Ready(self.base + 10))
                }
            }
            Phase::Persist => match result {
                Some(id) => {
                    self.id = id;
                    self.phase = Phase::Stop;
                    Action::Pending
                }
                None => self.request(Check::Persist(self.base + 10)),
            },
            Phase::Stop => {
                self.phase = Phase::Restart;
                Action::Stop(MemberId::Seed)
            }
            Phase::Restart => {
                self.phase = Phase::RestartReady;
                match self.restarted.take() {
                    Some(launch) => Action::Start(launch),
                    None => Action::Reject("logging restart missing".into()),
                }
            }
            Phase::RestartReady => {
                if result.is_some() {
                    self.phase = Phase::Detail;
                    match MemberId::Seed.next_generation() {
                        Ok(member) => Action::Admit(member),
                        Err(error) => Action::Reject(error.to_string()),
                    }
                } else {
                    self.request(Check::Ready(self.base + 20))
                }
            }
            Phase::Detail => {
                if result.is_some() {
                    self.completed.push("logging_restart_privacy");
                    self.phase = Phase::Access;
                    Action::Pending
                } else {
                    self.request(Check::Detail {
                        base: self.base + 20,
                        id: self.id.clone(),
                    })
                }
            }
            Phase::Access => {
                if result.is_some() {
                    self.completed.push("logging_trusted_local_rejection");
                    self.phase = Phase::Delete;
                    Action::Pending
                } else {
                    self.request(Check::Access(self.base + 20))
                }
            }
            Phase::Delete => {
                if result.is_some() {
                    self.completed.push("logging_retention_cascade");
                    self.phase = Phase::Replay;
                    Action::Pending
                } else {
                    self.request(Check::Delete {
                        base: self.base + 20,
                        id: self.id.clone(),
                    })
                }
            }
            Phase::Replay => {
                if result.is_some() {
                    self.completed.push("logging_sse_recovery");
                    self.phase = Phase::FailOpenStart;
                    Action::Pending
                } else {
                    self.request(Check::Replay(self.base + 20))
                }
            }
            Phase::FailOpenStart => {
                self.phase = Phase::FailOpenReady;
                match self.fail_open.take() {
                    Some(launch) => Action::Start(launch),
                    None => Action::Reject("fail-open launch missing".into()),
                }
            }
            Phase::FailOpenReady => {
                if result.is_some() {
                    self.phase = Phase::FailOpen;
                    Action::Admit(MemberId::WorkerOne)
                } else {
                    self.request(Check::Ready(self.base + 30))
                }
            }
            Phase::FailOpen => {
                if result.is_some() {
                    self.completed.push("logging_fail_open");
                    if self.model.is_some() {
                        self.completed.push("logging_fail_open_inference");
                    } else {
                        self.prerequisites.push("logging_fail_open_inference");
                    }
                    self.phase = Phase::Complete;
                    Action::Pending
                } else {
                    self.request(Check::FailOpen {
                        base: self.base + 30,
                        model: self.model.clone(),
                    })
                }
            }
            Phase::Complete => Action::Complete,
        }
    }
}

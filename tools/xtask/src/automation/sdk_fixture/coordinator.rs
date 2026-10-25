use super::readiness;
use crate::process::{
    ObservedLine, ProbeDecision, ProcessSpec, Value,
    retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
};
use std::{
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
    time::Duration,
};

pub(super) struct Owner {
    pub installer: Option<Launch>,
    pub installation_started: bool,
    pub daemon: Option<Launch>,
    pub command: Option<ProcessSpec>,
    pub ready: Receiver<Result<readiness::Ready, String>>,
    pub begin_readiness: Option<SyncSender<()>>,
    pub consumer_deadline: Duration,
    pub admitted: bool,
    pub launched: bool,
}

impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(launch) = self.installer.take() {
            self.installation_started = true;
            return match ExpectedExit::new(&[0], self.consumer_deadline) {
                Ok(policy) => Action::StartExpected { launch, policy },
                Err(error) => Action::Reject(error.to_string()),
            };
        }
        if self.installation_started
            && !context.members.iter().any(|member| {
                member.member.name() == b"runtime-install"
                    && matches!(member.state, MemberState::ExpectedExit { status: 0, .. })
            })
        {
            return Action::Pending;
        }
        if let Some(launch) = self.daemon.take() {
            return Action::Start(launch);
        }
        if !self.admitted {
            if let Some(begin) = self.begin_readiness.take()
                && begin.try_send(()).is_err()
            {
                return Action::Reject("SDK readiness worker disconnected".into());
            }
            match self.ready.try_recv() {
                Ok(Ok(ready)) => {
                    let Some(command) = self.command.as_mut() else {
                        return Action::Reject("SDK command already launched".into());
                    };
                    command.environment.insert(
                        "MESH_SDK_INVITE_TOKEN".into(),
                        Value::Secret(ready.token.into()),
                    );
                    command.environment.insert(
                        "MESH_SDK_MODEL_ID".into(),
                        Value::Public(ready.model.into()),
                    );
                    self.admitted = true;
                    return Action::Admit(MemberId::Seed);
                }
                Ok(Err(reason)) => return Action::Reject(reason),
                Err(TryRecvError::Disconnected) => {
                    return Action::Reject("SDK readiness worker disconnected".into());
                }
                Err(TryRecvError::Empty) => return Action::Pending,
            }
        }
        if !self.launched {
            self.launched = true;
            let Some(command) = self.command.take() else {
                return Action::Reject("SDK command already launched".into());
            };
            let policy = match ExpectedExit::new(&[0], self.consumer_deadline) {
                Ok(policy) => policy,
                Err(error) => return Action::Reject(error.to_string()),
            };
            return Action::StartExpected {
                launch: Launch {
                    member: MemberId::WorkerOne,
                    spec: command,
                    files: Default::default(),
                    readiness_deadline: self.consumer_deadline,
                },
                policy,
            };
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && matches!(member.state, MemberState::ExpectedExit { status: 0, .. })
        }) {
            Action::Complete
        } else {
            Action::Pending
        }
    }
}

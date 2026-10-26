use super::http_checks::{Check, Fact};
use crate::process::{
    ObservedLine, ProbeDecision, Value,
    retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
};

pub(super) enum Step {
    Start(Launch),
    Command {
        launch: Launch,
        prerequisite: bool,
    },
    WaitCommand {
        member: MemberId,
        prerequisite: bool,
    },
    Check {
        name: &'static str,
        check: Check,
    },
    Admit(MemberId),
    Capabilities {
        current: PathBuf,
        released: PathBuf,
    },
}
pub(super) struct Owner {
    pub steps: VecDeque<Step>,
    pub requests: SyncSender<Check>,
    pub responses: Receiver<Result<Fact, String>>,
    pub checking: Option<&'static str>,
    pub token: String,
    pub endpoint: String,
    pub completed: Vec<&'static str>,
    pub prerequisites: Vec<String>,
    pub owner_available: bool,
    pub current_help: String,
    pub released_help: String,
}
impl Owner {
    fn prepare(&self, launch: &mut Launch) {
        let released = launch
            .member
            .name()
            .windows(8)
            .any(|part| part == b"released");
        let help = if released {
            &self.released_help
        } else {
            &self.current_help
        };
        if launch
            .spec
            .arguments
            .iter()
            .any(|value| matches!(value,Value::Public(value)if value=="serve"||value=="--client"))
        {
            if !help.contains("--headless") {
                launch
                    .spec
                    .arguments
                    .retain(|value| !matches!(value,Value::Public(value)if value=="--headless"));
            }
            if (!self.owner_available || !help.contains("--owner-key"))
                && let Some(index) =
                    launch.spec.arguments.iter().position(
                        |value| matches!(value,Value::Public(value)if value=="--owner-key"),
                    )
            {
                launch.spec.arguments.drain(index..index + 2);
            }
        }
        for value in &mut launch.spec.arguments {
            match value {
                Value::Public(argument) if argument == "QA_JOIN_TOKEN" => {
                    *value = Value::Secret(self.token.clone().into())
                }
                Value::Public(argument) if argument == "QA_CONTROL_ENDPOINT" => {
                    *value = Value::Secret(self.endpoint.clone().into())
                }
                Value::Public(_) | Value::Secret(_) => (),
            }
        }
    }
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(name) = self.checking {
            match self.responses.try_recv() {
                Ok(Ok(fact)) => {
                    self.checking = None;
                    match fact {
                        Fact::Done => self.completed.push(name),
                        Fact::Token(token) => {
                            self.token = token;
                            self.completed.push(name);
                        }
                        Fact::Endpoint(endpoint) => {
                            self.endpoint = endpoint;
                            self.completed.push(name);
                        }
                        Fact::Prerequisite(reason) => self.prerequisites.push(reason.into()),
                        Fact::Capabilities { current, released } => {
                            self.current_help = current;
                            self.released_help = released;
                        }
                    }
                }
                Ok(Err(reason)) => return Action::Reject(reason),
                Err(TryRecvError::Empty) => return Action::Pending,
                Err(TryRecvError::Disconnected) => {
                    return Action::Reject("mixed-version HTTP worker disconnected".into());
                }
            }
        }
        let Some(step) = self.steps.pop_front() else {
            return Action::Complete;
        };
        match step {
            Step::Start(mut launch) => {
                self.prepare(&mut launch);
                Action::Start(launch)
            }
            Step::Command {
                mut launch,
                prerequisite,
            } => {
                if launch.spec.arguments.iter().any(
                    |value| matches!(value,Value::Public(value)if value=="QA_CONTROL_ENDPOINT"),
                ) && self.endpoint.is_empty()
                {
                    self.prerequisites
                        .push(String::from_utf8_lossy(launch.member.name()).into_owned());
                    return Action::Pending;
                }
                self.prepare(&mut launch);
                let member = launch.member;
                self.steps.push_front(Step::WaitCommand {
                    member,
                    prerequisite,
                });
                let statuses = if prerequisite {
                    (0..=255).collect::<Vec<_>>()
                } else {
                    vec![0]
                };
                match ExpectedExit::new(&statuses, launch.readiness_deadline) {
                    Ok(policy) => Action::StartExpected { launch, policy },
                    Err(error) => Action::Reject(error.to_string()),
                }
            }
            Step::WaitCommand {
                member,
                prerequisite,
            } => {
                let status = context.members.iter().find_map(|snapshot| {
                    if snapshot.member == member {
                        match snapshot.state {
                            MemberState::ExpectedExit { status, .. } => Some(status),
                            MemberState::Starting
                            | MemberState::Ready { .. }
                            | MemberState::IntentionalStop => None,
                        }
                    } else {
                        None
                    }
                });
                match status {
                    Some(status) => {
                        if member.name() == b"shared-owner-auth" {
                            self.owner_available = status == 0;
                        }
                        if status != 0 && prerequisite {
                            self.prerequisites
                                .push(String::from_utf8_lossy(member.name()).into_owned());
                        }
                        if status == 0 {
                            self.completed.push("command-completed");
                        }
                        Action::Pending
                    }
                    None => {
                        self.steps.push_front(Step::WaitCommand {
                            member,
                            prerequisite,
                        });
                        Action::Pending
                    }
                }
            }
            Step::Check { name, check } => {
                let check = match check {
                    Check::WrongOwner { console, endpoint }
                        if endpoint == "QA_CONTROL_ENDPOINT" =>
                    {
                        if self.endpoint.is_empty() {
                            self.prerequisites.push(name.into());
                            return Action::Pending;
                        }
                        Check::WrongOwner {
                            console,
                            endpoint: self.endpoint.clone(),
                        }
                    }
                    other => other,
                };
                if matches!(&check, Check::ValidateScan(_)) && self.endpoint.is_empty() {
                    self.prerequisites.push(name.into());
                    return Action::Pending;
                }
                if self.requests.try_send(check).is_err() {
                    return Action::Reject("mixed-version worker disconnected".into());
                }
                self.checking = Some(name);
                Action::Pending
            }
            Step::Admit(member) => Action::Admit(member),
            Step::Capabilities { current, released } => {
                if self
                    .requests
                    .try_send(Check::Capabilities { current, released })
                    .is_err()
                {
                    return Action::Reject("mixed-version worker disconnected".into());
                }
                self.checking = Some("binary-capabilities");
                Action::Pending
            }
        }
    }
}

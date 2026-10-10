use super::http_checks::{Check, Fact};
use crate::process::{
    ObservedLine, ProbeDecision, Value,
    retained::{Action, Context, Coordinator, Launch, MemberId},
};
use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, SyncSender, TryRecvError},
};
pub(super) enum Step {
    Start(Launch),
    Admit(MemberId),
    Stop(MemberId),
    Check(Check),
}
pub(super) struct Owner {
    pub steps: VecDeque<Step>,
    pub jobs: SyncSender<Check>,
    pub results: Receiver<Result<Fact, String>>,
    pub waiting: bool,
    pub token: String,
    pub cases: Vec<serde_json::Value>,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, _: Context<'_>) -> Action<String> {
        if self.waiting {
            match self.results.try_recv() {
                Ok(Ok(fact)) => {
                    self.waiting = false;
                    match fact {
                        Fact::Ready => (),
                        Fact::Invite(t) => self.token = t,
                        Fact::Case(row) => {
                            let passed = row["passed"] == true;
                            self.cases.push(row);
                            if !passed {
                                return Action::Reject(
                                    "compatibility expected HTTP/content/usage failed".into(),
                                );
                            }
                        }
                    }
                }
                Ok(Err(e)) => return Action::Reject(e),
                Err(TryRecvError::Empty) => return Action::Pending,
                Err(TryRecvError::Disconnected) => {
                    return Action::Reject("compatibility HTTP worker disconnected".into());
                }
            }
        }
        match self.steps.pop_front() {
            None => Action::Complete,
            Some(Step::Start(mut launch)) => {
                for value in &mut launch.spec.arguments {
                    if matches!(value,Value::Public(v)if v=="COMPATIBILITY_INVITE") {
                        if self.token.is_empty() {
                            return Action::Reject("compatibility invite unavailable".into());
                        }
                        *value = Value::Secret(self.token.clone().into());
                    }
                }
                Action::Start(launch)
            }
            Some(Step::Admit(id)) => Action::Admit(id),
            Some(Step::Stop(id)) => Action::Stop(id),
            Some(Step::Check(check)) => {
                if self.jobs.try_send(check).is_err() {
                    return Action::Reject("compatibility HTTP queue unavailable".into());
                }
                self.waiting = true;
                Action::Pending
            }
        }
    }
}

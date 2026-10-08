use super::{MAX_CONCURRENT_MEMBERS, MemberId};
use crate::process::Failure;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub index: u32,
    pub node_id: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    pub run_id: String,
    pub stages: Vec<Stage>,
}
#[derive(Debug, Clone, Default)]
pub struct Observation {
    pub topologies: Vec<Topology>,
    pub model_count: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    Pending,
    Replacement,
    LocalFallback,
    Withdraw,
}
#[derive(Debug, Clone, Copy)]
pub enum Expected {
    Replacement,
    LocalFallback,
    Withdraw,
    Any,
}
impl Observation {
    fn counts(&self) -> (usize, usize) {
        let stages = self.topologies.iter().flat_map(|topology| &topology.stages);
        let nodes: BTreeSet<_> = stages
            .clone()
            .filter(|stage| !stage.node_id.is_empty())
            .map(|stage| &stage.node_id)
            .collect();
        (stages.count(), nodes.len())
    }
    pub fn startup_ready(&self) -> bool {
        let (stages, nodes) = self.counts();
        !self.topologies.is_empty() && stages >= 2 && nodes >= 2 && self.model_count >= 1
    }
    pub fn recovery(&self, killed: &str, old_run: &str) -> Recovery {
        let killed_present = !killed.is_empty()
            && self
                .topologies
                .iter()
                .flat_map(|topology| &topology.stages)
                .any(|stage| stage.node_id.starts_with(killed));
        if killed_present {
            return Recovery::Pending;
        }
        let new_run = old_run.is_empty()
            || self
                .topologies
                .iter()
                .any(|topology| !topology.run_id.is_empty() && topology.run_id != old_run);
        let (stages, nodes) = self.counts();
        if new_run && self.startup_ready() {
            Recovery::Replacement
        } else if self.model_count >= 1 && nodes <= 1 {
            Recovery::LocalFallback
        } else if self.model_count == 0 || stages == 0 {
            Recovery::Withdraw
        } else {
            Recovery::Pending
        }
    }
    pub fn downstream_worker(
        &self,
        workers: &[(String, MemberId)],
        driver: MemberId,
    ) -> Option<MemberId> {
        let mut fallback = None;
        for stage in self
            .topologies
            .iter()
            .flat_map(|topology| &topology.stages)
            .filter(|stage| stage.index != 0)
        {
            for (short, member) in workers {
                if !short.is_empty() && stage.node_id.starts_with(short) {
                    if *member != driver {
                        return Some(*member);
                    }
                    fallback.get_or_insert(*member);
                }
            }
        }
        fallback
    }
}
pub struct Stability {
    required: NonZeroUsize,
    consecutive: usize,
    expected: Expected,
}
impl Stability {
    pub const fn new(required: NonZeroUsize, expected: Expected) -> Self {
        Self {
            required,
            consecutive: 0,
            expected,
        }
    }
    pub fn sweep(&mut self, observations: &[Recovery]) -> Option<Recovery> {
        let matched = observations
            .iter()
            .copied()
            .find(|recovery| match self.expected {
                Expected::Replacement => *recovery == Recovery::Replacement,
                Expected::LocalFallback => *recovery == Recovery::LocalFallback,
                Expected::Withdraw => *recovery == Recovery::Withdraw,
                Expected::Any => *recovery != Recovery::Pending,
            });
        self.consecutive = match matched {
            Some(_) => self.consecutive.saturating_add(1),
            None => 0,
        };
        matched.filter(|_| self.consecutive >= self.required.get())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    AwaitInvite,
    StartWorkers,
    AwaitPeers,
    AwaitSplit,
    StartupChat,
    SelectDownstream,
    StopWorker(MemberId),
    AwaitRecovery,
    RecoveryChat,
    Complete,
}
pub struct Sequence {
    step: Step,
    inference: bool,
    worker_count: usize,
}
impl Sequence {
    pub const fn new(inference: bool) -> Self {
        Self {
            step: Step::AwaitInvite,
            inference,
            worker_count: 2,
        }
    }
    pub fn with_worker_count(inference: bool, worker_count: usize) -> Result<Self, Failure> {
        if !(2..MAX_CONCURRENT_MEMBERS).contains(&worker_count) {
            return Err(Failure::InvalidSpec(
                "recovery requires 2..=15 workers plus seed",
            ));
        }
        Ok(Self {
            worker_count,
            ..Self::new(inference)
        })
    }
    pub const fn step(&self) -> Step {
        self.step
    }
    pub fn advance(&mut self, facts: Facts) -> Step {
        self.step = match self.step {
            Step::AwaitInvite if facts.invite_present => Step::StartWorkers,
            Step::StartWorkers if facts.workers_ready => Step::AwaitPeers,
            Step::AwaitPeers if facts.seed_peers >= self.worker_count => Step::AwaitSplit,
            Step::AwaitSplit if facts.split_ready => {
                if self.inference {
                    Step::StartupChat
                } else {
                    Step::SelectDownstream
                }
            }
            Step::StartupChat if facts.chat_valid => Step::SelectDownstream,
            Step::SelectDownstream => match facts.downstream {
                Some(member) if member.name() != MemberId::Seed.name() => Step::StopWorker(member),
                Some(_) | None => Step::SelectDownstream,
            },
            Step::StopWorker(member) if facts.stopped == Some(member) => Step::AwaitRecovery,
            Step::AwaitRecovery if facts.stable_recovery => {
                if self.inference {
                    Step::RecoveryChat
                } else {
                    Step::Complete
                }
            }
            Step::RecoveryChat if facts.chat_valid => Step::Complete,
            Step::AwaitInvite
            | Step::StartWorkers
            | Step::AwaitPeers
            | Step::AwaitSplit
            | Step::StartupChat
            | Step::StopWorker(_)
            | Step::AwaitRecovery
            | Step::RecoveryChat
            | Step::Complete => self.step,
        };
        self.step
    }
}
#[derive(Default)]
pub struct Facts {
    pub invite_present: bool,
    pub workers_ready: bool,
    pub seed_peers: usize,
    pub split_ready: bool,
    pub chat_valid: bool,
    pub downstream: Option<MemberId>,
    pub stopped: Option<MemberId>,
    pub stable_recovery: bool,
}
pub fn chat_valid(object: &str, choices: usize) -> bool {
    object == "chat.completion" && choices > 0
}
#[cfg(test)]
#[path = "recovery/tests.rs"]
mod tests;

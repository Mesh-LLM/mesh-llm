//! Coordinated one-shot trees, retained after readiness until the owner stops them.
//! Callbacks follow ReadinessProbe's bounded-work/privacy contract. No callback
//! runs during final cleanup. Survivor line callbacks may run during intentional
//! stop until a terminal observation; tick callbacks do not. Command adapters must install command_interrupt before
//! calling run and finish that scope after run, including error paths.
mod identity;
mod member;
mod owner;
pub mod recovery;
mod shutdown;
use super::{
    Failure, ObservedLine, Outcome, OutputFiles, ProbeDecision, ProcessReport, ProcessSpec,
};
pub use identity::MemberId;
pub use owner::run;
use std::time::{Duration, Instant};
pub const MAX_CONCURRENT_MEMBERS: usize = 16;
pub struct Launch {
    pub member: MemberId,
    pub spec: ProcessSpec,
    pub files: OutputFiles,
    pub readiness_deadline: Duration,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberState {
    Starting,
    Ready { elapsed: Duration },
    IntentionalStop,
}
#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub member: MemberId,
    pub pid: u32,
    pub started: Instant,
    pub state: MemberState,
}
pub struct Context<'a> {
    pub elapsed: Duration,
    pub remaining: Duration,
    pub members: &'a [Snapshot],
}
pub enum Action<Rejection> {
    Pending,
    Start(Launch),
    Admit(MemberId),
    Stop(MemberId),
    Complete,
    Reject(Rejection),
}
pub trait Coordinator {
    type Rejection;
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<Self::Rejection>;
    fn tick(&mut self, context: Context<'_>) -> Action<Self::Rejection>;
}
#[derive(Debug, Clone, Copy)]
pub enum Disposition {
    IntentionalStop,
    SessionCleanup,
    Failure(Outcome),
}
impl Disposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::IntentionalStop => "intentional-stop",
            Self::SessionCleanup => "session-cleanup",
            Self::Failure(_) => "failure",
        }
    }
}
#[derive(Debug)]
pub struct MemberReport {
    pub member: MemberId,
    pub admitted: Option<Duration>,
    pub disposition: Disposition,
    pub process: ProcessReport,
}
pub struct Report<Rejection> {
    pub outcome: Outcome,
    pub rejection: Option<Rejection>,
    pub failure: Option<Failure>,
    pub members: Vec<MemberReport>,
}
impl<Rejection> Report<Rejection> {
    /// Strict smoke success requires graceful completion, including intentional stops.
    pub fn success(&self) -> bool {
        self.recovery_success() && self.members.iter().all(|member| member.process.success())
    }
    /// Recovery may intentionally force a stopped tree, but never waive session cleanup.
    pub fn recovery_success(&self) -> bool {
        self.outcome == Outcome::Ready
            && self.rejection.is_none()
            && self.failure.is_none()
            && self.members.iter().all(|member| match member.disposition {
                Disposition::IntentionalStop => {
                    member.process.cleanup.complete
                        && member.process.status.is_some()
                        && member.process.cleanup.failure.is_none()
                        && !member.process.cleanup.graceful_signal_failed
                        && member.process.failure.is_none()
                }
                Disposition::SessionCleanup => member.process.success(),
                Disposition::Failure(_) => false,
            })
    }
}

use crate::{
    command::DynResult,
    process::{
        self,
        retained::{Action, Context, Coordinator, Launch, MemberId, MemberState},
    },
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    thread::{Scope, ScopedJoinHandle},
    time::Duration,
};
struct Owner {
    launch: Option<Launch>,
    admit: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    pid: Arc<AtomicU32>,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(
        &mut self,
        _: MemberId,
        _: process::ObservedLine<'_>,
    ) -> process::ProbeDecision<String> {
        process::ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(member) = context.members.first() {
            self.pid.store(member.pid, Ordering::Release);
        }
        if let Some(launch) = self.launch.take() {
            return Action::Start(launch);
        }
        if self.stop.load(Ordering::Acquire) {
            return Action::Complete;
        }
        if self.admit.load(Ordering::Acquire)
            && context
                .members
                .iter()
                .any(|member| matches!(member.state, MemberState::Starting))
        {
            return Action::Admit(MemberId::Seed);
        }
        Action::Pending
    }
}
pub(super) struct Session<'scope> {
    admit: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    pid: Arc<AtomicU32>,
    handle: Option<
        ScopedJoinHandle<'scope, Result<process::retained::Report<String>, process::Failure>>,
    >,
}
impl<'scope> Session<'scope> {
    pub fn start<'env>(
        scope: &'scope Scope<'scope, 'env>,
        launch: Launch,
        budget: Duration,
        cancellation: process::Cancellation,
    ) -> Self {
        let admit = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let pid = Arc::new(AtomicU32::new(0));
        let mut owner = Owner {
            launch: Some(launch),
            admit: Arc::clone(&admit),
            stop: Arc::clone(&stop),
            pid: Arc::clone(&pid),
        };
        let terminal = Arc::clone(&finished);
        let handle = scope.spawn(move || {
            let report = process::retained::run(
                &mut owner,
                &super::server_cell_worker::limits(budget),
                &cancellation,
            );
            terminal.store(true, Ordering::Release);
            report
        });
        Self {
            admit,
            stop,
            finished,
            pid,
            handle: Some(handle),
        }
    }
    pub fn admit(&self) {
        self.admit.store(true, Ordering::Release);
    }
    pub fn pid(&self) -> u32 {
        self.pid.load(Ordering::Acquire)
    }
    pub fn finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
    pub fn finish(mut self) -> DynResult<process::retained::Report<String>> {
        self.stop.store(true, Ordering::Release);
        self.handle
            .take()
            .ok_or("server session already joined")?
            .join()
            .map_err(|_| "disk-L3 supervisor panicked")?
            .map_err(|error| format!("disk-L3 supervisor failed: {error:?}").into())
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

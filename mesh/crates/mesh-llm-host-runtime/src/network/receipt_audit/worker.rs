//! The audit worker: one thread with a short queue, so audits never take
//! more than one core from serving, and a turn that would wait too long is
//! refused instead.
//!
//! The thread belongs to the node's audits ([`Slot`]), not to the process:
//!
//! - It starts with the first job and ends when the audits are shut down
//!   or dropped. A shutdown waits a moment for it and joins it.
//! - A job that panics does not end it: the panic is contained, whatever
//!   part of the job it came from, and the runner is told to drop state the
//!   panic may have left half-made.
//! - Should the thread end all the same, the next job says so once and
//!   starts another, a few times. After that the worker is disabled and
//!   says so, instead of failing every turn in silence.
//! - After [`IDLE_TICK`] without a job the runner is told, so it can
//!   release what it holds for the next one.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Jobs waiting for the worker; a full queue refuses the newest.
const QUEUE: usize = 2;
/// How often an idle worker tells its runner that it is idle.
const IDLE_TICK: Duration = Duration::from_secs(60);
/// Times a worker that ended unexpectedly is started again.
const MAX_RESTARTS: u32 = 3;
/// How long a shutdown waits for the worker to end.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);

/// What the worker thread does with its jobs. A runner is made on the
/// thread and never leaves it, so it may hold what only one thread may.
pub(super) trait Runner: 'static {
    type Job: Send + 'static;

    fn run(&mut self, job: Self::Job);
    /// No job came for a while.
    fn idle(&mut self);
    /// A job panicked: drop whatever it may have left half-made.
    fn recover(&mut self);
}

/// Why a job was not queued. It comes back, so the caller can say which
/// turn was not audited.
pub(super) enum Refused<J> {
    /// The queue is full.
    Busy(J),
    /// The audits were shut down.
    ShutDown(J),
    /// The worker ended too often, or could not be started.
    Disabled(J),
}

struct Running<J> {
    queue: SyncSender<J>,
    stop: Arc<AtomicBool>,
    ended: tokio::sync::oneshot::Receiver<()>,
    thread: std::thread::JoinHandle<()>,
}

enum State<J> {
    NotStarted,
    Running(Running<J>),
    ShutDown,
    Disabled,
}

/// The worker of one node's audits, started when its first job comes.
pub(super) struct Slot<J> {
    state: Mutex<State<J>>,
    restarts: Mutex<u32>,
    idle_tick: Duration,
}

impl<J> Default for Slot<J> {
    fn default() -> Self {
        Self::ticking(IDLE_TICK)
    }
}

impl<J> Slot<J> {
    fn ticking(idle_tick: Duration) -> Self {
        Self {
            state: Mutex::new(State::NotStarted),
            restarts: Mutex::new(0),
            idle_tick,
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State<J>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<J: Send + 'static> Slot<J> {
    /// Queues `job` for the worker, which `runner` makes when it has to be
    /// started. The runner gets the flag that says the audits are ending.
    pub(super) fn submit<R: Runner<Job = J>>(
        &self,
        job: J,
        runner: impl FnOnce(Arc<AtomicBool>) -> R + Send + 'static,
    ) -> Result<(), Refused<J>> {
        let mut state = self.state();
        let job = match &*state {
            State::ShutDown => return Err(Refused::ShutDown(job)),
            State::Disabled => return Err(Refused::Disabled(job)),
            State::NotStarted => job,
            State::Running(running) => match running.queue.try_send(job) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(job)) => return Err(Refused::Busy(job)),
                // the thread is gone: another takes the job, or none does
                Err(TrySendError::Disconnected(job)) => job,
            },
        };
        let ended = matches!(&*state, State::Running(_));
        let started = match ended && !self.may_restart() {
            true => None,
            false => self.start(runner),
        };
        let Some(running) = started else {
            *state = State::Disabled;
            return Err(Refused::Disabled(job));
        };
        let queued = running
            .queue
            .try_send(job)
            .map_err(|refused| match refused {
                TrySendError::Full(job) => Refused::Busy(job),
                TrySendError::Disconnected(job) => Refused::Disabled(job),
            });
        *state = State::Running(running);
        queued
    }

    /// Says once that the worker ended, and whether another may start.
    fn may_restart(&self) -> bool {
        let mut restarts = self
            .restarts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *restarts >= MAX_RESTARTS {
            tracing::error!(
                "the receipt audit worker ended unexpectedly {} times: receipt audits are off until this node restarts",
                *restarts + 1
            );
            return false;
        }
        *restarts += 1;
        tracing::error!(
            "the receipt audit worker ended unexpectedly; starting it again ({} of {MAX_RESTARTS})",
            *restarts
        );
        true
    }

    fn start<R: Runner<Job = J>>(
        &self,
        runner: impl FnOnce(Arc<AtomicBool>) -> R + Send + 'static,
    ) -> Option<Running<J>> {
        let (queue, jobs) = sync_channel(QUEUE);
        let stop = Arc::new(AtomicBool::new(false));
        let (ending, ended) = tokio::sync::oneshot::channel();
        let (stopping, idle_tick) = (stop.clone(), self.idle_tick);
        let thread = std::thread::Builder::new()
            .name("receipt-audit".into())
            .spawn(move || {
                run(runner(stopping.clone()), &jobs, &stopping, idle_tick);
                // dropped last: whoever waits for the end sees the jobs gone
                drop(ending);
            });
        match thread {
            Ok(thread) => Some(Running {
                queue,
                stop,
                ended,
                thread,
            }),
            Err(error) => {
                tracing::error!(
                    "receipt audits are off until this node restarts: no worker thread: {error}"
                );
                None
            }
        }
    }

    /// Stops the worker: no job is taken after this, and the one running
    /// ends at its next step. Returns the worker to wait for, if one ran.
    fn stop(&self) -> Option<Stopping> {
        let running = match std::mem::replace(&mut *self.state(), State::ShutDown) {
            State::Running(running) => running,
            State::NotStarted | State::ShutDown | State::Disabled => return None,
        };
        running.stop.store(true, Ordering::Release);
        Some(Stopping {
            ended: running.ended,
            thread: running.thread,
        })
    }

    /// Ends the worker, and joins its thread when it ends in time.
    pub(super) async fn shut_down(&self) {
        if let Some(stopping) = self.stop() {
            stopping.wait().await;
        }
    }
}

/// A worker that was told to stop.
struct Stopping {
    ended: tokio::sync::oneshot::Receiver<()>,
    thread: std::thread::JoinHandle<()>,
}

impl Stopping {
    async fn wait(self) {
        match tokio::time::timeout(SHUTDOWN_WAIT, self.ended).await {
            Ok(_) => {
                if self.thread.join().is_err() {
                    tracing::warn!("the receipt audit worker ended in a panic");
                }
            }
            Err(_) => tracing::warn!(
                "the receipt audit worker is still in an audit; it ends with that audit"
            ),
        }
    }
}

fn run<R: Runner>(mut runner: R, jobs: &Receiver<R::Job>, stop: &AtomicBool, idle_tick: Duration) {
    loop {
        let job = match jobs.recv_timeout(idle_tick) {
            Ok(job) => job,
            Err(RecvTimeoutError::Timeout) => {
                runner.idle();
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return,
        };
        if stop.load(Ordering::Acquire) {
            return;
        }
        // a panic in one audit must not end the audits after it
        if catch_unwind(AssertUnwindSafe(|| runner.run(job))).is_err() {
            tracing::error!("a receipt audit panicked; its turn is not audited");
            runner.recover();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{Sender, channel};

    use super::*;

    /// How long a test waits for the worker to do what it must.
    const GENEROUS: Duration = Duration::from_secs(60);

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Ran(u32),
        Idle,
        Recovered,
        Dropped,
    }

    enum Job {
        Report(u32),
        Panic,
        /// Blocks until the test lets it go.
        Hold(Receiver<()>),
    }

    struct Recorder {
        events: Sender<Event>,
        /// A runner that cannot recover: its panic ends the thread.
        fragile: bool,
    }

    impl Runner for Recorder {
        type Job = Job;

        fn run(&mut self, job: Job) {
            match job {
                Job::Report(n) => drop(self.events.send(Event::Ran(n))),
                Job::Panic => panic!("an audit that panics"),
                Job::Hold(release) => drop(release.recv_timeout(GENEROUS)),
            }
        }

        fn idle(&mut self) {
            let _ = self.events.send(Event::Idle);
        }

        fn recover(&mut self) {
            let _ = self.events.send(Event::Recovered);
            assert!(!self.fragile, "a runner that cannot recover");
        }
    }

    impl Drop for Recorder {
        fn drop(&mut self) {
            let _ = self.events.send(Event::Dropped);
        }
    }

    fn recorder(
        events: &Sender<Event>,
        fragile: bool,
    ) -> impl FnOnce(Arc<AtomicBool>) -> Recorder + Send + 'static {
        let events = events.clone();
        move |_| Recorder { events, fragile }
    }

    fn next(events: &Receiver<Event>) -> Event {
        events
            .recv_timeout(GENEROUS)
            .expect("the worker's next event")
    }

    fn submit(slot: &Slot<Job>, job: Job, events: &Sender<Event>, fragile: bool) -> Result<(), ()> {
        slot.submit(job, recorder(events, fragile)).map_err(drop)
    }

    #[test]
    fn a_panic_in_a_job_does_not_end_the_jobs_after_it() {
        let (tx, events) = channel();
        let slot = Slot::default();
        submit(&slot, Job::Report(1), &tx, false).unwrap();
        assert_eq!(next(&events), Event::Ran(1));
        submit(&slot, Job::Panic, &tx, false).unwrap();
        assert_eq!(next(&events), Event::Recovered);
        submit(&slot, Job::Report(2), &tx, false).unwrap();
        assert_eq!(next(&events), Event::Ran(2), "the same worker goes on");
        assert_eq!(*slot.restarts.lock().unwrap(), 0);
    }

    /// Waits until the worker thread has ended, as a later job finds it.
    fn ended(slot: &Slot<Job>, events: &Receiver<Event>) {
        assert_eq!(next(events), Event::Recovered);
        assert_eq!(next(events), Event::Dropped);
        let state = slot.state();
        let State::Running(running) = &*state else {
            panic!("a worker was running");
        };
        // the runner is dropped while the thread unwinds, its queue after
        let started = std::time::Instant::now();
        while !running.thread.is_finished() {
            assert!(started.elapsed() < GENEROUS, "the worker did not end");
            std::thread::yield_now();
        }
    }

    #[test]
    fn a_worker_that_ended_is_started_again_and_then_disabled() {
        let (tx, events) = channel();
        let slot = Slot::default();
        for restart in 1..=MAX_RESTARTS {
            // the runner's recovery panics too, which ends the thread
            submit(&slot, Job::Panic, &tx, true).unwrap();
            ended(&slot, &events);
            submit(&slot, Job::Report(restart), &tx, true).unwrap();
            assert_eq!(next(&events), Event::Ran(restart), "a new worker ran it");
            assert_eq!(*slot.restarts.lock().unwrap(), restart);
        }
        submit(&slot, Job::Panic, &tx, true).unwrap();
        ended(&slot, &events);
        // it is disabled, and says so for every job after
        for _ in 0..2 {
            let refused = slot.submit(Job::Report(9), recorder(&tx, true));
            assert!(matches!(refused, Err(Refused::Disabled(Job::Report(9)))));
        }
        assert!(matches!(*slot.state(), State::Disabled));
    }

    #[test]
    fn a_full_queue_refuses_the_newest_job() {
        let (tx, events) = channel();
        let slot = Slot::default();
        let (release, held) = channel();
        submit(&slot, Job::Hold(held), &tx, false).unwrap();
        // the queue holds two more while the first runs; how many it takes
        // depends on whether the worker took the first yet
        let mut queued = 0;
        let refused = loop {
            match slot.submit(Job::Report(queued), recorder(&tx, false)) {
                Ok(()) => queued += 1,
                Err(refused) => break refused,
            }
            assert!(queued <= QUEUE as u32 + 1, "the queue has no bound");
        };
        assert!(matches!(refused, Refused::Busy(Job::Report(_))));
        release.send(()).unwrap();
        for n in 0..queued {
            assert_eq!(next(&events), Event::Ran(n));
        }
    }

    #[test]
    fn an_idle_worker_tells_its_runner() {
        let (tx, events) = channel();
        let slot = Slot::ticking(Duration::from_millis(5));
        submit(&slot, Job::Report(1), &tx, false).unwrap();
        assert_eq!(next(&events), Event::Ran(1));
        assert_eq!(next(&events), Event::Idle);
        // it still takes jobs
        submit(&slot, Job::Report(2), &tx, false).unwrap();
        let mut seen = next(&events);
        while seen == Event::Idle {
            seen = next(&events);
        }
        assert_eq!(seen, Event::Ran(2));
    }

    #[tokio::test]
    async fn a_shutdown_ends_the_worker_and_refuses_later_jobs() {
        let (tx, events) = channel();
        let slot = Slot::default();
        submit(&slot, Job::Report(1), &tx, false).unwrap();
        assert_eq!(next(&events), Event::Ran(1));
        slot.shut_down().await;
        // the thread was joined: its runner, and what it held, is gone
        assert_eq!(next(&events), Event::Dropped);
        let refused = slot.submit(Job::Report(2), recorder(&tx, false));
        assert!(matches!(refused, Err(Refused::ShutDown(Job::Report(2)))));
        // a second shutdown has nothing to do
        slot.shut_down().await;
    }

    #[tokio::test]
    async fn a_shutdown_drops_the_jobs_still_waiting() {
        let (tx, events) = channel();
        let slot = Slot::default();
        let (release, held) = channel();
        submit(&slot, Job::Hold(held), &tx, false).unwrap();
        let _ = submit(&slot, Job::Report(7), &tx, false);
        // stopped while the first job still runs
        let stopping = slot.stop().expect("a running worker");
        // the worker may not have taken the first job yet, and then never does
        let _ = release.send(());
        stopping.wait().await;
        assert_eq!(next(&events), Event::Dropped, "the waiting job did not run");
    }

    #[test]
    fn dropping_the_audits_ends_the_worker() {
        let (tx, events) = channel();
        let slot = Slot::default();
        submit(&slot, Job::Report(1), &tx, false).unwrap();
        assert_eq!(next(&events), Event::Ran(1));
        drop(slot);
        assert_eq!(next(&events), Event::Dropped);
    }
}

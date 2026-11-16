//! The audit workers of a node: a few threads, so one slow audit does not
//! make the turns drawn behind it wait past their challenge window.
//!
//! Each worker keeps its own claimed models open, since a loaded model
//! stays on the thread that loaded it. A drawn turn goes to the worker
//! that holds the fewest jobs, and to the first of those: a node that
//! audits one turn at a time uses one worker, and only that one keeps
//! models open.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::worker::{Refused, Runner, Slot};

/// The most audit workers a node runs (`[receipts] audit_threads`).
pub(super) const MAX_WORKERS: usize = mesh_llm_config::MAX_AUDIT_THREADS as usize;

/// The audit workers of one node, each started when its first job comes.
pub(super) struct Pool<J> {
    workers: Vec<Slot<J>>,
}

impl<J> Default for Pool<J> {
    fn default() -> Self {
        Self {
            workers: (0..MAX_WORKERS).map(|_| Slot::default()).collect(),
        }
    }
}

impl<J: Send + 'static> Pool<J> {
    /// Queues `job` for one of the first `threads` workers: the one that
    /// holds the fewest jobs. `runner` makes the runner of a worker that
    /// has to be started. A job no worker takes comes back: busy when a
    /// queue was full, else for the reason the workers are off.
    pub(super) fn submit<R, F>(&self, job: J, threads: usize, runner: F) -> Result<(), Refused<J>>
    where
        R: Runner<Job = J>,
        F: Fn(Arc<AtomicBool>) -> R + Clone + Send + 'static,
    {
        let threads = threads.clamp(1, self.workers.len());
        let mut workers: Vec<&Slot<J>> = self.workers[..threads].iter().collect();
        // a stable sort: among workers that hold as few, the first takes it
        workers.sort_by_key(|worker| worker.held());
        let mut job = job;
        let mut busy = false;
        for worker in workers {
            job = match worker.submit(job, runner.clone()) {
                Ok(()) => return Ok(()),
                // the audits are shut down as one
                Err(Refused::ShutDown(job)) => return Err(Refused::ShutDown(job)),
                Err(Refused::Busy(job)) => {
                    busy = true;
                    job
                }
                Err(Refused::Disabled(job)) => job,
            };
        }
        Err(match busy {
            true => Refused::Busy(job),
            false => Refused::Disabled(job),
        })
    }

    /// Ends the workers: all are told to stop, then waited for together, so
    /// a node with many workers in audits shuts down as fast as with one.
    pub(super) async fn shut_down(&self) {
        let stopping = self.workers.iter().filter_map(Slot::stop);
        futures_util::future::join_all(stopping.map(|worker| worker.wait())).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::thread::ThreadId;
    use std::time::Duration;

    use super::super::worker::QUEUE;
    use super::*;

    /// How long a test waits for a worker to do what it must.
    const GENEROUS: Duration = Duration::from_secs(60);

    enum Job {
        /// Says which thread ran it.
        Report,
        /// Says which thread runs it, then blocks until the test lets it go.
        Hold(Receiver<()>),
    }

    #[derive(Clone)]
    struct Reporter(Sender<ThreadId>);

    impl Runner for Reporter {
        type Job = Job;

        fn run(&mut self, job: Job) {
            let _ = self.0.send(std::thread::current().id());
            if let Job::Hold(release) = job {
                let _ = release.recv_timeout(GENEROUS);
            }
        }

        fn idle(&mut self) {}

        fn recover(&mut self) {}
    }

    fn submit(pool: &Pool<Job>, job: Job, threads: usize, ran: &Sender<ThreadId>) -> bool {
        let reporter = Reporter(ran.clone());
        pool.submit(job, threads, move |_| reporter.clone()).is_ok()
    }

    fn next(ran: &Receiver<ThreadId>) -> ThreadId {
        ran.recv_timeout(GENEROUS).expect("a job ran")
    }

    #[test]
    fn a_job_goes_to_an_idle_worker_while_another_is_in_an_audit() {
        let (tx, ran) = channel();
        let pool = Pool::default();
        let (release, held) = channel();
        assert!(submit(&pool, Job::Hold(held), 2, &tx));
        let first = next(&ran);
        // the first worker is held: the second takes the next job at once
        assert!(submit(&pool, Job::Report, 2, &tx));
        assert_ne!(next(&ran), first);
        release.send(()).unwrap();
    }

    #[test]
    fn one_job_at_a_time_stays_on_the_first_worker() {
        let (tx, ran) = channel();
        let pool = Pool::default();
        assert!(submit(&pool, Job::Report, 4, &tx));
        let first = next(&ran);
        for _ in 0..4 {
            // the worker gives the job up before the next is counted
            let waited = std::time::Instant::now();
            while pool.workers[0].held() > 0 {
                assert!(waited.elapsed() < GENEROUS, "the worker holds a job it ran");
                std::thread::yield_now();
            }
            assert!(submit(&pool, Job::Report, 4, &tx));
            assert_eq!(next(&ran), first);
        }
    }

    #[test]
    fn a_pool_takes_what_its_threads_can_hold_and_refuses_the_rest_as_busy() {
        let (tx, ran) = channel();
        let pool = Pool::default();
        let threads = 2;
        let mut releases = Vec::new();
        let mut taken = 0;
        let refused = loop {
            let (release, held) = channel();
            releases.push(release);
            let reporter = Reporter(tx.clone());
            match pool.submit(Job::Hold(held), threads, move |_| reporter.clone()) {
                Ok(()) => taken += 1,
                Err(refused) => break refused,
            }
            assert!(taken <= threads * (QUEUE + 1), "the pool has no bound");
        };
        assert!(matches!(refused, Refused::Busy(_)));
        // each worker runs one job and holds a full queue behind it
        assert!(taken >= threads * QUEUE, "took {taken}");
        let used: std::collections::HashSet<_> = (0..threads).map(|_| next(&ran)).collect();
        assert_eq!(used.len(), threads);
        // a worker the setting does not count took nothing
        assert!(pool.workers[threads..].iter().all(|w| w.held() == 0));
        assert!(ran.try_recv().is_err(), "a third worker ran a job");
        drop(releases);
    }

    #[tokio::test]
    async fn a_shutdown_ends_every_worker_and_refuses_later_jobs() {
        let (tx, ran) = channel();
        let pool = Pool::default();
        let (release, held) = channel();
        assert!(submit(&pool, Job::Hold(held), 2, &tx));
        next(&ran);
        assert!(submit(&pool, Job::Report, 2, &tx));
        next(&ran);
        release.send(()).unwrap();
        pool.shut_down().await;
        let reporter = Reporter(tx.clone());
        let refused = pool.submit(Job::Report, 2, move |_| reporter.clone());
        assert!(matches!(refused, Err(Refused::ShutDown(_))));
    }
}

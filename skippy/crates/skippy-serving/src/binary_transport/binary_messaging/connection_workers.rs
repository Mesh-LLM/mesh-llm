//! Own stage connection workers and interrupt their I/O during teardown.
use crate::stage_transport::StageConnection;
use anyhow::{Result, bail};
use std::{
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

pub(super) const WORKER_SHUTDOWN_POLL: Duration = Duration::from_millis(100);
const EINVAL: i32 = 22;

#[derive(Default)]
pub(super) struct ConnectionWorkerControl {
    shutting_down: AtomicBool,
    sockets: Mutex<Vec<StageConnection>>,
}

impl ConnectionWorkerControl {
    pub(super) fn track(&self, stream: &StageConnection) -> io::Result<()> {
        let tracked = stream.try_clone()?;
        let mut sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        if self.shutting_down.load(Ordering::Acquire) {
            let _ = tracked.shutdown(std::net::Shutdown::Both);
        }
        sockets.push(tracked);
        Ok(())
    }

    pub(super) fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        for socket in sockets.iter() {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    }

    pub(super) fn clear(&self) {
        self.sockets
            .lock()
            .expect("connection sockets lock poisoned")
            .clear();
    }

    pub(super) fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Block until `stream` has readable data (or EOF), returning false when
    /// shutdown is requested instead. Peeks under a short read timeout so no
    /// message bytes are consumed and the worker never sits in an
    /// uninterruptible blocking read: `StageConnection::shutdown` on a tracked
    /// clone does not unblock an in-flight `read` on Windows (#1538).
    ///
    /// Darwin quirk: when a blocking socket has `SO_RCVTIMEO` set, an expired
    /// timeout surfaces from `peek` as `EINVAL` (os error 22), not
    /// `WouldBlock`/`TimedOut`. Treating EINVAL as fatal here killed healthy
    /// stage connections on their first idle poll (two-node split smoke:
    /// "wait for the first binary stage message: Invalid argument (os error
    /// 22)"). We therefore tolerate EINVAL only while our own read timeout is
    /// provably in effect — `timeout_armed` is captured when we set it, so the
    /// guard cannot drift from the socket state — and a timeout-less EINVAL
    /// still fails the connection loudly.
    ///
    /// The tail `set_read_timeout(None)` is best-effort for the same reason:
    /// during teardown Darwin can reject the clear with EINVAL too, and that
    /// cleanup failure must not flip an already-decided poll outcome into a
    /// connection error. For the same teardown window, if even arming the
    /// timeout fails with EINVAL while shutdown is in progress, report
    /// `Ok(false)` — the worker is winding down either way.
    pub(super) fn wait_for_readable(&self, stream: &StageConnection) -> io::Result<bool> {
        let timeout_armed = match stream.set_read_timeout(Some(WORKER_SHUTDOWN_POLL)) {
            Ok(()) => true,
            // Darwin rejects SO_RCVTIMEO operations with EINVAL (os error 22)
            // on a socket that has been shut down for teardown. If the worker
            // is shutting down, that is the expected outcome, not an error.
            Err(_) if self.is_shutting_down() => return Ok(false),
            Err(error) => return Err(error),
        };
        let mut probe = [0u8; 1];
        let ready = loop {
            if self.is_shutting_down() {
                break false;
            }
            let probe_started = Instant::now();
            match stream.peek(&mut probe) {
                Ok(_) => break true,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Interrupted
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                    ) => {}
                Err(error)
                    if timeout_armed
                        && error.raw_os_error() == Some(EINVAL)
                        && probe_started.elapsed() >= WORKER_SHUTDOWN_POLL => {}
                Err(error) => {
                    let _ = stream.set_read_timeout(None);
                    return Err(error);
                }
            }
        };
        // Best-effort cleanup: on Darwin, clearing SO_RCVTIMEO during
        // connection teardown can itself fail with EINVAL once the peer (or
        // our own shutdown path) has closed the socket. The poll outcome is
        // already established at this point, so a failing timeout clear must
        // not flip a healthy idle shutdown into "binary stage connection
        // failed: Invalid argument (os error 22)".
        let _ = stream.set_read_timeout(None);
        Ok(ready)
    }
}

pub(super) struct ConnectionWorker {
    pub(super) control: Arc<ConnectionWorkerControl>,
    pub(super) task: JoinHandle<()>,
}

#[derive(Default)]
pub(super) struct ConnectionWorkers(pub(super) Vec<ConnectionWorker>);

impl ConnectionWorkers {
    pub(super) fn push(&mut self, worker: ConnectionWorker) {
        self.0.push(worker);
    }

    pub(super) fn reap_finished(&mut self) -> usize {
        let mut panicked = 0;
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index].task.is_finished() {
                let worker = self.0.swap_remove(index);
                if worker.task.join().is_err() {
                    // A panicked worker only fails its own connection; the
                    // accept loop must keep serving other clients.
                    panicked += 1;
                }
            } else {
                index += 1;
            }
        }
        panicked
    }

    pub(super) fn shutdown(mut self) -> Result<()> {
        for worker in &self.0 {
            worker.control.shutdown();
        }
        let mut panicked = false;
        for worker in self.0.drain(..) {
            panicked |= worker.task.join().is_err();
        }
        if panicked {
            bail!("binary stage connection worker panicked during shutdown");
        }
        Ok(())
    }
}

pub(super) fn finish_connection_workers(
    accept_result: Result<()>,
    connection_workers: ConnectionWorkers,
) -> Result<()> {
    let shutdown_result = connection_workers.shutdown();
    match (accept_result, shutdown_result) {
        (Ok(()), result) => result,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(shutdown_error)) => Err(error.context(format!(
            "connection worker shutdown also failed: {shutdown_error:#}"
        ))),
    }
}

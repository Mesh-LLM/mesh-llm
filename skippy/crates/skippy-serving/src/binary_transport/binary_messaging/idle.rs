//! Idle retention policy for admitted stage connections.

use std::{io, net::TcpStream, sync::atomic::Ordering, time::Duration};

use super::{ConnectionWorkerControl, wait_for_first_message};

// Ordinary connections get the same initial budget as the handshake. Keep a
// separate name so subsequent-message policy can evolve independently.
const ORDINARY_IDLE_DEADLINE: Duration = super::STAGE_HANDSHAKE_DEADLINE;
// Pooled lanes may be quiet between requests. Ten minutes retains warm lanes
// across short gaps; checkout reconnects lanes closed after longer gaps.
const PERSISTENT_LANE_IDLE_DEADLINE: Duration = Duration::from_secs(600);
// Once a frame has started arriving, the rest of it must arrive within the
// same budget the first frame gets, so a peer cannot stall mid-frame and pin
// the worker.
const FRAME_DECODE_DEADLINE: Duration = super::STAGE_HANDSHAKE_DEADLINE;

pub(super) struct IdleDeadlines {
    pub(super) ordinary: Duration,
    pub(super) persistent_lane: Duration,
    pub(super) frame_decode: Duration,
}

impl Default for IdleDeadlines {
    fn default() -> Self {
        Self {
            ordinary: ORDINARY_IDLE_DEADLINE,
            persistent_lane: PERSISTENT_LANE_IDLE_DEADLINE,
            frame_decode: FRAME_DECODE_DEADLINE,
        }
    }
}

impl ConnectionWorkerControl {
    /// The reader handed a message to the executor; the connection is busy
    /// until the executor comes back for the next one.
    pub(super) fn request_started(&self) {
        self.requests_in_flight.fetch_add(1, Ordering::AcqRel);
    }

    /// The executor finished the message it held and is asking for more.
    pub(super) fn request_finished(&self) {
        let _ =
            self.requests_in_flight
                .try_update(Ordering::AcqRel, Ordering::Acquire, |in_flight| {
                    Some(in_flight.saturating_sub(1))
                });
    }

    /// Waits for the next message. The idle deadline applies only while no
    /// request is in flight: a long request legitimately leaves the upstream
    /// socket quiet, and closing it then would tear down work in progress.
    pub(super) fn wait_for_idle_message(&self, stream: &TcpStream) -> io::Result<bool> {
        let persistent_lane = self.persistent_lane.load(Ordering::Acquire);
        if self.requests_in_flight.load(Ordering::Acquire) > 0 {
            return self.wait_for_readable_while_busy(stream);
        }
        let deadline = if persistent_lane {
            self.idle_deadlines.persistent_lane
        } else {
            self.idle_deadlines.ordinary
        };
        match self.wait_for_readable(stream, Some(deadline)) {
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                tracing::debug!(persistent_lane, "closing idle stage connection");
                Ok(false)
            }
            result => result,
        }
    }

    /// Wait without a deadline while a request is in flight, re-checking each
    /// ordinary-deadline slice so the idle deadline resumes once the executor
    /// finishes and the connection really is idle.
    fn wait_for_readable_while_busy(&self, stream: &TcpStream) -> io::Result<bool> {
        loop {
            match self.wait_for_readable(stream, Some(self.idle_deadlines.ordinary)) {
                Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                    if self.requests_in_flight.load(Ordering::Acquire) == 0 {
                        return self.wait_for_idle_message(stream);
                    }
                }
                result => return result,
            }
        }
    }
}

pub(super) fn first_message_prefix(
    control: &ConnectionWorkerControl,
    stream: &mut TcpStream,
    deadline: Duration,
) -> io::Result<Option<Vec<u8>>> {
    if !wait_for_first_message(control, stream, deadline)? {
        return Ok(None);
    }
    match crate::binary_transport::connection_mode::opening_word(stream, deadline) {
        Ok(Some(word)) => Ok(Some(word.to_vec())),
        Ok(None) => {
            control.persistent_lane.store(true, Ordering::Release);
            control
                .wait_for_idle_message(stream)
                .map(|ready| (ready && !control.is_shutting_down()).then(Vec::new))
        }
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
    }
}

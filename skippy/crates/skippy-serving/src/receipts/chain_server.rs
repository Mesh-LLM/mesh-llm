//! Serves the receipts connections the stage before this one opens on the
//! binary stage listener ([`super::downstream`]).
//!
//! Anything that reaches the stage port can open such a connection, so it
//! gets little for free: a stage with receipts off answers once that it
//! has none and hangs up; a stage serves a few receipts connections at a
//! time; a connection that sends nothing for a while is closed, and so is
//! the one that waited longest for a request when a new connection finds
//! every slot held; and a request is admitted like one on the HTTP API, a
//! bound turn only with its client's proof ([`super::admission`]).
//!
//! The stage port does not authenticate its peer: that is out of scope
//! here, and an operator keeps the port to the stages of the pipeline. What
//! a peer that reaches it can still do is described in the receipts
//! design document.

use std::io::Write;
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use skippy_protocol::binary::{
    ReceiptsReply, StageWireMessage, WireMessageKind, read_stage_message, send_receipts_reply,
};

use super::downstream::ChainRequest;
use super::refusal::{self, Unavailable};
use super::{SessionTurns, StageReceipts};

/// Receipts connections a stage serves at once. The stage before it keeps
/// two, and opens two more when one breaks mid-request.
pub(super) const RECEIPTS_CONNECTIONS: usize = 8;
/// A receipts connection that sends no request for this long is closed.
/// The stage before opens another when it next needs one.
const IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// A reply the stage before does not take in this long ends the connection.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// How often a connection that waits for a request looks whether it lost
/// its slot. A shutdown from another thread ends a blocked read on some
/// platforms and not on others, so the wait does not rely on it.
const SLOT_CHECK: Duration = Duration::from_millis(200);

/// The receipts connections a stage serves.
#[derive(Default)]
pub(super) struct ConnectionSlots(Mutex<Connections>);

#[derive(Default)]
struct Connections {
    held: Vec<Held>,
    next_id: u64,
}

struct Held {
    id: u64,
    /// Since when the connection waits for a request; `None` while it is
    /// answering one.
    idle_since: Option<Instant>,
    /// A handle that closes the connection.
    stream: TcpStream,
}

/// One served connection; frees its slot when dropped.
struct Slot {
    slots: Arc<ConnectionSlots>,
    id: u64,
}

impl ConnectionSlots {
    fn lock(&self) -> MutexGuard<'_, Connections> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A slot for the connection `stream`. When every slot is held, the
    /// connection that has waited longest for a request is closed for it,
    /// so connections that only hold a slot cannot keep out the stage
    /// before; `None` when every connection is answering a request.
    fn acquire(self: &Arc<Self>, stream: &TcpStream) -> Result<Option<Slot>> {
        let stream = stream.try_clone()?;
        let mut connections = self.lock();
        if connections.held.len() >= RECEIPTS_CONNECTIONS {
            let Some(oldest) = connections
                .held
                .iter()
                .enumerate()
                .filter_map(|(i, held)| Some((held.idle_since?, i)))
                .min()
                .map(|(_, i)| i)
            else {
                return Ok(None);
            };
            let evicted = connections.held.swap_remove(oldest);
            // its peer sees the end at once; its own wait sees the lost
            // slot at its next look ([`wait_for_request`])
            let _ = evicted.stream.shutdown(Shutdown::Both);
        }
        let id = connections.next_id;
        connections.next_id += 1;
        connections.held.push(Held {
            id,
            idle_since: None,
            stream,
        });
        Ok(Some(Slot {
            slots: self.clone(),
            id,
        }))
    }

    #[cfg(test)]
    pub(super) fn held(&self) -> usize {
        self.lock().held.len()
    }
}

impl Slot {
    /// Notes whether the connection waits for a request.
    fn set_idle(&self, idle: bool) {
        let mut connections = self.slots.lock();
        if let Some(held) = connections.held.iter_mut().find(|h| h.id == self.id) {
            held.idle_since = idle.then(Instant::now);
        }
    }
}

impl Slot {
    /// Whether the connection still has its slot; not once a newer
    /// connection took it.
    fn held(&self) -> bool {
        let connections = self.slots.lock();
        connections.held.iter().any(|held| held.id == self.id)
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.slots.lock().held.retain(|held| held.id != self.id);
    }
}

/// Serves a receipts connection that an earlier stage opened with `first`,
/// until that stage closes it or stops using it.
pub(crate) fn serve_connection(
    receipts: Option<Arc<StageReceipts>>,
    sessions: &dyn SessionTurns,
    upstream: TcpStream,
    first: StageWireMessage,
) -> Result<()> {
    serve_connection_until_idle(receipts, sessions, upstream, first, IDLE_TIMEOUT)
}

pub(super) fn serve_connection_until_idle(
    receipts: Option<Arc<StageReceipts>>,
    sessions: &dyn SessionTurns,
    mut upstream: TcpStream,
    first: StageWireMessage,
    idle: Duration,
) -> Result<()> {
    upstream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let Some(receipts) = receipts else {
        // the stage before learns it once, and asks no more
        let off = ReceiptsReply::Unavailable("this stage issues no receipts".to_owned());
        return send(&mut upstream, &off);
    };
    let Some(slot) = receipts.connections.acquire(&upstream)? else {
        let full = ReceiptsReply::Refused(format!(
            "this stage is answering requests on {RECEIPTS_CONNECTIONS} receipts connections"
        ));
        return send(&mut upstream, &full);
    };
    // the connection must not keep an unloaded stage's receipts alive
    let receipts = {
        let weak = Arc::downgrade(&receipts);
        drop(receipts);
        weak
    };
    let mut message = first;
    loop {
        send(&mut upstream, &answer(&receipts, sessions, &message))?;
        slot.set_idle(true);
        if !wait_for_request(&upstream, &slot, idle)? {
            return Ok(());
        }
        upstream.set_read_timeout(Some(idle))?;
        message = match read_stage_message(&mut upstream, 0) {
            Ok(message) => message,
            Err(error) if closed_or_idle(&error) => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        slot.set_idle(false);
    }
}

/// Waits until the stage before sends its next request. `false` when the
/// connection ended instead: the peer closed it, it sent nothing for
/// `idle`, or a newer connection took its slot.
fn wait_for_request(upstream: &TcpStream, slot: &Slot, idle: Duration) -> Result<bool> {
    let waiting_since = Instant::now();
    let mut first_byte = [0u8; 1];
    loop {
        let left = idle.saturating_sub(waiting_since.elapsed());
        if left.is_zero() || !slot.held() {
            return Ok(false);
        }
        upstream.set_read_timeout(Some(left.min(SLOT_CHECK)))?;
        match upstream.peek(&mut first_byte) {
            Ok(0) => return Ok(false),
            Ok(_) => return Ok(true),
            Err(error) if timed_out(&error) => {}
            Err(error) if closed_or_idle(&error) => return Ok(false),
            Err(error) => return Err(error.into()),
        }
    }
}

fn timed_out(error: &std::io::Error) -> bool {
    use std::io::ErrorKind::{TimedOut, WouldBlock};
    matches!(error.kind(), TimedOut | WouldBlock)
}

fn closed_or_idle(error: &std::io::Error) -> bool {
    use std::io::ErrorKind::{TimedOut, UnexpectedEof, WouldBlock};
    matches!(error.kind(), UnexpectedEof | TimedOut | WouldBlock)
}

fn send(upstream: &mut TcpStream, reply: &ReceiptsReply) -> Result<()> {
    send_receipts_reply(&mut *upstream, reply)?;
    upstream.flush()?;
    Ok(())
}

fn answer(
    receipts: &Weak<StageReceipts>,
    sessions: &dyn SessionTurns,
    message: &StageWireMessage,
) -> ReceiptsReply {
    let answered = (|| -> Result<Vec<u8>> {
        let receipts = receipts
            .upgrade()
            .ok_or_else(|| Unavailable("this stage was unloaded".to_owned()))?;
        if message.kind != WireMessageKind::Receipts {
            bail!("a receipts connection carries only receipts requests");
        }
        let request =
            serde_json::from_slice(&message.raw_bytes).context("decode a receipts request")?;
        Ok(match request {
            ChainRequest::Commit(request) => {
                serde_json::to_vec(&receipts.commit_session(sessions, &request)?)?
            }
            ChainRequest::Open(request) => serde_json::to_vec(&receipts.open_forwarded(&request)?)?,
            ChainRequest::Rows(request) => serde_json::to_vec(&receipts.rows_forwarded(&request)?)?,
        })
    })();
    match answered {
        Ok(body) => ReceiptsReply::Answer(body),
        Err(error) => refusal::reply(&error),
    }
}

#[cfg(test)]
#[path = "chain_server_tests.rs"]
mod tests;

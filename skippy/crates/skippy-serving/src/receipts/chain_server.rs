//! Serves the receipts connections the stage before this one opens on the
//! binary stage listener ([`super::downstream`]).
//!
//! Anything that reaches the stage port can open such a connection, so it
//! gets little for free: a stage with receipts off answers once that it
//! has none and hangs up; a stage serves a few receipts connections at a
//! time; a connection that sends nothing for a while is closed; and a
//! request is admitted like one on the HTTP API, a bound turn only with its
//! client's proof ([`super::admission`]).

use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

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

/// Counts the receipts connections a stage serves.
#[derive(Default)]
pub(super) struct ConnectionSlots(AtomicUsize);

/// One served connection; frees its slot when dropped.
struct Slot(Arc<ConnectionSlots>);

impl ConnectionSlots {
    fn acquire(self: &Arc<Self>) -> Option<Slot> {
        self.0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |held| {
                (held < RECEIPTS_CONNECTIONS).then_some(held + 1)
            })
            .ok()
            .map(|_| Slot(self.clone()))
    }

    #[cfg(test)]
    pub(super) fn held(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.0.fetch_sub(1, Ordering::SeqCst);
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
    let Some(_slot) = receipts.connections.acquire() else {
        let full = ReceiptsReply::Refused(format!(
            "this stage already serves {RECEIPTS_CONNECTIONS} receipts connections"
        ));
        return send(&mut upstream, &full);
    };
    // the connection must not keep an unloaded stage's receipts alive
    let receipts = {
        let weak = Arc::downgrade(&receipts);
        drop(receipts);
        weak
    };
    upstream.set_read_timeout(Some(idle))?;
    let mut message = first;
    loop {
        send(&mut upstream, &answer(&receipts, sessions, &message))?;
        message = match read_stage_message(&mut upstream, 0) {
            Ok(message) => message,
            Err(error) if closed_or_idle(&error) => return Ok(()),
            Err(error) => return Err(error.into()),
        };
    }
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

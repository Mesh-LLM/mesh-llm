//! Accept-loop error handling and connection admission for the OpenAI
//! ingress listeners.
//!
//! A bound `TcpListener` stays valid across every error `accept` can report:
//! the errors describe one connection or a momentary shortage of host
//! resources, never the listening socket itself. So an accept loop has no
//! reason to ever give the listener up, and #1703 is what happens when it
//! does. A joined node's `:9338` listener disappeared after hours of uptime
//! while the process, the console, and mesh gossip all kept running, so peers
//! went on routing to an API surface that no longer accepted anything.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// How long to wait before accepting again after an error that is not a
/// plain per-connection failure. Short enough that a node recovers promptly
/// once file descriptors free up, long enough that the loop does not spin.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(100);

/// What an accept loop should do after `accept` returns an error.
///
/// There is deliberately no variant that stops the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcceptRecovery {
    /// One inbound connection failed. The next accept can run immediately and
    /// is not worth logging: clients abort mid-handshake all the time.
    Retry,
    /// Anything else, which in practice means host resource pressure such as
    /// a per-process descriptor limit. Wait before accepting again.
    Backoff(Duration),
}

/// Classifies an accept error. Every input produces a recovery, by design.
pub(crate) fn recover_from_accept_error(error: &io::Error) -> AcceptRecovery {
    match error.kind() {
        io::ErrorKind::ConnectionAborted
        | io::ErrorKind::ConnectionReset
        | io::ErrorKind::Interrupted
        | io::ErrorKind::WouldBlock => AcceptRecovery::Retry,
        _ => AcceptRecovery::Backoff(ACCEPT_BACKOFF),
    }
}

/// Waits out an accept error so the caller can loop again.
///
/// Neither ingress loop drops its listener on an accept error. The listening
/// socket survives every error `accept` reports, so returning here is how
/// #1703 silently removed `:9338` from a node that otherwise looked healthy.
/// Resource pressure is logged, because the previous code swallowed it and
/// left nothing behind to explain the missing listener.
pub(super) async fn recover_accept_loop(port: u16, error: &std::io::Error, surface: &'static str) {
    match recover_from_accept_error(error) {
        AcceptRecovery::Retry => {}
        AcceptRecovery::Backoff(delay) => {
            tracing::warn!(
                port,
                surface,
                error = %error,
                backoff_ms = delay.as_millis() as u64,
                "accept failed; retrying without dropping the listener"
            );
            tokio::time::sleep(delay).await;
        }
    }
}

pub(super) async fn bind_api_proxy_listener(
    port: u16,
    existing_listener: Option<IngressListener>,
    listen_all: bool,
) -> Option<IngressListener> {
    match existing_listener {
        Some(listener) => Some(listener),
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(listener) => Some(listener.into()),
                Err(error) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {error}");
                    None
                }
            }
        }
    }
}

/// Most connections one ingress listener serves at once. Each connection holds
/// a task and a socket until its request finishes, so an unbounded count lets
/// a client that opens connections faster than they complete exhaust both.
pub(crate) const MAX_INGRESS_CONNECTIONS: usize = 1024;

/// Tells a connection refused for capacity why, without waiting on it, then
/// closes it. The response is tiny, so a non-blocking write fits in the socket
/// buffer; if it does not, the connection is closed without one.
pub(crate) fn refuse_over_capacity(stream: tokio::net::TcpStream) {
    use std::io::Write;

    const RESPONSE: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n";
    // The std socket stays non-blocking, so this write never waits.
    if let Ok(mut stream) = stream.into_std() {
        let _ = stream.write(RESPONSE);
    }
}

/// Bounds how many accepted connections an ingress listener serves at once.
#[derive(Clone)]
pub(crate) struct ConnectionSlots(Arc<Semaphore>);

/// An ingress listener and the connection slots it admits against.
///
/// When the bootstrap proxy hands its listener to the API proxy, the
/// connections it is still serving hold slots. The slots travel with the
/// listener, so those connections keep counting against the limit instead of
/// the API proxy starting over with a full set.
pub(crate) struct IngressListener {
    pub(crate) listener: tokio::net::TcpListener,
    pub(crate) slots: ConnectionSlots,
}

impl From<tokio::net::TcpListener> for IngressListener {
    /// A freshly bound listener, with every slot free.
    fn from(listener: tokio::net::TcpListener) -> Self {
        Self {
            listener,
            slots: ConnectionSlots::new(MAX_INGRESS_CONNECTIONS),
        }
    }
}

impl ConnectionSlots {
    pub(crate) fn new(limit: usize) -> Self {
        Self(Arc::new(Semaphore::new(limit)))
    }

    /// Claims a slot for one accepted connection, or returns `None` when every
    /// slot is taken and the connection should be dropped. The slot frees
    /// itself when the returned permit is dropped.
    pub(crate) fn try_claim(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.0).try_acquire_owned().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn handed_off_listener_keeps_the_slots_its_connections_hold() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bootstrap = IngressListener {
            listener,
            slots: ConnectionSlots::new(1),
        };
        // A connection the bootstrap proxy is still serving at handoff.
        let still_serving = bootstrap.slots.try_claim().expect("a slot is free");

        let IngressListener { slots, .. } = bootstrap;
        assert!(
            slots.try_claim().is_none(),
            "the API proxy must not admit past the limit while handed-off connections run"
        );
        drop(still_serving);
        assert!(slots.try_claim().is_some());
    }

    fn error(kind: io::ErrorKind) -> io::Error {
        io::Error::new(kind, "accept failed")
    }

    #[test]
    fn aborted_connections_retry_immediately() {
        for kind in [
            io::ErrorKind::ConnectionAborted,
            io::ErrorKind::ConnectionReset,
            io::ErrorKind::Interrupted,
            io::ErrorKind::WouldBlock,
        ] {
            assert_eq!(
                recover_from_accept_error(&error(kind)),
                AcceptRecovery::Retry,
                "{kind:?} is a per-connection failure"
            );
        }
    }

    #[test]
    fn descriptor_exhaustion_backs_off_instead_of_spinning() {
        // EMFILE is the error a long-running node hits first, and it has no
        // stable ErrorKind, so it arrives as Uncategorized.
        let emfile = io::Error::from_raw_os_error(libc::EMFILE);

        assert_eq!(
            recover_from_accept_error(&emfile),
            AcceptRecovery::Backoff(ACCEPT_BACKOFF)
        );
    }

    #[test]
    fn unrecognized_errors_never_stop_the_loop() {
        for kind in [
            io::ErrorKind::Other,
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::OutOfMemory,
            io::ErrorKind::InvalidInput,
        ] {
            assert!(
                matches!(
                    recover_from_accept_error(&error(kind)),
                    AcceptRecovery::Backoff(_)
                ),
                "{kind:?} must not be treated as fatal"
            );
        }
    }

    #[tokio::test]
    async fn refused_connections_get_a_503_instead_of_a_silent_close() {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();

        refuse_over_capacity(server);

        let mut response = String::new();
        client.read_to_string(&mut response).await.unwrap();
        assert!(
            response.starts_with("HTTP/1.1 503 "),
            "unexpected refusal response: {response:?}"
        );
    }

    #[test]
    fn connection_slots_refuse_connections_past_the_limit() {
        let slots = ConnectionSlots::new(2);
        let first = slots.try_claim().expect("first slot");
        let _second = slots.try_claim().expect("second slot");
        assert!(slots.try_claim().is_none(), "a third connection is refused");

        drop(first);
        assert!(
            slots.try_claim().is_some(),
            "a finished connection frees its slot"
        );
    }

    #[test]
    fn backoff_is_bounded_so_recovery_stays_prompt() {
        assert!(ACCEPT_BACKOFF <= Duration::from_millis(250));
        assert!(ACCEPT_BACKOFF > Duration::ZERO);
    }
}

//! Accept-loop error handling for the OpenAI ingress listeners.
//!
//! A bound `TcpListener` stays valid across every error `accept` can report:
//! the errors describe one connection or a momentary shortage of host
//! resources, never the listening socket itself. So an accept loop has no
//! reason to ever give the listener up, and #1703 is what happens when it
//! does. A joined node's `:9338` listener disappeared after hours of uptime
//! while the process, the console, and mesh gossip all kept running, so peers
//! went on routing to an API surface that no longer accepted anything.

use std::io;
use std::time::Duration;

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn backoff_is_bounded_so_recovery_stays_prompt() {
        assert!(ACCEPT_BACKOFF <= Duration::from_millis(250));
        assert!(ACCEPT_BACKOFF > Duration::ZERO);
    }
}

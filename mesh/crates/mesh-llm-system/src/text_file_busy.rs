//! Retry for tests that write a script and then run it.
//!
//! A process another test thread forks while the script is being written
//! holds its write descriptor until that process execs, and running the
//! script fails with "Text file busy" until then (Mesh-LLM/mesh-llm#2716).

use std::time::Duration;

/// Runs `run` again, every 50 ms for up to 5 s, while it fails because the
/// script it starts is still open for writing somewhere.
pub(crate) fn retry<T>(mut run: impl FnMut() -> anyhow::Result<T>) -> anyhow::Result<T> {
    let mut attempts = 0;
    loop {
        match run() {
            Err(err) if attempts < 100 && is_text_file_busy(&err) => {
                attempts += 1;
                std::thread::sleep(Duration::from_millis(50));
            }
            result => return result,
        }
    }
}

fn is_text_file_busy(err: &anyhow::Error) -> bool {
    err.downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::ExecutableFileBusy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use std::io::{Error, ErrorKind};

    #[test]
    fn a_busy_start_is_retried_until_it_succeeds() {
        let mut calls = 0;
        let result = retry(|| {
            calls += 1;
            if calls < 3 {
                Err(Error::from(ErrorKind::ExecutableFileBusy)).context("start the script")
            } else {
                Ok(calls)
            }
        });
        assert_eq!(result.unwrap(), 3);
    }

    #[test]
    fn other_errors_are_returned_at_once() {
        let mut calls = 0;
        let result: anyhow::Result<()> = retry(|| {
            calls += 1;
            Err(Error::from(ErrorKind::PermissionDenied)).context("start the script")
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}

//! Redirects the process's stdout and stderr into the dashboard while it owns
//! the screen.
//!
//! Converting individual `println!`/`eprintln!` call sites cannot close this
//! hole, because the writers are not all ours: `plugin/runtime.rs` hands
//! spawned plugins `Stdio::inherit()`, the staged llama.cpp runtime is C, and
//! third-party crates print whatever they like. All of them write to fd 1 or
//! fd 2, so that is where the interception belongs.
//!
//! While installed, fd 1 and fd 2 point at a pipe. A reader thread turns each
//! line into an `OutputEvent`, so stray output shows up as a dashboard event
//! instead of painting over the frame. On restore the original descriptors are
//! put back and any still-buffered lines are written to the real stderr so
//! nothing is silently swallowed.
//!
//! This is only safe because the dashboard renders to the controlling terminal
//! (see [`super::terminal_out`]) rather than to fd 2; installing capture while
//! the dashboard still rendered to stderr would redirect the dashboard into its
//! own pipe.

#[cfg(unix)]
pub(in crate::output) use unix::ConsoleCapture;

#[cfg(not(unix))]
pub(in crate::output) use fallback::ConsoleCapture;

#[cfg(unix)]
mod unix {
    use mesh_llm_events::{OutputEvent, emit_event};
    use rustix::io::fcntl_dupfd_cloexec;
    use rustix::stdio::{dup2_stderr, dup2_stdout, stderr, stdout};
    use std::fs::File;
    use std::io::{self, BufRead, BufReader, PipeReader, Write};
    use std::os::fd::OwnedFd;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Lines longer than this are split rather than buffered without bound, so
    /// a child spewing bytes with no newline cannot grow the reader's buffer
    /// until the process dies.
    const MAX_CAPTURED_LINE_BYTES: usize = 8 * 1024;

    pub(in crate::output) struct ConsoleCapture {
        saved_stdout: OwnedFd,
        saved_stderr: OwnedFd,
        active: Arc<AtomicBool>,
    }

    impl ConsoleCapture {
        /// Point fd 1 and fd 2 at a pipe drained into the dashboard.
        pub(in crate::output) fn install() -> io::Result<Self> {
            // Anything already buffered belongs on the real terminal, not in
            // the pipe we are about to install.
            let _ = io::stdout().flush();
            let _ = io::stderr().flush();

            // Both pipe ends are close-on-exec. `dup2` clears that flag on the
            // descriptor it installs, so children still inherit the redirected
            // fd 1/2 while the read end stays private to this process.
            let (read_fd, write_fd) = io::pipe()?;

            let saved_stdout = fcntl_dupfd_cloexec(stdout(), 0)?;
            let saved_stderr = fcntl_dupfd_cloexec(stderr(), 0)?;
            // The reader keeps its own handle on the real stderr so it can
            // still deliver lines after the dashboard goes away.
            let reader_stderr = fcntl_dupfd_cloexec(stderr(), 0)?;

            dup2_stdout(&write_fd)?;
            dup2_stderr(&write_fd)?;
            drop(write_fd);

            let active = Arc::new(AtomicBool::new(true));
            spawn_reader(read_fd, reader_stderr, Arc::clone(&active));

            Ok(Self {
                saved_stdout,
                saved_stderr,
                active,
            })
        }

        /// Put the original descriptors back. Safe to call more than once.
        pub(in crate::output) fn restore(&mut self) -> io::Result<()> {
            if !self.active.swap(false, Ordering::Release) {
                return Ok(());
            }
            let _ = io::stdout().flush();
            let _ = io::stderr().flush();
            dup2_stdout(&self.saved_stdout)?;
            dup2_stderr(&self.saved_stderr)?;
            Ok(())
        }
    }

    impl Drop for ConsoleCapture {
        fn drop(&mut self) {
            let _ = self.restore();
        }
    }

    /// The reader is deliberately detached rather than joined. A plugin child
    /// that inherited the write end keeps the pipe open past restore, so a join
    /// could block shutdown indefinitely; the thread instead exits on EOF
    /// whenever that arrives, and dies with the process at worst.
    fn spawn_reader(read_fd: PipeReader, original_stderr: OwnedFd, active: Arc<AtomicBool>) {
        let _ = std::thread::Builder::new()
            .name("mesh-console-capture".to_string())
            .spawn(move || {
                let mut passthrough = File::from(original_stderr);
                let mut reader = BufReader::new(read_fd);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match read_bounded_line(&mut reader, &mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let text = String::from_utf8_lossy(&line)
                        .trim_end_matches(['\r', '\n'])
                        .to_string();
                    if text.trim().is_empty() {
                        continue;
                    }
                    if active.load(Ordering::Acquire) {
                        // Failure here means the dashboard sink is gone; fall
                        // back to the real terminal rather than dropping it.
                        if emit_event(dashboard_event(text.clone())).is_ok() {
                            continue;
                        }
                    }
                    let _ = writeln!(passthrough, "{text}");
                }
            });
    }

    /// Captured output has no level of its own. Anything that looks like a
    /// complaint is surfaced as a warning so it is not lost among info rows;
    /// the `stdout` context tells the reader it was intercepted rather than
    /// emitted through the normal event path.
    pub(super) fn dashboard_event(message: String) -> OutputEvent {
        let lowered = message.to_ascii_lowercase();
        let looks_like_a_problem = ["error", "warn", "failed", "panic"]
            .iter()
            .any(|needle| lowered.contains(needle));
        let context = Some("stdout".to_string());
        if looks_like_a_problem {
            OutputEvent::Warning { message, context }
        } else {
            OutputEvent::Info { message, context }
        }
    }

    /// Like `read_until(b'\n')`, but refuses to grow past
    /// `MAX_CAPTURED_LINE_BYTES` so unterminated output cannot exhaust memory.
    pub(super) fn read_bounded_line<R: BufRead>(
        reader: &mut R,
        line: &mut Vec<u8>,
    ) -> io::Result<usize> {
        let mut total = 0usize;
        loop {
            let (consumed, done) = {
                let available = match reader.fill_buf() {
                    Ok(buffer) => buffer,
                    Err(ref err) if err.kind() == io::ErrorKind::Interrupted => continue,
                    Err(err) => return Err(err),
                };
                if available.is_empty() {
                    return Ok(total);
                }
                let (chunk, done) = match available.iter().position(|byte| *byte == b'\n') {
                    Some(index) => (&available[..=index], true),
                    None => (available, false),
                };
                let take = chunk
                    .len()
                    .min(MAX_CAPTURED_LINE_BYTES.saturating_sub(total));
                line.extend_from_slice(&chunk[..take]);
                total += take;
                (chunk.len(), done)
            };
            reader.consume(consumed);
            if done || total >= MAX_CAPTURED_LINE_BYTES {
                return Ok(total.max(1));
            }
        }
    }
}

#[cfg(not(unix))]
mod fallback {
    use std::io;

    /// Descriptor-level capture is POSIX-specific. On other platforms the
    /// dashboard still renders to the controlling terminal, and stray output
    /// remains repairable with the `R` key.
    pub(in crate::output) struct ConsoleCapture;

    impl ConsoleCapture {
        pub(in crate::output) fn install() -> io::Result<Self> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "console capture requires a POSIX platform",
            ))
        }

        pub(in crate::output) fn restore(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::unix::{dashboard_event, read_bounded_line};
    use mesh_llm_events::OutputEvent;
    use std::io::BufReader;

    fn read_all_lines(input: &[u8]) -> Vec<String> {
        let mut reader = BufReader::new(input);
        let mut lines = Vec::new();
        loop {
            let mut line = Vec::new();
            match read_bounded_line(&mut reader, &mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => lines.push(String::from_utf8_lossy(&line).into_owned()),
            }
        }
        lines
    }

    #[test]
    fn captured_output_splits_on_newlines() {
        assert_eq!(
            read_all_lines(b"first\nsecond\nthird\n"),
            vec!["first\n", "second\n", "third\n"]
        );
    }

    #[test]
    fn captured_output_keeps_a_trailing_unterminated_line() {
        assert_eq!(
            read_all_lines(b"done\nno trailing newline"),
            vec!["done\n", "no trailing newline"]
        );
    }

    #[test]
    fn captured_output_without_newlines_cannot_grow_without_bound() {
        // A child spewing bytes with no newline must not be buffered forever.
        let flood = vec![b'x'; 64 * 1024];

        let lines = read_all_lines(&flood);

        assert!(
            lines.iter().all(|line| line.len() <= 8 * 1024),
            "an unterminated flood must be split, not accumulated"
        );
        assert_eq!(
            lines.iter().map(String::len).sum::<usize>(),
            flood.len(),
            "splitting must not drop bytes"
        );
    }

    #[test]
    fn captured_output_is_classified_by_what_it_says() {
        assert!(matches!(
            dashboard_event("llama_model_loader: loaded meta data".to_string()),
            OutputEvent::Info { .. }
        ));
        assert!(matches!(
            dashboard_event("ggml_cuda_init: failed to initialise".to_string()),
            OutputEvent::Warning { .. }
        ));
    }

    #[test]
    fn captured_output_is_labelled_as_intercepted() {
        let OutputEvent::Info { context, .. } = dashboard_event("plain line".to_string()) else {
            panic!("expected an info event");
        };
        assert_eq!(
            context.as_deref(),
            Some("stdout"),
            "the dashboard should show that this line was intercepted, not emitted"
        );
    }
}

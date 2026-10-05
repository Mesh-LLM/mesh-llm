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
//! nothing is silently swallowed. If a child keeps an old pipe open across a
//! later dashboard session, that stale reader follows the current capture
//! state and routes its output into the new dashboard.
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
    use mesh_llm_events::OutputEvent;
    use rustix::event::{PollFd, PollFlags, poll};
    use rustix::io::fcntl_dupfd_cloexec;
    use rustix::stdio::{dup2_stderr, dup2_stdout, stderr, stdout};
    use std::fs::File;
    use std::io::{self, PipeReader, Read, Write};
    use std::os::fd::OwnedFd;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// How long a partial line may sit unflushed before it is shown anyway.
    const IDLE_FLUSH: Duration = Duration::from_millis(150);

    /// Lines longer than this are split rather than buffered without bound, so
    /// a child spewing bytes with no newline cannot grow the reader's buffer
    /// until the process dies.
    const MAX_CAPTURED_LINE_BYTES: usize = 8 * 1024;

    /// Readers can outlive the dashboard session that spawned them when a
    /// child inherits a pipe write end, so delivery follows global activity.
    static ACTIVE_CAPTURES: AtomicUsize = AtomicUsize::new(0);

    pub(in crate::output) struct ConsoleCapture {
        saved_stdout: OwnedFd,
        saved_stderr: OwnedFd,
        active: bool,
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

            // Start the reader before redirecting either descriptor. Once the
            // capture is registered, stale readers from earlier children also
            // route into the active dashboard.
            register_capture_reader(&ACTIVE_CAPTURES, || spawn_reader(read_fd, reader_stderr))?;

            if let Err(error) = dup2_stdout(&write_fd) {
                unregister_capture(&ACTIVE_CAPTURES);
                return Err(error.into());
            }
            if let Err(error) = dup2_stderr(&write_fd) {
                // stdout was already redirected. Restore it before the pipe
                // handles drop so a partial install cannot strand fd 1.
                let _ = dup2_stdout(&saved_stdout);
                unregister_capture(&ACTIVE_CAPTURES);
                return Err(error.into());
            }
            drop(write_fd);

            Ok(Self {
                saved_stdout,
                saved_stderr,
                active: true,
            })
        }

        /// Put the original descriptors back. Safe to call more than once.
        pub(in crate::output) fn restore(&mut self) -> io::Result<()> {
            if !self.active {
                return Ok(());
            }
            let _ = io::stdout().flush();
            let _ = io::stderr().flush();
            dup2_stdout(&self.saved_stdout)?;
            dup2_stderr(&self.saved_stderr)?;
            // Keep restoration retryable until both descriptors are back.
            self.active = false;
            unregister_capture(&ACTIVE_CAPTURES);
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
    fn spawn_reader(mut read_fd: PipeReader, original_stderr: OwnedFd) -> io::Result<()> {
        std::thread::Builder::new()
            .name("mesh-console-capture".to_string())
            .spawn(move || {
                let mut passthrough = File::from(original_stderr);
                let mut pending = Vec::new();
                let mut buffer = [0u8; 4096];
                // Wait for data, but not forever: a writer that emitted a
                // partial line and then went quiet (`print!`, a `\r` progress
                // counter) must still be shown rather than sit in this buffer
                // until the next newline arrives.
                while let Ok(readable) = wait_for_input(&read_fd, IDLE_FLUSH) {
                    if !readable {
                        for line in take_pending_lines(&mut pending, true) {
                            deliver(line, &mut passthrough);
                        }
                        continue;
                    }
                    match read_fd.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(count) => pending.extend_from_slice(&buffer[..count]),
                    }
                    // A writer with no line breaks at all must not grow this
                    // buffer without bound.
                    let force = pending.len() >= MAX_CAPTURED_LINE_BYTES;
                    for line in take_pending_lines(&mut pending, force) {
                        deliver(line, &mut passthrough);
                    }
                }
                for line in take_pending_lines(&mut pending, true) {
                    deliver(line, &mut passthrough);
                }
            })
            .map(|_| ())
    }

    pub(super) fn register_capture_reader(
        active_captures: &AtomicUsize,
        spawn_reader: impl FnOnce() -> io::Result<()>,
    ) -> io::Result<()> {
        active_captures.fetch_add(1, Ordering::AcqRel);
        if let Err(error) = spawn_reader() {
            unregister_capture(active_captures);
            return Err(error);
        }
        Ok(())
    }

    fn unregister_capture(active_captures: &AtomicUsize) {
        let result = active_captures.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_sub(1)
        });
        debug_assert!(result.is_ok(), "capture activity count underflowed");
    }

    fn dashboard_capture_active(active_captures: &AtomicUsize) -> bool {
        active_captures.load(Ordering::Acquire) > 0
    }

    fn deliver(text: String, passthrough: &mut File) {
        deliver_with(
            text,
            &ACTIVE_CAPTURES,
            passthrough,
            mesh_llm_events::emit_event,
        );
    }

    pub(super) fn deliver_with<W, E>(
        text: String,
        active_captures: &AtomicUsize,
        passthrough: &mut W,
        emit: E,
    ) where
        W: Write,
        E: FnOnce(OutputEvent) -> io::Result<()>,
    {
        if text.trim().is_empty() {
            return;
        }
        if dashboard_capture_active(active_captures) {
            // Failure here means the dashboard sink is gone; fall back to the
            // real terminal rather than dropping the line.
            if emit(dashboard_event(text.clone())).is_ok() {
                return;
            }
        }
        let _ = writeln!(passthrough, "{text}");
    }

    /// Split `pending` into displayable lines.
    ///
    /// `\r` terminates a line as well as `\n` so carriage-return progress
    /// counters surface instead of accumulating. When `flush_remainder` is set
    /// the trailing partial line is taken too.
    pub(super) fn take_pending_lines(pending: &mut Vec<u8>, flush_remainder: bool) -> Vec<String> {
        let mut lines = Vec::new();
        while let Some(index) = pending
            .iter()
            .position(|byte| *byte == b'\n' || *byte == b'\r')
        {
            let mut line: Vec<u8> = pending.drain(..=index).collect();
            line.pop();
            lines.push(decode(&line));
        }
        if flush_remainder && !pending.is_empty() {
            let remainder = std::mem::take(pending);
            lines.push(decode(&remainder));
        }
        lines
    }

    fn decode(line: &[u8]) -> String {
        // Captured bytes are arbitrary. A raw escape sequence rendered into a
        // dashboard cell would move the cursor and corrupt the very frame this
        // exists to protect, so control characters are stripped here.
        String::from_utf8_lossy(line)
            .chars()
            .filter_map(|character| match character {
                // A tab is alignment, not damage — llama.cpp's loader lines are
                // full of them. Dropping it would run two columns together, so
                // it degrades to a space instead.
                '\t' => Some(' '),
                character if character.is_control() => None,
                character => Some(character),
            })
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    /// Block until the pipe has data or `timeout` elapses. `Ok(false)` is a
    /// timeout, `Err` means the descriptor is unusable and the reader stops.
    fn wait_for_input(read_fd: &PipeReader, timeout: Duration) -> io::Result<bool> {
        let mut fds = [PollFd::new(read_fd, PollFlags::IN)];
        let timeout = timeout.try_into().map_err(io::Error::other)?;
        loop {
            match poll(&mut fds, Some(&timeout)) {
                Ok(0) => return Ok(false),
                Ok(_) => return Ok(true),
                Err(rustix::io::Errno::INTR) => continue,
                Err(err) => return Err(err.into()),
            }
        }
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
    use super::unix::{dashboard_event, deliver_with, register_capture_reader, take_pending_lines};
    use mesh_llm_events::OutputEvent;
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;

    fn split(input: &[u8], flush_remainder: bool) -> Vec<String> {
        let mut pending = input.to_vec();
        take_pending_lines(&mut pending, flush_remainder)
    }

    #[test]
    fn captured_output_splits_on_newlines() {
        assert_eq!(
            split(b"first\nsecond\nthird\n", false),
            vec!["first", "second", "third"]
        );
    }

    #[test]
    fn captured_output_holds_a_partial_line_until_it_is_flushed() {
        // Mid-write: the rest of the line may still be coming.
        assert_eq!(split(b"done\npartial", false), vec!["done"]);
        // The writer went quiet, so show it rather than wait forever. This is
        // the `print!`-with-no-newline case, which otherwise never surfaces.
        assert_eq!(split(b"done\npartial", true), vec!["done", "partial"]);
    }

    #[test]
    fn captured_output_treats_carriage_returns_as_line_ends() {
        // llama.cpp-style progress counters overwrite one line with `\r`.
        assert_eq!(
            split(b"loading 10%\rloading 20%\r", false),
            vec!["loading 10%", "loading 20%"]
        );
    }

    #[test]
    fn captured_output_strips_control_characters() {
        // The whole point is to keep stray bytes off the frame. An escape
        // sequence rendered into a dashboard cell would move the cursor and
        // corrupt the frame this exists to protect.
        assert_eq!(
            split(b"\x1b[10;5HXXXX\n", false),
            vec!["[10;5HXXXX"],
            "escape bytes must not survive into a rendered cell"
        );
    }

    #[test]
    fn captured_output_keeps_tabs_as_spaces() {
        // llama.cpp separates its loader columns with tabs. Stripping them as
        // control characters would run the columns together.
        assert_eq!(
            split(b"llm_load_print_meta: n_ctx\t= 4096\n", false),
            vec!["llm_load_print_meta: n_ctx = 4096"]
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

    #[test]
    fn stale_reader_routes_child_output_to_reentered_dashboard() {
        let active_captures = AtomicUsize::new(1);
        let (mut read_fd, write_fd) = std::io::pipe().expect("pipe should open");
        let (child_ready_tx, child_ready_rx) = mpsc::channel();
        let (write_tx, write_rx) = mpsc::channel();
        let child = std::thread::spawn(move || {
            let mut retained_write_fd = write_fd;
            child_ready_tx
                .send(())
                .expect("parent should wait for retained pipe");
            write_rx.recv().expect("parent should release child writer");
            retained_write_fd
                .write_all(b"late child output\n")
                .expect("child output should write");
        });

        child_ready_rx
            .recv()
            .expect("child should retain the first pipe write end");
        active_captures.fetch_sub(1, Ordering::AcqRel);
        active_captures.fetch_add(1, Ordering::AcqRel);
        write_tx.send(()).expect("child should still be waiting");

        let mut captured = String::new();
        read_fd
            .read_to_string(&mut captured)
            .expect("reader should receive late child output");
        child.join().expect("child writer should exit");

        let mut emitted = Vec::new();
        let mut passthrough = Vec::new();
        deliver_with(
            captured.trim_end().to_string(),
            &active_captures,
            &mut passthrough,
            |event| {
                emitted.push(event);
                Ok(())
            },
        );

        assert_eq!(emitted.len(), 1, "late output belongs in the new dashboard");
        assert!(passthrough.is_empty(), "it must not bypass the dashboard");
    }

    #[test]
    fn stale_reader_routes_output_during_reader_registration() {
        let active_captures = Arc::new(AtomicUsize::new(0));
        let (deliver_tx, deliver_rx) = mpsc::channel();
        let (delivered_tx, delivered_rx) = mpsc::channel();
        let reader_activity = Arc::clone(&active_captures);
        let stale_reader = std::thread::spawn(move || {
            deliver_rx
                .recv()
                .expect("registration should release reader");
            let mut emitted = Vec::new();
            let mut passthrough = Vec::new();
            deliver_with(
                "concurrent stale output".to_string(),
                &reader_activity,
                &mut passthrough,
                |event| {
                    emitted.push(event);
                    Ok(())
                },
            );
            delivered_tx
                .send(())
                .expect("registration should await delivery");
            (emitted, passthrough)
        });

        register_capture_reader(&active_captures, || {
            deliver_tx.send(()).expect("stale reader should be waiting");
            delivered_rx
                .recv()
                .expect("stale reader should deliver during registration");
            Ok(())
        })
        .expect("reader registration should succeed");

        let (emitted, passthrough) = stale_reader.join().expect("stale reader should exit");
        assert_eq!(
            emitted.len(),
            1,
            "concurrent output belongs in the dashboard"
        );
        assert!(passthrough.is_empty(), "it must not bypass the dashboard");
    }

    #[test]
    fn failed_reader_registration_rolls_back_capture_activity() {
        let active_captures = AtomicUsize::new(0);

        let result = register_capture_reader(&active_captures, || {
            Err(std::io::Error::other("reader spawn failed"))
        });

        assert!(result.is_err());
        assert_eq!(active_captures.load(Ordering::Acquire), 0);
    }
}

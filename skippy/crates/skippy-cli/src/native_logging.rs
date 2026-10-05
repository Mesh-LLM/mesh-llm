//! Keep native diagnostics quiet unless debugging or a failure needs context.
use std::{collections::VecDeque, sync::Mutex};

use skippy_runtime::logging::NativeLogSink;

const MAX_DIAGNOSTIC_BYTES: usize = 128 * 1024;
// Values from ggml_log_level; CONT fragments inherit the previous severity.
const ERROR: i32 = 4;
const CONT: i32 = 5;

pub(crate) struct NativeDiagnostics {
    debug: bool,
    buffer: Mutex<DiagnosticBuffer>,
}

impl NativeDiagnostics {
    pub(crate) fn new(debug: bool) -> Self {
        Self {
            debug,
            buffer: Mutex::new(DiagnosticBuffer::default()),
        }
    }

    pub(crate) fn flush_on_error(&self) {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let text = buffer.drain();
        if !text.is_empty() {
            let _ = skippy_commands::console::native_log(&text);
        }
    }
}

impl NativeLogSink for NativeDiagnostics {
    fn write(&self, level: i32, text: &str) {
        // Serialize native callbacks and output to preserve fragment ordering.
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let text = buffer.record(self.debug, level, text);
        if !text.is_empty() {
            let _ = skippy_commands::console::native_log(&text);
        }
    }
}

#[derive(Default)]
struct DiagnosticBuffer {
    chunks: VecDeque<String>,
    bytes: usize,
    last_level: i32,
}

impl DiagnosticBuffer {
    fn record(&mut self, debug: bool, level: i32, text: &str) -> String {
        if level != CONT {
            self.last_level = level;
        }
        if debug {
            return text.to_owned();
        }
        self.push(text);
        if self.last_level == ERROR {
            self.drain()
        } else {
            String::new()
        }
    }

    fn push(&mut self, text: &str) {
        // Bound even a single huge message, retaining a valid UTF-8 tail.
        let mut start = text.len().saturating_sub(MAX_DIAGNOSTIC_BYTES);
        while !text.is_char_boundary(start) {
            start += 1;
        }
        let text = &text[start..];
        while self.bytes + text.len() > MAX_DIAGNOSTIC_BYTES {
            let Some(oldest) = self.chunks.pop_front() else {
                break;
            };
            self.bytes -= oldest.len();
        }
        if !text.is_empty() {
            self.chunks.push_back(text.to_owned());
            self.bytes += text.len();
        }
    }

    fn drain(&mut self) -> String {
        self.bytes = 0;
        self.chunks.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn debug_is_opt_in_and_global() {
        assert!(
            !crate::cli::Cli::try_parse_from(["skippy", "example-config"])
                .unwrap()
                .debug
        );
        for args in [
            vec!["skippy", "--debug", "example-config"],
            vec!["skippy", "example-config", "--debug"],
            vec!["skippy", "serve", "--model-path", "model.gguf", "--debug"],
        ] {
            assert!(crate::cli::Cli::try_parse_from(args).unwrap().debug);
        }
    }

    #[test]
    fn quiet_logs_are_replayed_on_native_error_with_continuations() {
        let mut buffer = DiagnosticBuffer::default();
        assert_eq!(buffer.record(false, 1, "loading\n"), "");
        assert_eq!(buffer.record(false, 2, "allocating\n"), "");
        assert_eq!(buffer.record(false, 3, "warning\n"), "");
        assert_eq!(
            buffer.record(false, ERROR, "failed: "),
            "loading\nallocating\nwarning\nfailed: "
        );
        assert_eq!(
            buffer.record(false, CONT, "out of memory\n"),
            "out of memory\n"
        );
        assert_eq!(buffer.record(false, 2, "next request\n"), "");
        assert_eq!(buffer.drain(), "next request\n");
        assert!(buffer.drain().is_empty());
    }

    #[test]
    fn debug_logs_stream_without_duplicate_error_replay() {
        let mut buffer = DiagnosticBuffer::default();
        for level in [0, 1, 2, 3, ERROR, CONT] {
            assert_eq!(
                buffer.record(true, level, "native fragment"),
                "native fragment"
            );
        }
        assert!(buffer.drain().is_empty());
    }

    #[test]
    fn diagnostic_history_is_bounded_and_handles_large_unicode_messages() {
        let mut buffer = DiagnosticBuffer::default();
        buffer.record(false, 2, "old message\n");
        let large = "🧠".repeat(MAX_DIAGNOSTIC_BYTES);
        buffer.record(false, 1, &large);
        assert!(buffer.bytes <= MAX_DIAGNOSTIC_BYTES);
        assert_eq!(buffer.chunks.len(), 1);
        let retained = buffer.drain();
        assert!(retained.ends_with("🧠"));
        assert!(!retained.contains("old message"));
    }
}

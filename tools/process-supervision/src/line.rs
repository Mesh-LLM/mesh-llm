use super::Stream;

/// How the owner completed a raw line. EOF is an actual zero-byte pipe read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    Eof,
}

/// Borrowed, pre-redaction content, excluding LF but retaining any preceding CR.
/// At most 8192 bytes, including CR. Oversized lines never reach a matcher.
/// Intentionally has no Debug implementation: raw diagnostics may contain secrets.
pub struct ObservedLine<'a> {
    pub stream: Stream,
    pub bytes: &'a [u8],
    pub ending: LineEnding,
}

/// Trusted repository code only: bounded in-memory classification, no I/O,
/// threads, blocking, logging, panics, or retained copies of the borrowed bytes.
/// Each concrete matcher needs bounded-work/privacy tests; this type alone
/// cannot enforce those requirements. No matcher is loaded from child/CLI input.
pub type LineMatcher = fn(ObservedLine<'_>) -> bool;

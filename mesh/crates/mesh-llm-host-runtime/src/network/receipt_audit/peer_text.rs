//! Text a peer chose, made safe for a log line and a block reason.
//!
//! A receipt's id and model, an error body and every reason an audit gives
//! hold whatever the peer sent. Written as they are, a newline starts a log
//! line of the peer's own, an escape sequence rewrites the operator's
//! terminal, and a megabyte of text fills the log. So everything this node
//! logs or keeps about a peer goes through [`shown`]: one line, control
//! characters written out, cut to [`SHOWN_CHARS`].

use std::fmt::Write;

/// The most characters of one peer-chosen text this node logs or keeps.
pub(super) const SHOWN_CHARS: usize = 240;

/// Whether a character moves the cursor, ends the line, or changes the
/// direction text is drawn in.
fn hidden(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{2028}' | '\u{2029}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
        )
}

/// `text` on one line: control characters, line separators and
/// bidirectional overrides are written as escapes, and a long text is cut,
/// so that what is shown is at most [`SHOWN_CHARS`] characters and a mark.
pub(super) fn shown(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(SHOWN_CHARS));
    let mut written = 0;
    let mut piece = String::new();
    for c in text.chars() {
        piece.clear();
        match c {
            '\n' => piece.push_str("\\n"),
            '\r' => piece.push_str("\\r"),
            '\t' => piece.push_str("\\t"),
            c if hidden(c) => {
                let _ = write!(piece, "{}", c.escape_unicode());
            }
            c => piece.push(c),
        }
        let chars = piece.chars().count();
        if written + chars > SHOWN_CHARS {
            out.push('…');
            break;
        }
        out.push_str(&piece);
        written += chars;
    }
    out
}

/// Text that is safe to log and to keep: whatever it was made from went
/// through [`shown`]. What this node says about a peer is built of these,
/// so no path from a peer's bytes to a log line skips the escaping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Shown(String);

impl Shown {
    pub(super) fn of(text: &str) -> Self {
        Self(shown(text))
    }

    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Shown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An error and its causes, shown. Errors of the audit quote what the peer
/// sent: a status line, a body, a name out of a receipt.
pub(super) fn shown_error(error: &anyhow::Error) -> String {
    shown(&format!("{error:#}"))
}

/// Bytes a peer sent as text, shown. Reads no more of them than it shows.
pub(super) fn shown_bytes(bytes: &[u8]) -> String {
    // a character is at most four bytes
    let head = &bytes[..bytes.len().min(4 * SHOWN_CHARS + 4)];
    let mut text = shown(&String::from_utf8_lossy(head));
    if head.len() < bytes.len() && !text.ends_with('…') {
        text.push('…');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_peer_cannot_write_a_log_line_of_its_own() {
        let forged = "r1\nWARN blocked peer abc\r\n\x1b[2K\x1b[1Aaudit passed";
        let shown = shown(forged);
        assert!(!shown.contains('\n') && !shown.contains('\r') && !shown.contains('\x1b'));
        assert_eq!(
            shown,
            "r1\\nWARN blocked peer abc\\r\\n\\u{1b}[2K\\u{1b}[1Aaudit passed"
        );
        for hidden in [
            '\u{2028}', '\u{202e}', '\u{2066}', '\u{0}', '\u{7f}', '\u{85}',
        ] {
            let text = format!("a{hidden}b");
            let shown = super::shown(&text);
            assert!(!shown.contains(hidden), "{hidden:?} in {shown:?}");
        }
        assert_eq!(super::shown("Qwen3-4B Q8_0 é 模型"), "Qwen3-4B Q8_0 é 模型");
    }

    #[test]
    fn long_text_is_cut_between_characters() {
        let long = "é".repeat(10_000);
        let shown = shown(&long);
        assert_eq!(shown.chars().count(), SHOWN_CHARS + 1);
        assert!(shown.ends_with('…'));
        let error = anyhow::anyhow!("{}", "x".repeat(5_000)).context("503\nSTRIKE");
        let shown = shown_error(&error);
        assert!(shown.starts_with("503\\nSTRIKE: xxx"), "{shown}");
        assert_eq!(shown.chars().count(), SHOWN_CHARS + 1);
        // escapes count toward the bound: a text of nothing else stays short
        let overrides = "\u{202e}".repeat(10_000);
        assert!(super::shown(&overrides).chars().count() <= SHOWN_CHARS + 1);
    }

    #[test]
    fn a_body_is_shown_from_its_first_bytes_only() {
        let body = vec![b'a'; 64 << 20];
        let shown = shown_bytes(&body);
        assert_eq!(shown.chars().count(), SHOWN_CHARS + 1);
        // bytes that are not text are shown, never passed on
        let shown = shown_bytes(b"gone\xff\x00\n");
        assert_eq!(shown, "gone\u{fffd}\\u{0}\\n");
        assert_eq!(shown_bytes(b"short"), "short");
    }
}

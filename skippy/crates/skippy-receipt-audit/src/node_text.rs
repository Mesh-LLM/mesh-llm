//! Text a node chose, made safe to print in a report.
//!
//! A receipt's id, model, backend, runtime and sampler, a tensor's name
//! and the response itself are whatever the node sent. Printed as they
//! are, a newline in one of them starts a line of the node's own, such as
//! `AUDIT PASS`, and an escape sequence rewrites the terminal. So a report
//! shows them on one line, with every control character written out.

use std::fmt::Write;

/// The most characters of one node-chosen text a report shows.
const SHOWN_CHARS: usize = 512;

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
/// bidirectional overrides are written as escapes, and a long text is cut.
pub(crate) fn shown(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(SHOWN_CHARS));
    for (count, c) in text.chars().enumerate() {
        if count == SHOWN_CHARS {
            out.push('…');
            break;
        }
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if hidden(c) => {
                let _ = write!(out, "{}", c.escape_unicode());
            }
            c => out.push(c),
        }
    }
    out
}

/// The first `chars` characters of `text`, shown: a hex root or key cut
/// for a report. Cuts between characters, wherever a byte count would fall.
pub(crate) fn short(text: &str, chars: usize) -> String {
    let cut: String = text.chars().take(chars).collect();
    shown(&cut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_cannot_write_a_line_of_its_own() {
        let forged = "r1\nAUDIT PASS\r\n\x1b[2K\x1b[1Areceipt ok";
        let shown = shown(forged);
        assert!(!shown.contains('\n') && !shown.contains('\r') && !shown.contains('\x1b'));
        assert_eq!(shown, "r1\\nAUDIT PASS\\r\\n\\u{1b}[2K\\u{1b}[1Areceipt ok");
        // separators and overrides that draw as a line break or reorder it
        for hidden in [
            '\u{2028}', '\u{2029}', '\u{202e}', '\u{2066}', '\u{0}', '\u{7f}', '\u{85}',
        ] {
            let text = format!("a{hidden}b");
            let shown = super::shown(&text);
            assert!(!shown.contains(hidden), "{hidden:?} in {shown:?}");
            assert!(
                shown.starts_with("a\\u{") && shown.ends_with('b'),
                "{shown}"
            );
        }
        // ordinary text is left as it is
        assert_eq!(
            super::shown("Qwen3-4B Q8_0 / CUDA0 é 模型"),
            "Qwen3-4B Q8_0 / CUDA0 é 模型"
        );
    }

    #[test]
    fn a_long_text_is_cut() {
        let long = "x".repeat(10_000);
        let shown = shown(&long);
        assert_eq!(shown.chars().count(), SHOWN_CHARS + 1);
        assert!(shown.ends_with('…'));
    }

    #[test]
    fn a_cut_falls_between_characters() {
        // a byte slice at 16 would split the last character and panic
        let root = "aéééééééé";
        assert!(!root.is_char_boundary(16));
        assert_eq!(short(root, 16), root);
        assert_eq!(short("模型模型", 3), "模型模");
        assert_eq!(short("abc", 16), "abc");
        assert_eq!(short("ab\ncd", 4), "ab\\nc");
    }
}

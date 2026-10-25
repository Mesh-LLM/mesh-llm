//! The writer the dashboard paints through.
//!
//! The dashboard used to render straight to `io::stderr()`, which is also
//! where `eprintln!`, `tracing`'s default writer, inherited child stderr, and
//! most noisy C libraries write. Sharing that descriptor is what made stray
//! output land *on top of* the dashboard instead of in it, and it is why no
//! amount of converting individual call sites could ever close the hole.
//!
//! Rendering to the controlling terminal instead gives the dashboard a channel
//! nothing else holds a descriptor for, which in turn frees fd 1 and fd 2 to be
//! captured (see [`super::console_capture`]) without fighting over the screen.
//! This is the same reason `less`, `fzf`, and `vim` open the tty directly
//! rather than trusting their standard descriptors.

#[cfg(not(windows))]
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};

#[cfg(not(windows))]
type Tty = File;
#[cfg(windows)]
type Tty = ConsoleOut;

/// Where terminal control sequences and dashboard frames are written.
///
/// Falls back to stderr when there is no controlling terminal to open — a
/// piped or redirected session never reaches the dashboard path anyway, but
/// the fallback keeps enter/exit sequences working for the fallback console.
pub(in crate::output) enum TerminalOut {
    Tty(BufWriter<Tty>),
    Stderr(BufWriter<io::Stderr>),
}

impl TerminalOut {
    pub(in crate::output) fn open() -> Self {
        match open_controlling_terminal() {
            Some(file) => Self::Tty(BufWriter::new(file)),
            None => Self::Stderr(BufWriter::new(io::stderr())),
        }
    }

    /// True when the dashboard owns a descriptor that is independent of fd 1
    /// and fd 2. Console capture is only safe to install in that case,
    /// otherwise redirecting stderr would redirect the dashboard itself.
    pub(in crate::output) fn is_private(&self) -> bool {
        matches!(self, Self::Tty(_))
    }
}

#[cfg(unix)]
fn open_controlling_terminal() -> Option<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .ok()
}

#[cfg(windows)]
fn open_controlling_terminal() -> Option<ConsoleOut> {
    // `CONOUT$` is the Windows analogue: it resolves to the active console
    // screen buffer regardless of how the standard handles were redirected.
    let handle =
        crossterm_winapi::Handle::new(crossterm_winapi::HandleType::CurrentOutputHandle).ok()?;
    Some(ConsoleOut {
        console: crossterm_winapi::Console::from(handle),
        pending: Vec::new(),
    })
}

#[cfg(not(any(unix, windows)))]
fn open_controlling_terminal() -> Option<File> {
    None
}

/// Writes UTF-8 to the console as UTF-16 through `WriteConsoleW`.
///
/// Plain byte writes to `CONOUT$` are decoded with the console's output code
/// page, which is an OEM page such as 437 or 850 unless the user ran
/// `chcp 65001`: every box-drawing character then lands as two or three wrong
/// cells and the dashboard turns to garbage. `std::io::stdout()` avoids this by
/// writing UTF-16 itself; this writer does the same for the dashboard's own
/// handle.
#[cfg(windows)]
pub(in crate::output) struct ConsoleOut {
    console: crossterm_winapi::Console,
    /// The start of a character split across two writes.
    pending: Vec<u8>,
}

#[cfg(windows)]
impl Write for ConsoleOut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut bytes = std::mem::take(&mut self.pending);
        let carried = bytes.len();
        bytes.extend_from_slice(buf);
        let complete = utf8_tail_start(&bytes);
        let text = String::from_utf8_lossy(&bytes[..complete]);
        if !text.is_empty()
            && let Err(error) = self.console.write_char_buffer(text.as_bytes())
        {
            bytes.truncate(carried);
            self.pending = bytes;
            return Err(error);
        }
        self.pending = bytes.split_off(complete);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Where a trailing UTF-8 sequence that is still missing bytes starts, or
/// `bytes.len()` when the buffer ends on a character boundary.
#[cfg(any(windows, test))]
fn utf8_tail_start(bytes: &[u8]) -> usize {
    for back in 1..=bytes.len().min(3) {
        let start = bytes.len() - back;
        let width = match bytes[start] {
            0x80..=0xBF => continue,
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => 1,
        };
        return if width > back { start } else { bytes.len() };
    }
    bytes.len()
}

impl Write for TerminalOut {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Tty(file) => file.write(buf),
            Self::Stderr(stderr) => stderr.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Tty(file) => file.flush(),
            Self::Stderr(stderr) => stderr.flush(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utf8_tail_start;

    #[test]
    fn a_buffer_that_ends_on_a_character_boundary_has_no_tail() {
        for text in ["", "abc", "│", "╭─ Mesh Events ─╮", "ok 😀"] {
            assert_eq!(utf8_tail_start(text.as_bytes()), text.len(), "{text}");
        }
    }

    #[test]
    fn a_character_cut_by_a_write_is_carried_whole_to_the_next_one() {
        let text = "╭▶ Mesh Events ↑/↓ · 😀 ╮";
        let bytes = text.as_bytes();
        for cut in 0..=bytes.len() {
            let mut carried = Vec::new();
            let mut written = String::new();
            for chunk in [&bytes[..cut], &bytes[cut..]] {
                carried.extend_from_slice(chunk);
                let complete = utf8_tail_start(&carried);
                written.push_str(&String::from_utf8_lossy(&carried[..complete]));
                carried.drain(..complete);
            }
            assert_eq!(written, text, "cut at byte {cut}");
            assert!(carried.is_empty(), "cut at byte {cut}");
        }
    }

    #[test]
    fn invalid_bytes_are_not_held_back() {
        // Stray continuation bytes and a lead byte followed by ASCII can never
        // complete, so they are written (as U+FFFD) instead of carried.
        assert_eq!(utf8_tail_start(b"a\x80\x80\x80\x80"), 5);
        assert_eq!(utf8_tail_start(b"a\xE2A"), 3);
        assert_eq!(utf8_tail_start(b"a\xFF"), 2);
        assert_eq!(utf8_tail_start(b"a\xE2\x94"), 1);
        assert_eq!(utf8_tail_start(b"\xF0\x9F\x98"), 0);
    }
}

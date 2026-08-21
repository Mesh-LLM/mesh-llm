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

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};

/// Where terminal control sequences and dashboard frames are written.
///
/// Falls back to stderr when there is no controlling terminal to open — a
/// piped or redirected session never reaches the dashboard path anyway, but
/// the fallback keeps enter/exit sequences working for the fallback console.
pub(in crate::output) enum TerminalOut {
    Tty(BufWriter<File>),
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
fn open_controlling_terminal() -> Option<File> {
    // `CONOUT$` is the Windows analogue: it resolves to the active console
    // screen buffer regardless of how the standard handles were redirected.
    OpenOptions::new()
        .read(true)
        .write(true)
        .open("CONOUT$")
        .ok()
}

#[cfg(not(any(unix, windows)))]
fn open_controlling_terminal() -> Option<File> {
    None
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

//! Preserve main's failed-chat body and native runtime log diagnostics before state cleanup.
use super::{Check, http::ResultBody};
use crate::automation::private_state::PrivateState;
use std::{
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};
const RETAINED: usize = 65536;
pub(super) fn response(check: Check, result: &ResultBody) {
    if check != Check::Chat {
        return;
    }
    if let ResultBody::Complete(status, body) = result {
        let mut output = std::io::stderr().lock();
        let _ = writeln!(
            output,
            "Non-stream chat completion failed (HTTP {status}): {}",
            String::from_utf8_lossy(&body[..body.len().min(RETAINED)])
        );
    }
}
pub(super) fn native_logs(state: &PrivateState) {
    let files = state.output_files();
    let Some(parent) = files.stdout.as_deref().and_then(Path::parent) else {
        return;
    };
    let mut pending = vec![(parent.join("runtime"), 0_u8)];
    let mut examined = 0_usize;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 16 || examined >= 4096 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries {
            examined += 1;
            if examined > 4096 {
                return;
            }
            let Ok(entry) = entry else {
                continue;
            };
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file() && entry.file_name() == "skippy-native.log" {
                print_tail(&entry.path());
            }
        }
    }
}
fn print_tail(path: &Path) {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let Ok(mut file) = options.open(path) else {
        return;
    };
    let Ok(metadata) = file.metadata() else {
        return;
    };
    if !metadata.is_file() {
        return;
    }
    let offset = metadata.len().saturating_sub(RETAINED as u64);
    if file.seek(SeekFrom::Start(offset)).is_err() {
        return;
    }
    let mut bytes = Vec::new();
    if file.take(RETAINED as u64).read_to_end(&mut bytes).is_err() {
        return;
    }
    let mut output = std::io::stderr().lock();
    let _ = writeln!(output, "--- native runtime log: {} ---", path.display());
    let _ = writeln!(output, "{}", last_lines(&bytes));
    let _ = writeln!(output, "--- end native runtime log ---");
}
fn last_lines(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut lines: Vec<_> = text.lines().rev().take(100).collect();
    lines.reverse();
    lines.join("\n")
}
#[cfg(test)]
mod tests {
    #[test]
    fn native_log_tail_retains_last_hundred_lines() {
        let source = (0..150).map(|i| format!("entry-{i}\n")).collect::<String>();
        let tail = super::last_lines(source.as_bytes());
        assert_eq!(tail.lines().count(), 100);
        assert_eq!(tail.lines().next(), Some("entry-50"));
        assert_eq!(tail.lines().last(), Some("entry-149"));
    }
}

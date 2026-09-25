//! Python's wording for the I/O failures the legacy consumers report:
//! `str(OSError)` and `str(UnicodeDecodeError)` for UTF-8 text reads.

use crate::repository::python_text;
use std::path::Path;

/// `str(OSError)` for a failed open or read: `[Errno N] text: 'path'`.
pub(crate) fn os_error(path: &Path, error: &std::io::Error) -> String {
    let Some(code) = error.raw_os_error() else {
        return error.to_string();
    };
    let rendered = error.to_string();
    let text = rendered
        .strip_suffix(&format!(" (os error {code})"))
        .unwrap_or(&rendered);
    let filename = python_text::repr(&path.to_string_lossy());
    format!("[Errno {code}] {text}: {filename}")
}

/// `path.read_text(encoding="utf-8")` with Python's error text.
pub(crate) fn read_text(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| os_error(path, &error))?;
    decode_utf8(bytes)
}

/// `bytes.decode("utf-8")` with `UnicodeDecodeError` wording.
pub(crate) fn decode_utf8(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|error| {
        let utf8 = error.utf8_error();
        let bytes = error.as_bytes();
        let start = utf8.valid_up_to();
        let lead = bytes.get(start).copied().unwrap_or_default();
        let prefix = "'utf-8' codec can't decode";
        match utf8.error_len() {
            None if bytes.len() - start > 1 => format!(
                "{prefix} bytes in position {start}-{}: unexpected end of data",
                bytes.len() - 1
            ),
            None => {
                format!("{prefix} byte 0x{lead:02x} in position {start}: unexpected end of data")
            }
            Some(_) => {
                let reason = if (0x80..=0xc1).contains(&lead) || lead >= 0xf5 {
                    "invalid start byte"
                } else {
                    "invalid continuation byte"
                };
                format!("{prefix} byte 0x{lead:02x} in position {start}: {reason}")
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_prepared_inputs_decode_errors_match_python() {
        let cases: [(&[u8], &str); 4] = [
            (b"ab\xff", "byte 0xff in position 2: invalid start byte"),
            (
                b"\xe2\x28\xa1",
                "byte 0xe2 in position 0: invalid continuation byte",
            ),
            (b"\xe2", "byte 0xe2 in position 0: unexpected end of data"),
            (
                b"\xf0\x9f\x98",
                "bytes in position 0-2: unexpected end of data",
            ),
        ];
        for (bytes, expected) in cases {
            let message = decode_utf8(bytes.to_vec()).expect_err("invalid UTF-8");
            assert_eq!(message, format!("'utf-8' codec can't decode {expected}"));
        }
    }

    #[test]
    fn migration_prepared_inputs_os_error_matches_python() {
        let path = Path::new("/nonexistent/it's");
        let error = std::fs::read(path).expect_err("missing file");
        assert_eq!(
            os_error(path, &error),
            "[Errno 2] No such file or directory: \"/nonexistent/it's\""
        );
    }
}

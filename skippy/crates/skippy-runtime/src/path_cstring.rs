use std::ffi::CString;
use std::path::Path;

use anyhow::{Context, Result};

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;

pub(crate) fn path_to_cstring(path: &Path, label: &str) -> Result<CString> {
    path_bytes(path, label).and_then(|bytes| {
        CString::new(bytes).with_context(|| format!("{label} contains an interior NUL byte"))
    })
}

#[cfg(unix)]
fn path_bytes(path: &Path, _label: &str) -> Result<Vec<u8>> {
    Ok(path.as_os_str().as_bytes().to_vec())
}

#[cfg(not(unix))]
fn path_bytes(path: &Path, label: &str) -> Result<Vec<u8>> {
    let path = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("{label} is not valid UTF-8"))?;
    Ok(path.as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::path_to_cstring;
    use std::path::PathBuf;

    #[cfg(any(unix, windows))]
    use std::ffi::OsString;
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;
    #[cfg(windows)]
    use std::os::windows::ffi::OsStringExt;

    #[test]
    #[cfg(unix)]
    fn path_to_cstring_preserves_unix_os_bytes() -> anyhow::Result<()> {
        let path = PathBuf::from(OsString::from_vec(vec![b'/', b't', b'm', b'p', b'/', 0xff]));

        let cstring = path_to_cstring(&path, "model path")?;

        assert_eq!(cstring.as_bytes(), b"/tmp/\xff");
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn path_to_cstring_rejects_unix_interior_nul() {
        let path = PathBuf::from(OsString::from_vec(vec![b'/', b't', b'm', b'p', b'/', 0]));

        let error = path_to_cstring(&path, "model path")
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("model path contains an interior NUL byte"),
            "unexpected error: {error}"
        );
    }

    #[test]
    #[cfg(not(unix))]
    fn path_to_cstring_accepts_utf8_paths_on_non_unix() -> anyhow::Result<()> {
        let path = PathBuf::from("C:/models/model.gguf");

        let cstring = path_to_cstring(&path, "model path")?;

        assert_eq!(cstring.as_bytes(), b"C:/models/model.gguf");
        Ok(())
    }

    #[test]
    #[cfg(windows)]
    fn path_to_cstring_rejects_windows_paths_that_are_not_utf8() {
        let path = PathBuf::from(OsString::from_wide(&[0xD800]));

        let error = path_to_cstring(&path, "model path")
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("model path is not valid UTF-8"),
            "unexpected error: {error}"
        );
    }
}

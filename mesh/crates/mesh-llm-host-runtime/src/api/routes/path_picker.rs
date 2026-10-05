use super::super::http::{respond_error, respond_json};
use serde::Serialize;
use std::process::Command;
use tokio::net::TcpStream;

#[derive(Serialize)]
struct DirectoryPickerResponse {
    path: Option<String>,
    cancelled: bool,
}

pub(super) async fn handle(stream: &mut TcpStream) -> anyhow::Result<()> {
    match tokio::task::spawn_blocking(pick_directory).await {
        Ok(Ok(Some(path))) => {
            respond_json(
                stream,
                200,
                &DirectoryPickerResponse {
                    path: Some(path),
                    cancelled: false,
                },
            )
            .await
        }
        Ok(Ok(None)) => {
            respond_json(
                stream,
                200,
                &DirectoryPickerResponse {
                    path: None,
                    cancelled: true,
                },
            )
            .await
        }
        Ok(Err(error)) => respond_error(stream, 503, &error).await,
        Err(_) => respond_error(stream, 500, "The directory picker stopped unexpectedly").await,
    }
}

#[cfg(target_os = "macos")]
fn pick_directory() -> Result<Option<String>, String> {
    picker_command(
        "osascript",
        &[
            "-e",
            "POSIX path of (choose folder with prompt \"Choose a MeshLLM log storage folder\")",
        ],
    )
}

#[cfg(target_os = "windows")]
fn pick_directory() -> Result<Option<String>, String> {
    picker_command(
        "powershell.exe",
        &[
            "-NoProfile",
            "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath } else { exit 1 }",
        ],
    )
}

#[cfg(target_os = "linux")]
fn pick_directory() -> Result<Option<String>, String> {
    picker_command(
        "zenity",
        &[
            "--file-selection",
            "--directory",
            "--title=Choose a MeshLLM log storage folder",
        ],
    )
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
fn pick_directory() -> Result<Option<String>, String> {
    Err(
        "A system directory picker is not available on this platform; enter the host path manually"
            .to_string(),
    )
}

fn picker_command(program: &str, args: &[&str]) -> Result<Option<String>, String> {
    let output = Command::new(program).args(args).output().map_err(|_| {
        "A system directory picker is not available on this host; enter the host path manually"
            .to_string()
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.trim().is_empty() || stderr.contains("User canceled") || stderr.contains("(-128)")
        {
            return Ok(None);
        }
        return Err(
            "The system directory picker could not open; enter the host path manually".to_string(),
        );
    }
    let path = trim_trailing_separators(String::from_utf8_lossy(&output.stdout).trim());
    Ok((!path.is_empty()).then_some(path))
}

/// Trim trailing path separators while preserving a root path.
///
/// Both `/` and `\` count as separators regardless of host platform so the
/// Windows drive-root case (`C:\`) stays unit-testable on any host. A root
/// (`/`, `C:\`) is preserved; an all-separator path collapses to `/`.
fn trim_trailing_separators(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let trimmed = path.trim_end_matches(['/', '\\']);
    if trimmed.is_empty() {
        return "/".to_string();
    }
    if path.ends_with(['/', '\\']) && is_drive_root(trimmed) {
        let separator = &path[path.len() - 1..];
        return format!("{trimmed}{separator}");
    }
    trimmed.to_string()
}

fn is_drive_root(path: &str) -> bool {
    let mut chars = path.chars();
    matches!(
        (chars.next(), chars.next(), chars.next()),
        (Some(letter), Some(':'), None) if letter.is_ascii_alphabetic()
    )
}

#[cfg(test)]
mod tests {
    use super::trim_trailing_separators;

    #[test]
    fn forward_slash_root_is_preserved() {
        assert_eq!(trim_trailing_separators("/"), "/");
    }

    #[test]
    fn windows_drive_root_is_preserved() {
        assert_eq!(trim_trailing_separators(r"C:\"), r"C:\");
    }

    #[test]
    fn all_separators_collapse_to_root() {
        assert_eq!(trim_trailing_separators("//"), "/");
    }

    #[test]
    fn trims_trailing_forward_slash() {
        assert_eq!(trim_trailing_separators("/foo/bar/"), "/foo/bar");
    }

    #[test]
    fn trims_trailing_windows_separators() {
        assert_eq!(trim_trailing_separators(r"C:\foo\bar\"), r"C:\foo\bar");
    }

    #[test]
    fn bare_drive_letter_is_unchanged() {
        assert_eq!(trim_trailing_separators("C:"), "C:");
    }

    #[test]
    fn trims_relative_trailing_slash() {
        assert_eq!(trim_trailing_separators("foo/"), "foo");
    }

    #[test]
    fn path_without_trailing_separator_is_unchanged() {
        assert_eq!(
            trim_trailing_separators("/home/user/logs"),
            "/home/user/logs"
        );
    }

    #[test]
    fn empty_path_is_unchanged() {
        assert_eq!(trim_trailing_separators(""), "");
    }
}

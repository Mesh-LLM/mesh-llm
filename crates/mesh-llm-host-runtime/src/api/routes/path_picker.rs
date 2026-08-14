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
    let path = String::from_utf8_lossy(&output.stdout)
        .trim()
        .trim_end_matches(std::path::MAIN_SEPARATOR)
        .to_string();
    Ok((!path.is_empty()).then_some(path))
}

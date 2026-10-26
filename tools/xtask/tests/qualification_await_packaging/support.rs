//! `release await-packaging` support: a fake `gh` executable serving canned
//! API pages, and a minimal store-method ZIP writer for the readiness
//! artifact.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub(crate) const CORRELATION: &str = "mesh-100-1-v1.2.3";
pub(crate) const UPSTREAM_SHA: &str = "0123456789abcdef0123456789abcdef01234567";
pub(crate) const MANIFEST_SHA: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// A store-method (uncompressed) ZIP containing exactly one file.
pub(crate) fn single_file_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut crc = flate2::Crc::new();
    crc.update(data);
    let crc = crc.sum();
    let mut zip = Vec::new();
    let name_bytes = name.as_bytes();
    // Local file header.
    zip.extend_from_slice(b"PK\x03\x04");
    zip.extend_from_slice(&20_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&crc.to_le_bytes());
    zip.extend_from_slice(&(data.len() as u32).to_le_bytes());
    zip.extend_from_slice(&(data.len() as u32).to_le_bytes());
    zip.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(name_bytes);
    zip.extend_from_slice(data);
    // Central directory.
    let central_offset = zip.len() as u32;
    zip.extend_from_slice(b"PK\x01\x02");
    zip.extend_from_slice(&20_u16.to_le_bytes());
    zip.extend_from_slice(&20_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip.extend_from_slice(&crc.to_le_bytes());
    zip.extend_from_slice(&(data.len() as u32).to_le_bytes());
    zip.extend_from_slice(&(data.len() as u32).to_le_bytes());
    zip.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
    zip.extend_from_slice(&[0; 12]);
    zip.extend_from_slice(&0_u32.to_le_bytes());
    zip.extend_from_slice(name_bytes);
    let central_size = zip.len() as u32 - central_offset;
    zip.extend_from_slice(b"PK\x05\x06");
    zip.extend_from_slice(&[0; 4]);
    zip.extend_from_slice(&1_u16.to_le_bytes());
    zip.extend_from_slice(&1_u16.to_le_bytes());
    zip.extend_from_slice(&central_size.to_le_bytes());
    zip.extend_from_slice(&central_offset.to_le_bytes());
    zip.extend_from_slice(&0_u16.to_le_bytes());
    zip
}

pub(crate) fn readiness_receipt() -> Value {
    json!({
        "schema": "mesh-packaging-readiness-v1",
        "correlation_id": CORRELATION,
        "packaging_run_id": 42,
        "packaging_run_attempt": 1,
        "upstream": {
            "repository": "Mesh-LLM/mesh-llm",
            "ref": "v1.2.3",
            "sha": UPSTREAM_SHA,
            "manifest_sha256": MANIFEST_SHA,
        },
        "requested": {
            "publish_images": "true",
            "publish_release_assets": "true",
            "publish_npm": "true",
        },
        "status": "success",
        "results": {"images": "pushed"},
    })
}

pub(crate) fn runs_page(titles: &[(&str, u64, u64, &str, &str)]) -> Value {
    json!({
        "workflow_runs": titles.iter().map(|(title, id, attempt, status, conclusion)| json!({
            "id": id,
            "run_attempt": attempt,
            "display_title": title,
            "status": status,
            "conclusion": conclusion,
        })).collect::<Vec<_>>(),
    })
}

pub(crate) fn artifacts_page() -> Value {
    json!({
        "artifacts": [{
            "id": 7,
            "name": "packaging-readiness",
            "expired": false,
            "size_in_bytes": 1024,
        }],
    })
}

/// Write a fake `gh` shell script that answers from the canned files.
#[cfg(unix)]
pub(crate) fn write_fake_gh(dir: &Path, runs: &Value, artifacts: &Value, zip: &[u8]) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::write(dir.join("runs.json"), json!(runs).to_string()).unwrap();
    std::fs::write(dir.join("artifacts.json"), json!(artifacts).to_string()).unwrap();
    std::fs::write(dir.join("payload.zip"), zip).unwrap();
    let script = dir.join("gh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\ncase \"$*\" in\n  *\"/zip\") cat \"{}\";;\n  *\"/artifacts?\"*) cat \"{}\";;\n  *\"/runs?\"*) cat \"{}\";;\n  *) echo \"unexpected gh call: $*\" >&2; exit 1;;\nesac\n",
            dir.join("payload.zip").display(),
            dir.join("artifacts.json").display(),
            dir.join("runs.json").display(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

/// Invoke `release await-packaging` with the fake `gh` substituted.
pub(crate) fn await_packaging(
    root: &Path,
    gh: Option<&Path>,
    extra: &[&str],
) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(root)
        .args([
            "release",
            "await-packaging",
            "--repository",
            "Mesh-LLM/mesh-packaging",
            "--correlation-id",
            CORRELATION,
            "--upstream-repository",
            "Mesh-LLM/mesh-llm",
            "--upstream-ref",
            "v1.2.3",
            "--upstream-sha",
            UPSTREAM_SHA,
            "--manifest-sha256",
            MANIFEST_SHA,
            "--out",
            root.join("readiness.json").to_str().unwrap(),
        ])
        .args(extra);
    if let Some(gh) = gh {
        command.env("MESH_LLM_TEST_GH", gh);
    }
    command.output().unwrap()
}

pub(crate) fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

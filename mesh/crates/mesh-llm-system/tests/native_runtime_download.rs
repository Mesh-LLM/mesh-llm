//! MeshLLM downloads a Skippy runtime from a release catalog and refuses the
//! ones it must not load. The catalog and archive are served from a local
//! HTTP server, so no Skippy release has to exist.

use mesh_llm_system::native_runtime_install::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest,
    current_runtime_release, current_skippy_abi_version, host_runtime_profile,
    install_native_runtime, mesh_native_runtime_install_options,
};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::Arc;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Serve fixed bodies by path; anything else is a 404.
fn serve(listener: TcpListener, files: HashMap<String, Vec<u8>>) {
    let files = Arc::new(files);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let files = files.clone();
            std::thread::spawn(move || {
                let mut request = Vec::new();
                let mut chunk = [0; 1024];
                while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                    match stream.read(&mut chunk) {
                        Ok(read) if read > 0 => request.extend_from_slice(&chunk[..read]),
                        _ => return,
                    }
                }
                let request = String::from_utf8_lossy(&request);
                let path = request.split_whitespace().nth(1).unwrap_or("/");
                let (status, body) = match files.get(path) {
                    Some(body) => ("200 OK", body.as_slice()),
                    None => ("404 Not Found", &b""[..]),
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body);
            });
        }
    });
}

/// A runtime this host and this build accept, before any test changes it.
fn compatible_artifact() -> NativeRuntimeArtifact {
    let profile = host_runtime_profile();
    let min_glibc = (profile.os == "linux").then(|| "2.17".to_string());
    NativeRuntimeArtifact {
        id: "fixture-runtime".to_string(),
        release_version: Some(current_runtime_release().to_string()),
        skippy_abi: current_skippy_abi_version(),
        platform: NativeRuntimePlatform {
            os: profile.os,
            arch: profile.arch,
            target: profile.target_triple,
            min_glibc,
        },
        backend: NativeRuntimeBackend::cpu(),
        rank: 0,
        libraries: vec!["lib/runtime.bin".to_string()],
        files: Default::default(),
        tools: Default::default(),
        url: None,
        sha256: None,
        signature: None,
    }
}

/// Package `artifact` as the release would: a gzip tar holding one runtime directory.
fn archive(work: &Path, artifact: &NativeRuntimeArtifact) -> Vec<u8> {
    let dir = work.join("package").join(&artifact.id);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::write(dir.join("lib/runtime.bin"), b"fixture runtime").unwrap();
    NativeRuntimeManifest {
        runtime: artifact.clone(),
    }
    .write_to_dir(&dir)
    .unwrap();
    let archive = work.join("runtime.tar.gz");
    let status = std::process::Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(work.join("package"))
        .arg(&artifact.id)
        .status()
        .expect("run tar");
    assert!(status.success());
    std::fs::read(archive).unwrap()
}

struct Release {
    artifact: NativeRuntimeArtifact,
    /// Checksum the catalog claims for the archive; `None` is the real one.
    claimed_archive_sha256: Option<String>,
    /// Checksum published beside the catalog; `None` is the real one.
    published_catalog_sha256: Option<String>,
}

impl Release {
    fn of(artifact: NativeRuntimeArtifact) -> Self {
        Self {
            artifact,
            claimed_archive_sha256: None,
            published_catalog_sha256: None,
        }
    }
}

/// Publish `release` on a local server and install from it into an empty cache.
fn install(release: Release) -> anyhow::Result<String> {
    let work = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());

    let archive = archive(work.path(), &release.artifact);
    let mut entry = release.artifact;
    entry.url = Some(format!("{base}/runtime.tar.gz"));
    entry.sha256 = Some(
        release
            .claimed_archive_sha256
            .unwrap_or_else(|| sha256(&archive)),
    );
    let catalog = serde_json::to_vec(&NativeRuntimeReleaseManifest {
        release_version: current_runtime_release().to_string(),
        skippy_abi: current_skippy_abi_version(),
        artifacts: vec![entry],
    })
    .unwrap();
    let catalog_sha256 = release
        .published_catalog_sha256
        .unwrap_or_else(|| sha256(&catalog));
    serve(
        listener,
        HashMap::from([
            ("/runtime.tar.gz".to_string(), archive),
            ("/native-runtimes.json".to_string(), catalog),
            (
                "/native-runtimes.json.sha256".to_string(),
                catalog_sha256.into_bytes(),
            ),
        ]),
    );

    let mut options = mesh_native_runtime_install_options();
    options.manifest_url = Some(format!("{base}/native-runtimes.json"));
    options.cache_dir = Some(work.path().join("cache"));
    let outcome = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(install_native_runtime(options))?;
    assert!(outcome.runtime.path.starts_with(work.path().join("cache")));
    assert_eq!(
        std::fs::read(outcome.runtime.path.join("lib/runtime.bin")).unwrap(),
        b"fixture runtime"
    );
    Ok(outcome.runtime.native_runtime_id)
}

fn refusal(release: Release) -> String {
    format!(
        "{:#}",
        install(release).expect_err("the runtime must be refused")
    )
}

#[test]
fn a_compatible_runtime_is_downloaded_verified_and_installed() {
    let id = install(Release::of(compatible_artifact())).unwrap();
    assert_eq!(id, "fixture-runtime");
}

#[test]
fn an_archive_that_does_not_match_its_checksum_is_refused() {
    let mut release = Release::of(compatible_artifact());
    release.claimed_archive_sha256 = Some("0".repeat(64));
    assert!(refusal(release).contains("native runtime checksum mismatch"));
}

#[test]
fn a_catalog_that_does_not_match_its_published_checksum_is_refused() {
    let mut release = Release::of(compatible_artifact());
    release.published_catalog_sha256 = Some("0".repeat(64));
    assert!(refusal(release).contains("native runtime manifest checksum mismatch"));
}

#[test]
fn a_runtime_built_for_another_skippy_abi_is_refused() {
    let mut artifact = compatible_artifact();
    artifact.skippy_abi = "0.0.0".to_string();
    assert!(refusal(Release::of(artifact)).contains("has skippy_abi 0.0.0, expected"));
}

#[test]
fn a_runtime_built_for_another_platform_is_refused() {
    let mut artifact = compatible_artifact();
    artifact.platform.os = "plan9".to_string();
    artifact.platform.target = Some("riscv64-unknown-plan9".to_string());
    assert!(refusal(Release::of(artifact)).contains("no compatible native runtime found"));
}

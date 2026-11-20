//! Upgrading MeshLLM on a machine whose runtime cache was filled by an
//! earlier release (issue #2727). The runtime release is the Skippy version,
//! so the cached runtime is never reused: setup downloads the new one, and
//! when it cannot, it fails with the old cache untouched.
//!
//! These run setup's own install-then-prune step against a catalog file and
//! a local HTTP server. No Skippy release has to exist and nothing reaches
//! the network. The "ROCm" files are a fixture payload on this host's
//! platform: the test shows the downloaded files are the ones selected, not
//! that a Windows ROCm package is built correctly.

use mesh_llm_commands::runtime_native::{
    NativeRuntimeConfigSelection, SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome,
    SetupNativeRuntimePruneResult, SetupNativeRuntimeStatus,
    install_and_prune_native_runtime_for_setup,
};
use mesh_llm_system::native_runtime_install::{
    NativeRuntimeArtifact, NativeRuntimeManifest, NativeRuntimeReleaseManifest,
    current_runtime_release, current_skippy_abi_version, host_runtime_profile,
};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{NativeRuntimeBackend, NativeRuntimeCache, NativeRuntimePlatform};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

/// Runtime release every MeshLLM up to 0.80.0 cached its runtime under.
const EARLIER_RELEASE: &str = "0.76.1";
const OLDER_RELEASE: &str = "0.70.0";
const KERNEL_FILES: [&str; 2] = [
    "bin/rocblas/library/kernels.dat",
    "bin/hipblaslt/library/kernels.dat",
];

/// Settings a developer or a CI job may have exported that would hand setup
/// a bundled runtime, which it would use instead of downloading.
const BUNDLE_OVERRIDES: [&str; 2] = [
    "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
    "SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR",
];

/// When a bundle override is set, run `test` again in a child process without
/// it and return `true`; the caller then returns.
fn reran_without_bundle_overrides(test: &str) -> bool {
    if BUNDLE_OVERRIDES
        .iter()
        .all(|name| std::env::var_os(name).is_none())
    {
        return false;
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    child.args(["--exact", test, "--nocapture"]);
    for name in BUNDLE_OVERRIDES {
        child.env_remove(name);
    }
    let output = child.output().expect("run the test in a child process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "{test} failed without bundle overrides:\n{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

macro_rules! without_bundle_overrides {
    ($test:ident) => {
        if reran_without_bundle_overrides(stringify!($test)) {
            return;
        }
    };
}

fn artifact(release: &str) -> NativeRuntimeArtifact {
    let profile = host_runtime_profile();
    let min_glibc = (profile.os == "linux").then(|| "2.17".to_string());
    NativeRuntimeArtifact {
        id: "fixture-runtime".to_string(),
        release_version: Some(release.to_string()),
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

/// Write an unpacked runtime under `root`, with `extra` files beside the
/// library. Every file holds the runtime's release, to tell copies apart.
fn runtime_dir(root: &Path, artifact: &NativeRuntimeArtifact, extra: &[&str]) -> PathBuf {
    let dir = root.join(&artifact.id);
    let release = artifact.release_version.as_deref().unwrap();
    for file in ["lib/runtime.bin"].iter().chain(extra) {
        let path = dir.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, release).unwrap();
    }
    NativeRuntimeManifest {
        runtime: artifact.clone(),
    }
    .write_to_dir(&dir)
    .unwrap();
    dir
}

/// Answer every request with `body`, or with a 404 when there is none.
fn serve(listener: TcpListener, body: Option<Vec<u8>>) {
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = Vec::new();
            let mut chunk = [0; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                match stream.read(&mut chunk) {
                    Ok(read) if read > 0 => request.extend_from_slice(&chunk[..read]),
                    _ => break,
                }
            }
            let (status, body) = match &body {
                Some(body) => ("200 OK", body.as_slice()),
                None => ("404 Not Found", &b""[..]),
            };
            let head = format!(
                "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
        }
    });
}

enum Download {
    Works,
    NotFound,
    WrongChecksum,
}

/// A machine that ran two earlier MeshLLM releases, and the release it is
/// upgrading to.
struct Upgrade {
    work: tempfile::TempDir,
    catalog: PathBuf,
}

impl Upgrade {
    fn new(download: Download) -> Self {
        let work = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(work.path().join("cache"));
        for release in [OLDER_RELEASE, EARLIER_RELEASE] {
            let dir = runtime_dir(&work.path().join(release), &artifact(release), &[]);
            cache.install_from_dir(&dir).unwrap();
        }

        let new = artifact(current_runtime_release());
        runtime_dir(&work.path().join("package"), &new, &KERNEL_FILES);
        let archive_path = work.path().join("runtime.tar.gz");
        let status = std::process::Command::new("tar")
            .arg("-czf")
            .arg(&archive_path)
            .arg("-C")
            .arg(work.path().join("package"))
            .arg(&new.id)
            .status()
            .expect("run tar");
        assert!(status.success());
        let archive = std::fs::read(archive_path).unwrap();
        let sha256: String = Sha256::digest(&archive)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut entry = new;
        entry.url = Some(format!(
            "http://{}/runtime.tar.gz",
            listener.local_addr().unwrap()
        ));
        entry.sha256 = Some(match download {
            Download::WrongChecksum => "0".repeat(64),
            _ => sha256,
        });
        serve(
            listener,
            (!matches!(download, Download::NotFound)).then_some(archive),
        );
        let catalog = work.path().join("native-runtimes.json");
        let manifest = NativeRuntimeReleaseManifest {
            release_version: current_runtime_release().to_string(),
            skippy_abi: current_skippy_abi_version(),
            artifacts: vec![entry],
        };
        std::fs::write(&catalog, serde_json::to_vec(&manifest).unwrap()).unwrap();
        Self { work, catalog }
    }

    fn cache(&self) -> PathBuf {
        self.work.path().join("cache")
    }

    /// The runtime step of `mesh-llm setup`.
    fn setup(&self) -> anyhow::Result<SetupNativeRuntimeOutcome> {
        let cache = self.cache();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(install_and_prune_native_runtime_for_setup(
                SetupNativeRuntimeOptions {
                    skip_runtime: false,
                    requested_runtime: None,
                    manifest_path: Some(&self.catalog),
                    bundle_dirs: &[],
                    cache_dir: Some(&cache),
                    configured: NativeRuntimeConfigSelection {
                        mesh_version: None,
                        skippy_abi_version: None,
                        selection: None,
                    },
                    progress: None,
                },
            ))
    }

    fn cached_releases(&self) -> Vec<String> {
        let mut releases: Vec<String> = std::fs::read_dir(self.cache())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        releases.sort();
        releases
    }

    /// A failed upgrade leaves both earlier runtimes as they were and adds nothing.
    fn assert_cache_untouched(&self) {
        assert_eq!(self.cached_releases(), [OLDER_RELEASE, EARLIER_RELEASE]);
        for release in [OLDER_RELEASE, EARLIER_RELEASE] {
            let library = self
                .cache()
                .join(release)
                .join("fixture-runtime/lib/runtime.bin");
            assert_eq!(std::fs::read_to_string(library).unwrap(), release);
        }
    }
}

#[test]
fn upgrade_downloads_the_new_runtime_instead_of_reusing_the_cached_one() {
    without_bundle_overrides!(upgrade_downloads_the_new_runtime_instead_of_reusing_the_cached_one);
    let upgrade = Upgrade::new(Download::Works);
    let outcome = upgrade.setup().unwrap();

    let SetupNativeRuntimeStatus::Installed(installed) = outcome.status else {
        panic!("setup skipped the runtime");
    };
    let selected = installed.runtime.path.canonicalize().unwrap();
    assert_eq!(
        selected,
        upgrade
            .cache()
            .join(current_runtime_release())
            .join("fixture-runtime")
            .canonicalize()
            .unwrap(),
        "the runtime cached by {EARLIER_RELEASE} must not be selected"
    );
    for file in ["lib/runtime.bin"].iter().chain(&KERNEL_FILES) {
        assert_eq!(
            std::fs::read_to_string(selected.join(file)).unwrap(),
            current_runtime_release(),
            "{file} is not the downloaded one"
        );
    }

    // Setup keeps the previous release's runtimes and removes older ones.
    let SetupNativeRuntimePruneResult::Pruned(plan) = outcome.prune else {
        panic!("setup did not prune: {:?}", outcome.prune);
    };
    assert_eq!(plan.remove_dirs, [upgrade.cache().join(OLDER_RELEASE)]);
    assert_eq!(
        upgrade.cached_releases(),
        [EARLIER_RELEASE, current_runtime_release()]
    );
}

#[test]
fn upgrade_whose_download_fails_reports_it_and_leaves_the_cache_alone() {
    without_bundle_overrides!(upgrade_whose_download_fails_reports_it_and_leaves_the_cache_alone);
    let upgrade = Upgrade::new(Download::NotFound);
    let error = format!("{:#}", upgrade.setup().unwrap_err());
    assert!(error.contains("404"), "{error}");
    upgrade.assert_cache_untouched();
}

#[test]
fn upgrade_whose_download_fails_its_checksum_leaves_the_cache_alone() {
    without_bundle_overrides!(upgrade_whose_download_fails_its_checksum_leaves_the_cache_alone);
    let upgrade = Upgrade::new(Download::WrongChecksum);
    let error = format!("{:#}", upgrade.setup().unwrap_err());
    assert!(
        error.contains("native runtime checksum mismatch"),
        "{error}"
    );
    upgrade.assert_cache_untouched();
}

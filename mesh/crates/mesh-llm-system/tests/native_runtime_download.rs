//! MeshLLM downloads a Skippy runtime from a release catalog, refuses the
//! ones it must not load, and chooses between a download and the CPU runtime
//! bundled in its own archive. The catalog and archive are served from a
//! local HTTP server, so no Skippy release has to exist and nothing here
//! reaches the network.

use mesh_llm_system::native_runtime_install::{
    NativeRuntimeArtifact, NativeRuntimeInstallOptions, NativeRuntimeInstallOutcome,
    NativeRuntimeManifest, NativeRuntimeReleaseManifest, current_runtime_release,
    current_skippy_abi_version, host_runtime_profile, install_native_runtime,
    load_release_manifest_with_sources, mesh_native_runtime_install_options,
    mesh_native_runtime_manifest_options,
};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{
    HostCudaProfile, HostRuntimeProfile, NativeRuntimeBackend, NativeRuntimeBackendKind,
    NativeRuntimeCache, NativeRuntimePlatform, NativeRuntimeResolver, NativeRuntimeSource,
    RuntimeSelection,
};
use std::collections::{BTreeSet, HashMap};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Settings a developer or a CI job may have exported that would hand these
/// tests another bundle, cache or catalog.
const RUNTIME_OVERRIDES: [&str; 6] = [
    "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR",
    "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR",
    "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL",
    "SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR",
    "SKIPPY_NATIVE_RUNTIME_CACHE_DIR",
    "SKIPPY_NATIVE_RUNTIME_MANIFEST_URL",
];

/// When any of [`RUNTIME_OVERRIDES`] is set, run `test` again in a child
/// process without them and return `true`; the caller then returns. The
/// environment of this process is left alone, since tests share it.
fn reran_without_runtime_overrides(test: &str) -> bool {
    if RUNTIME_OVERRIDES
        .iter()
        .all(|name| std::env::var_os(name).is_none())
    {
        return false;
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    child.args(["--exact", test, "--nocapture"]);
    for name in RUNTIME_OVERRIDES {
        child.env_remove(name);
    }
    let output = child.output().expect("run the test in a child process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{test} failed without runtime overrides:\n{stdout}{stderr}"
    );
    // A name that matches no test would pass with nothing run.
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "{test} did not run in the child:\n{stdout}"
    );
    true
}

/// First line of every test here.
macro_rules! without_runtime_overrides {
    ($test:ident) => {
        if reran_without_runtime_overrides(stringify!($test)) {
            return;
        }
    };
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Serve fixed bodies by path; anything else is a 404. Returns the paths asked for.
fn serve(listener: TcpListener, files: HashMap<String, Vec<u8>>) -> Arc<Mutex<Vec<String>>> {
    let files = Arc::new(files);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let files = files.clone();
            let log = log.clone();
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
                log.lock().unwrap().push(path.to_string());
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
    requests
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

/// Write `artifact` as an unpacked runtime directory under `root`.
fn runtime_dir(root: &Path, artifact: &NativeRuntimeArtifact) -> PathBuf {
    let dir = root.join(&artifact.id);
    std::fs::create_dir_all(dir.join("lib")).unwrap();
    std::fs::write(dir.join("lib/runtime.bin"), b"fixture runtime").unwrap();
    NativeRuntimeManifest {
        runtime: artifact.clone(),
    }
    .write_to_dir(&dir)
    .unwrap();
    dir
}

/// Package `artifact` as the release would: a gzip tar holding one runtime directory.
fn archive(work: &Path, artifact: &NativeRuntimeArtifact) -> Vec<u8> {
    runtime_dir(&work.join("package"), artifact);
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

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

struct Release {
    artifact: NativeRuntimeArtifact,
    /// Runtime actually inside the archive; `None` is the catalog's own.
    packaged: Option<NativeRuntimeArtifact>,
    /// Checksum the catalog claims for the archive; `None` is the real one.
    claimed_archive_sha256: Option<String>,
    /// Checksum published beside the catalog; `None` is the real one.
    published_catalog_sha256: Option<String>,
}

impl Release {
    fn of(artifact: NativeRuntimeArtifact) -> Self {
        Self {
            artifact,
            packaged: None,
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

    let archive = archive(
        work.path(),
        release.packaged.as_ref().unwrap_or(&release.artifact),
    );
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
    let outcome = block_on(install_native_runtime(options));
    if outcome.is_err() {
        let cache = work.path().join("cache");
        let published = std::fs::read_dir(&cache).map_or(0, |entries| entries.count());
        assert_eq!(published, 0, "a refused runtime was left in the cache");
    }
    let outcome = outcome?;
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
    without_runtime_overrides!(a_compatible_runtime_is_downloaded_verified_and_installed);
    let id = install(Release::of(compatible_artifact())).unwrap();
    assert_eq!(id, "fixture-runtime");
}

#[test]
fn an_archive_that_does_not_match_its_checksum_is_refused() {
    without_runtime_overrides!(an_archive_that_does_not_match_its_checksum_is_refused);
    let mut release = Release::of(compatible_artifact());
    release.claimed_archive_sha256 = Some("0".repeat(64));
    assert!(refusal(release).contains("native runtime checksum mismatch"));
}

#[test]
fn a_catalog_that_does_not_match_its_published_checksum_is_refused() {
    without_runtime_overrides!(a_catalog_that_does_not_match_its_published_checksum_is_refused);
    let mut release = Release::of(compatible_artifact());
    release.published_catalog_sha256 = Some("0".repeat(64));
    assert!(refusal(release).contains("native runtime manifest checksum mismatch"));
}

#[test]
fn a_runtime_built_for_another_skippy_abi_is_refused() {
    without_runtime_overrides!(a_runtime_built_for_another_skippy_abi_is_refused);
    let mut artifact = compatible_artifact();
    artifact.skippy_abi = "0.0.0".to_string();
    assert!(refusal(Release::of(artifact)).contains("has skippy_abi 0.0.0, expected"));
}

#[test]
fn a_runtime_built_for_another_platform_is_refused() {
    without_runtime_overrides!(a_runtime_built_for_another_platform_is_refused);
    let mut artifact = compatible_artifact();
    artifact.platform.os = "plan9".to_string();
    artifact.platform.target = Some("riscv64-unknown-plan9".to_string());
    assert!(refusal(Release::of(artifact)).contains("no compatible native runtime found"));
}

// The archive's checksum is right, so only a comparison of the selected
// catalog entry with the manifest inside the archive can catch these.
// Skippy 0558e33 installs all five: its installer hands the extracted
// directory to the cache without the selected entry. Remove the `ignore`
// when the pinned Skippy compares them.
#[test]
#[ignore = "Skippy's installer does not compare the archive's manifest with the catalog entry"]
fn an_archive_holding_a_different_runtime_than_its_catalog_entry_is_refused() {
    without_runtime_overrides!(
        an_archive_holding_a_different_runtime_than_its_catalog_entry_is_refused
    );
    type Change = fn(&mut NativeRuntimeArtifact);
    let changes: [(&str, Change); 5] = [
        ("id", |runtime| runtime.id = "another-runtime".to_string()),
        ("release", |runtime| {
            runtime.release_version = Some("0.0.1".to_string())
        }),
        ("skippy abi", |runtime| {
            runtime.skippy_abi = "0.0.0".to_string()
        }),
        ("platform", |runtime| {
            runtime.platform.os = "plan9".to_string()
        }),
        ("backend", |runtime| {
            runtime.backend = NativeRuntimeBackend::metal()
        }),
    ];
    let mut installed = Vec::new();
    for (field, change) in changes {
        let mut release = Release::of(compatible_artifact());
        let mut packaged = compatible_artifact();
        change(&mut packaged);
        release.packaged = Some(packaged);
        if install(release).is_ok() {
            installed.push(field);
        }
    }
    assert!(
        installed.is_empty(),
        "installed an archive whose manifest differs from the catalog entry in: {installed:?}"
    );
}

// Every MeshLLM archive bundles a CPU runtime (Metal on macOS). The tests
// below are about what a machine with something better available ends up
// loading. They pin the current policy; they do not choose it.
//
// `real_host_*` tests run MeshLLM's installer on this machine, download
// included. `synthetic_cuda_*` tests describe an NVIDIA machine that is not
// there, and stop at the choice. Neither downloads a CUDA runtime or runs
// inference.

/// A MeshLLM install: the CPU runtime from its archive on disk, an empty
/// cache, and a local server standing in for the Skippy release.
struct Machine {
    work: tempfile::TempDir,
    base: String,
    bundle: PathBuf,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Machine {
    /// `offered` is what the release catalog lists; `None` is a release host
    /// that answers every request with HTTP 404. A refused connection or a
    /// DNS failure is not simulated.
    fn new(bundled: &NativeRuntimeArtifact, offered: Option<&NativeRuntimeArtifact>) -> Self {
        let work = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let bundle = runtime_dir(&work.path().join("bundle"), bundled)
            .canonicalize()
            .unwrap();
        let mut files = HashMap::new();
        if let Some(offered) = offered {
            let archive = archive(work.path(), offered);
            let mut entry = offered.clone();
            entry.url = Some(format!("{base}/runtime.tar.gz"));
            entry.sha256 = Some(sha256(&archive));
            let catalog = serde_json::to_vec(&NativeRuntimeReleaseManifest {
                release_version: current_runtime_release().to_string(),
                skippy_abi: current_skippy_abi_version(),
                artifacts: vec![entry],
            })
            .unwrap();
            // The path MeshLLM asks its release host for by default.
            let catalog_path = mesh_native_runtime_install_options()
                .catalog
                .manifest_url(current_runtime_release())
                .split_once("/releases")
                .unwrap()
                .1
                .to_string();
            files.insert(
                format!("{catalog_path}.sha256"),
                sha256(&catalog).into_bytes(),
            );
            files.insert(catalog_path, catalog);
            files.insert("/runtime.tar.gz".to_string(), archive);
        }
        let requests = serve(listener, files);
        Self {
            work,
            base,
            bundle,
            requests,
        }
    }

    /// MeshLLM's own install options, with the release host swapped for the
    /// local server. Nothing names the catalog: it is found the default way.
    fn options(&self) -> NativeRuntimeInstallOptions {
        let mut options = mesh_native_runtime_install_options();
        options.catalog.releases_url = self.base.clone();
        options.bundle_dirs = vec![self.bundle.clone()];
        options.cache_dir = Some(self.work.path().join("cache"));
        options
    }

    fn install(
        &self,
        options: NativeRuntimeInstallOptions,
    ) -> anyhow::Result<NativeRuntimeInstallOutcome> {
        block_on(install_native_runtime(options))
    }

    fn cache(&self) -> NativeRuntimeCache {
        NativeRuntimeCache::new(self.work.path().join("cache"))
    }

    /// Put `artifact` in the cache, as an earlier `runtime install` would have.
    fn already_installed(&self, artifact: &NativeRuntimeArtifact) {
        let dir = runtime_dir(&self.work.path().join("earlier"), artifact);
        self.cache().install_from_dir(&dir).unwrap();
    }
}

fn bundled_cpu() -> NativeRuntimeArtifact {
    let mut artifact = compatible_artifact();
    artifact.id = "bundled-cpu".to_string();
    artifact
}

/// A runtime this host ranks above the bundled CPU one. Where the host
/// reports Metal that is a Metal runtime, a real backend difference. Anywhere
/// else (Linux, Windows, a macOS VM without Metal) a GPU runtime would be
/// refused, so a CPU runtime with a higher catalog rank stands in, and the
/// test then shows only that the higher rank wins.
fn better_than_bundled() -> NativeRuntimeArtifact {
    let mut artifact = compatible_artifact();
    artifact.id = "better-download".to_string();
    if host_runtime_profile().supports_flavor(&NativeRuntimeBackendKind::Metal) {
        artifact.backend = NativeRuntimeBackend::metal();
    } else {
        artifact.rank = 500;
    }
    artifact
}

#[test]
fn real_host_downloads_a_better_runtime_over_the_bundled_cpu_runtime() {
    without_runtime_overrides!(real_host_downloads_a_better_runtime_over_the_bundled_cpu_runtime);
    let machine = Machine::new(&bundled_cpu(), Some(&better_than_bundled()));
    let outcome = machine.install(machine.options()).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "better-download");
    assert!(
        outcome
            .runtime
            .path
            .starts_with(machine.work.path().join("cache"))
    );
}

#[test]
fn real_host_keeps_the_bundled_cpu_runtime_over_an_equally_ranked_download() {
    without_runtime_overrides!(
        real_host_keeps_the_bundled_cpu_runtime_over_an_equally_ranked_download
    );
    let mut same_rank = compatible_artifact();
    same_rank.id = "another-cpu".to_string();
    let machine = Machine::new(&bundled_cpu(), Some(&same_rank));
    let outcome = machine.install(machine.options()).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "bundled-cpu");
    assert_eq!(outcome.runtime.path, machine.bundle);
}

#[test]
fn real_host_with_only_the_bundle_uses_it_when_the_catalog_is_a_404() {
    without_runtime_overrides!(real_host_with_only_the_bundle_uses_it_when_the_catalog_is_a_404);
    let machine = Machine::new(&bundled_cpu(), None);
    let outcome = machine.install(machine.options()).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "bundled-cpu");
    assert_eq!(outcome.runtime.path, machine.bundle);
    assert!(
        outcome.sources.remote_error.is_some(),
        "the fallback to the bundle must record the catalog failure"
    );
    assert!(!machine.requests.lock().unwrap().is_empty());
}

#[test]
fn real_host_with_only_the_bundle_uses_it_when_downloads_are_off() {
    without_runtime_overrides!(real_host_with_only_the_bundle_uses_it_when_downloads_are_off);
    let machine = Machine::new(&bundled_cpu(), Some(&better_than_bundled()));
    let mut options = machine.options();
    options.allow_download = false;
    let outcome = machine.install(options).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "bundled-cpu");
    assert_eq!(outcome.runtime.path, machine.bundle);
    assert_eq!(outcome.sources.manifest_url, None);
    assert_eq!(*machine.requests.lock().unwrap(), Vec::<String>::new());
}

#[test]
fn real_host_refuses_an_explicit_gpu_request_it_cannot_meet() {
    without_runtime_overrides!(real_host_refuses_an_explicit_gpu_request_it_cannot_meet);
    let machine = Machine::new(&bundled_cpu(), None);
    let mut options = machine.options();
    options.selection = RuntimeSelection::parse(Some("cuda")).unwrap();
    let error = format!("{:#}", machine.install(options).unwrap_err());
    assert!(
        error.contains("no compatible native runtime found"),
        "{error}"
    );
}

#[test]
fn real_host_uses_a_better_runtime_it_already_has_without_the_release() {
    without_runtime_overrides!(real_host_uses_a_better_runtime_it_already_has_without_the_release);
    // The catalog is a 404.
    let machine = Machine::new(&bundled_cpu(), None);
    machine.already_installed(&better_than_bundled());
    let outcome = machine.install(machine.options()).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "better-download");
    assert!(outcome.runtime.path.starts_with(machine.cache().root()));

    // Downloads off.
    let machine = Machine::new(&bundled_cpu(), Some(&better_than_bundled()));
    machine.already_installed(&better_than_bundled());
    let mut options = machine.options();
    options.allow_download = false;
    let outcome = machine.install(options).unwrap();
    assert_eq!(outcome.runtime.native_runtime_id, "better-download");
    assert!(outcome.runtime.path.starts_with(machine.cache().root()));
    assert_eq!(*machine.requests.lock().unwrap(), Vec::<String>::new());
}

/// An x86_64 Linux machine, with or without an NVIDIA GPU and a CUDA 12 toolkit.
fn linux_machine(nvidia: bool) -> HostRuntimeProfile {
    let mut flavors = BTreeSet::from([NativeRuntimeBackendKind::Cpu]);
    if nvidia {
        flavors.insert(NativeRuntimeBackendKind::Cuda);
    }
    HostRuntimeProfile {
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
        target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
        glibc_version: Some("2.39".to_string()),
        available_flavors: flavors,
        gpus: Vec::new(),
        cuda: nvidia.then(|| HostCudaProfile {
            toolkit_majors: BTreeSet::from([12]),
            driver_max_major: Some(12),
            ..Default::default()
        }),
        rocm: None,
        vulkan: None,
    }
}

fn linux_runtime(id: &str, backend: NativeRuntimeBackend) -> NativeRuntimeArtifact {
    let mut artifact = compatible_artifact();
    artifact.id = id.to_string();
    artifact.backend = backend;
    artifact.platform = NativeRuntimePlatform {
        os: "linux".to_string(),
        arch: "x86_64".to_string(),
        target: Some("x86_64-unknown-linux-gnu".to_string()),
        min_glibc: Some("2.17".to_string()),
    };
    artifact
}

/// What the installer would fetch or reuse on `profile`. The installer reads
/// the profile from the machine it runs on and cannot be given another, so
/// this takes the same two steps it does (MeshLLM's merged catalog, then
/// Skippy's resolver) and stops before the download. No GPU is involved.
fn choice_on(profile: HostRuntimeProfile, machine: &Machine) -> (String, NativeRuntimeSource) {
    let mut options = mesh_native_runtime_manifest_options();
    options.catalog.releases_url = machine.base.clone();
    options.bundle_dirs = vec![machine.bundle.clone()];
    let (catalog, sources) = block_on(load_release_manifest_with_sources(options)).unwrap();
    let resolution =
        NativeRuntimeResolver::new(current_runtime_release(), profile, catalog, machine.cache())
            .with_skippy_abi_version(current_skippy_abi_version())
            .with_bundle_dirs(sources.bundle_dirs)
            .resolve(&RuntimeSelection::Recommended)
            .unwrap();
    (resolution.selected.id, resolution.source)
}

#[test]
fn synthetic_cuda_machine_chooses_the_cuda_runtime_over_the_bundled_cpu_runtime() {
    without_runtime_overrides!(
        synthetic_cuda_machine_chooses_the_cuda_runtime_over_the_bundled_cpu_runtime
    );
    let machine = Machine::new(
        &linux_runtime("bundled-cpu", NativeRuntimeBackend::cpu()),
        Some(&linux_runtime(
            "cuda12",
            NativeRuntimeBackend::cuda(12, Vec::new()),
        )),
    );
    let bundle = NativeRuntimeSource::Bundle {
        path: machine.bundle.clone(),
    };

    let (id, source) = choice_on(linux_machine(true), &machine);
    assert_eq!(id, "cuda12");
    assert_eq!(
        source,
        NativeRuntimeSource::Download {
            url: format!("{}/runtime.tar.gz", machine.base)
        }
    );

    // The same release on a machine without the GPU keeps the bundle.
    assert_eq!(
        choice_on(linux_machine(false), &machine),
        ("bundled-cpu".to_string(), bundle)
    );

    // The GPU machine keeps it too when the catalog is a 404 and the bundle
    // is all it has.
    let offline = Machine::new(
        &linux_runtime("bundled-cpu", NativeRuntimeBackend::cpu()),
        None,
    );
    let (id, source) = choice_on(linux_machine(true), &offline);
    assert_eq!(id, "bundled-cpu");
    assert!(matches!(source, NativeRuntimeSource::Bundle { .. }));

    // With the CUDA runtime already installed, it needs no release.
    offline.already_installed(&linux_runtime(
        "cuda12",
        NativeRuntimeBackend::cuda(12, Vec::new()),
    ));
    let (id, source) = choice_on(linux_machine(true), &offline);
    assert_eq!(id, "cuda12");
    assert!(matches!(source, NativeRuntimeSource::Installed { .. }));
}

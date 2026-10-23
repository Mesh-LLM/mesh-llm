use minisign::{KeyPair, PublicKeyBox};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
use skippy_runtime_install::{
    NativeRuntimeArtifact, NativeRuntimeCatalog, NativeRuntimeInstallOptions,
    NativeRuntimeManifest, NativeRuntimeReleaseManifest, NativeRuntimeSource,
    NativeRuntimeVerificationPolicy, RuntimeSelection, host_runtime_profile,
    install_native_runtime, install_native_runtime_explicit,
};
use std::io::Cursor;
use std::path::{Path, PathBuf};

const RELEASE: &str = "strict-fixture-1";
const ABI: &str = "0.0.strict-fixture";
const RUNTIME_ID: &str = "strict-fixture-runtime";

struct SignedFixture {
    _root: tempfile::TempDir,
    catalog_path: PathBuf,
    bundle_path: PathBuf,
    cache_path: PathBuf,
    public_key: String,
    key_pair: KeyPair,
    artifact: NativeRuntimeArtifact,
}

impl SignedFixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let bundle_path = root.path().join("bundle");
        let artifact = write_bundle(&bundle_path, b"authenticated runtime");
        let catalog_path = root.path().join("native-runtimes.json");
        let cache_path = root.path().join("cache");
        let key_pair = KeyPair::generate_unencrypted_keypair().unwrap();
        let public_key = key_pair.pk.to_box().unwrap().into_string();
        write_signed_catalog(&catalog_path, RELEASE, &artifact, &key_pair);
        Self {
            _root: root,
            catalog_path,
            bundle_path,
            cache_path,
            public_key,
            key_pair,
            artifact,
        }
    }

    fn options(&self) -> NativeRuntimeInstallOptions {
        let mut options = NativeRuntimeInstallOptions::new(RELEASE, test_catalog());
        options.skippy_abi_version = Some(ABI.to_string());
        options.selection = RuntimeSelection::Id(RUNTIME_ID.to_string());
        options.manifest_path = Some(self.catalog_path.clone());
        options.bundle_dirs = vec![self.bundle_path.clone()];
        options.cache_dir = Some(self.cache_path.clone());
        options.verification_policy = NativeRuntimeVerificationPolicy::RequireChecksumAndSignature;
        options.trusted_public_key = Some(self.public_key.clone());
        options.allow_download = false;
        options
    }
}

fn test_catalog() -> NativeRuntimeCatalog {
    NativeRuntimeCatalog {
        release_tags: Default::default(),
        releases_url: "https://example.invalid/releases".to_string(),
        rolling_release: None,
    }
}

fn write_bundle(root: &Path, payload: &[u8]) -> NativeRuntimeArtifact {
    let profile = host_runtime_profile();
    let min_glibc = (profile.os == "linux").then(|| "2.17".to_string());
    std::fs::create_dir_all(root.join("lib")).unwrap();
    std::fs::write(root.join("lib/runtime.bin"), payload).unwrap();
    NativeRuntimeManifest {
        runtime: NativeRuntimeArtifact {
            id: RUNTIME_ID.to_string(),
            release_version: Some(RELEASE.to_string()),
            skippy_abi: ABI.to_string(),
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
        },
    }
    .write_to_dir(root)
    .unwrap();
    let mut artifact = NativeRuntimeManifest::read_from_dir(root).unwrap().runtime;
    artifact.url = Some("https://example.invalid/runtime.tar.gz".to_string());
    artifact.sha256 = Some("a".repeat(64));
    artifact
}

fn write_signed_catalog(
    path: &Path,
    release: &str,
    artifact: &NativeRuntimeArtifact,
    key_pair: &KeyPair,
) {
    let mut artifact = artifact.clone();
    artifact.release_version = Some(release.to_string());
    let manifest = NativeRuntimeReleaseManifest {
        release_version: release.to_string(),
        skippy_abi: ABI.to_string(),
        artifacts: vec![artifact],
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    bytes.push(b'\n');
    std::fs::write(path, &bytes).unwrap();
    write_checksum(path, &bytes);
    let signature = minisign::sign(
        Some(&key_pair.pk),
        &key_pair.sk,
        Cursor::new(&bytes),
        None,
        None,
    )
    .unwrap()
    .into_string();
    std::fs::write(sidecar(path, ".minisig"), signature).unwrap();
}

fn write_checksum(path: &Path, bytes: &[u8]) {
    let checksum = hex::encode(Sha256::digest(bytes));
    std::fs::write(
        sidecar(path, ".sha256"),
        format!("{checksum}  native-runtimes.json\n"),
    )
    .unwrap();
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(suffix);
    PathBuf::from(sidecar)
}

fn install(
    options: NativeRuntimeInstallOptions,
) -> anyhow::Result<skippy_runtime_install::NativeRuntimeInstallOutcome> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(install_native_runtime_explicit(options))
}

#[test]
fn offline_signed_local_catalog_and_bundle_are_accepted() {
    let fixture = SignedFixture::new();
    let outcome = install(fixture.options()).unwrap();

    let NativeRuntimeSource::Bundle { path } = outcome.resolution.source else {
        panic!("expected the authenticated bundle to satisfy the install");
    };
    assert_eq!(
        path.canonicalize().unwrap(),
        fixture.bundle_path.canonicalize().unwrap()
    );
}

#[test]
fn catalog_tampering_is_rejected_even_when_the_checksum_is_rewritten() {
    let fixture = SignedFixture::new();
    let mut bytes = std::fs::read(&fixture.catalog_path).unwrap();
    bytes.extend_from_slice(b" \n");
    std::fs::write(&fixture.catalog_path, &bytes).unwrap();
    write_checksum(&fixture.catalog_path, &bytes);

    let error = install(fixture.options()).unwrap_err();
    assert!(
        format!("{error:#}").contains("Minisign signature"),
        "{error:#}"
    );
}

#[test]
fn missing_and_wrong_signatures_are_rejected() {
    let fixture = SignedFixture::new();
    std::fs::remove_file(sidecar(&fixture.catalog_path, ".minisig")).unwrap();
    let error = install(fixture.options()).unwrap_err();
    assert!(format!("{error:#}").contains("detached native runtime catalog signature"));

    let fixture = SignedFixture::new();
    let wrong_key = KeyPair::generate_unencrypted_keypair().unwrap();
    let bytes = std::fs::read(&fixture.catalog_path).unwrap();
    let wrong_signature = minisign::sign(
        Some(&wrong_key.pk),
        &wrong_key.sk,
        Cursor::new(bytes),
        None,
        None,
    )
    .unwrap()
    .into_string();
    std::fs::write(sidecar(&fixture.catalog_path, ".minisig"), wrong_signature).unwrap();
    let error = install(fixture.options()).unwrap_err();
    assert!(
        format!("{error:#}").contains("Minisign signature"),
        "{error:#}"
    );
}

#[test]
fn wrong_trust_key_and_wrong_signed_release_are_rejected() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.trusted_public_key = Some(
        KeyPair::generate_unencrypted_keypair()
            .unwrap()
            .pk
            .to_box()
            .unwrap()
            .into_string(),
    );
    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("Minisign signature"),
        "{error:#}"
    );

    let fixture = SignedFixture::new();
    write_signed_catalog(
        &fixture.catalog_path,
        "another-release",
        &fixture.artifact,
        &fixture.key_pair,
    );
    let error = install(fixture.options()).unwrap_err();
    assert!(
        format!("{error:#}").contains("does not match requested release"),
        "{error:#}"
    );
}

#[test]
fn strict_mode_has_no_unsigned_bundle_fallback() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.manifest_path = None;

    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("unsigned bundle-only installation is not allowed"),
        "{error:#}"
    );
}

#[test]
fn strict_mode_requires_an_independently_supplied_trust_key() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.trusted_public_key = None;

    let error = install(options).unwrap_err();
    assert!(format!("{error:#}").contains("requires a trusted Minisign public key"));
}

#[test]
fn public_key_fixture_is_a_minisign_key_file_not_catalog_data() {
    let fixture = SignedFixture::new();
    PublicKeyBox::from_string(&fixture.public_key)
        .unwrap()
        .into_public_key()
        .unwrap();
    assert!(!fixture.public_key.contains("example.invalid"));
}

fn serve_archive(bytes: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/runtime.tar.gz", listener.local_addr().unwrap());
    let worker = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        let mut request = [0; 4096];
        let read = stream.read(&mut request).unwrap();
        assert!(read > 0);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        )
        .unwrap();
        stream.write_all(&bytes).unwrap();
    });
    (url, worker)
}

fn bundle_archive(path: &Path) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);
    archive.append_dir_all("runtime", path).unwrap();
    archive.into_inner().unwrap().finish().unwrap()
}

#[test]
fn strict_download_verifies_archive_and_reuses_authenticated_cache() {
    let mut fixture = SignedFixture::new();
    let bytes = bundle_archive(&fixture.bundle_path);
    fixture.artifact.sha256 = Some(hex::encode(Sha256::digest(&bytes)));
    let (url, worker) = serve_archive(bytes);
    fixture.artifact.url = Some(url);
    write_signed_catalog(
        &fixture.catalog_path,
        RELEASE,
        &fixture.artifact,
        &fixture.key_pair,
    );
    let mut options = fixture.options();
    options.bundle_dirs.clear();
    options.allow_download = true;
    let result = install(options.clone());
    worker.join().unwrap();
    result.unwrap();
    options.allow_download = false;
    let cached = install(options).unwrap();
    assert!(matches!(
        cached.resolution.source,
        NativeRuntimeSource::Installed { .. }
    ));
}

#[test]
fn strict_download_rejects_archive_tampering_before_install() {
    let mut fixture = SignedFixture::new();
    let bytes = bundle_archive(&fixture.bundle_path);
    fixture.artifact.sha256 = Some(hex::encode(Sha256::digest(&bytes)));
    let (url, worker) = serve_archive(b"replaced archive".to_vec());
    fixture.artifact.url = Some(url);
    write_signed_catalog(
        &fixture.catalog_path,
        RELEASE,
        &fixture.artifact,
        &fixture.key_pair,
    );
    let mut options = fixture.options();
    options.bundle_dirs.clear();
    options.allow_download = true;
    let result = install(options);
    worker.join().unwrap();
    let error = result.unwrap_err();
    assert!(
        format!("{error:#}").contains("checksum mismatch"),
        "{error:#}"
    );
}

#[test]
fn signed_catalog_cannot_override_requested_abi_or_host_platform() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.skippy_abi_version = Some("wrong-abi".into());
    assert!(install(options).is_err());

    let mut fixture = SignedFixture::new();
    fixture.artifact.platform.os = "incompatible-os".into();
    write_signed_catalog(
        &fixture.catalog_path,
        RELEASE,
        &fixture.artifact,
        &fixture.key_pair,
    );
    let mut options = fixture.options();
    options.bundle_dirs.clear();
    assert!(install(options).is_err());
}

#[test]
fn constructors_embed_the_reviewed_production_pin_without_enabling_strict_policy() {
    let install = NativeRuntimeInstallOptions::new(RELEASE, test_catalog());
    let manifest =
        skippy_runtime_install::NativeRuntimeManifestOptions::new(RELEASE, test_catalog());
    let pin = include_str!("../trust/native-runtime-catalog.pub");
    assert_eq!(install.trusted_public_key.as_deref(), Some(pin));
    assert_eq!(manifest.trusted_public_key.as_deref(), Some(pin));
    assert_eq!(
        install.verification_policy,
        NativeRuntimeVerificationPolicy::RequireChecksum
    );
    PublicKeyBox::from_string(pin)
        .unwrap()
        .into_public_key()
        .unwrap();
}

/// Re-sign the fixture catalog so it lists only an official runtime with a
/// different id, as when a distro ships a runtime it built itself.
fn sign_catalog_without_bundle(fixture: &SignedFixture) {
    let mut official = fixture.artifact.clone();
    official.id = "official-runtime".to_string();
    write_signed_catalog(&fixture.catalog_path, RELEASE, &official, &fixture.key_pair);
}

fn assert_bundle_source(
    outcome: &skippy_runtime_install::NativeRuntimeInstallOutcome,
    fixture: &SignedFixture,
) {
    let NativeRuntimeSource::Bundle { path } = &outcome.resolution.source else {
        panic!("expected the shipped bundle to satisfy the install");
    };
    assert_eq!(
        path.canonicalize().unwrap(),
        fixture.bundle_path.canonicalize().unwrap()
    );
}

#[test]
fn strict_mode_rejects_a_shipped_bundle_the_catalog_does_not_list() {
    let fixture = SignedFixture::new();
    sign_catalog_without_bundle(&fixture);

    let error = install(fixture.options()).unwrap_err();
    assert!(
        format!("{error:#}").contains("is not listed by the authenticated catalog"),
        "{error:#}"
    );
}

#[test]
fn trusted_explicit_bundles_accept_a_shipped_bundle_the_catalog_does_not_list() {
    let fixture = SignedFixture::new();
    sign_catalog_without_bundle(&fixture);
    let mut options = fixture.options();
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let outcome = install(options).unwrap();
    assert_bundle_source(&outcome, &fixture);
}

#[test]
fn trusted_explicit_bundles_work_offline_without_a_catalog() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.manifest_path = None;
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let outcome = install(options).unwrap();
    assert_bundle_source(&outcome, &fixture);
}

#[test]
fn trusted_explicit_bundles_still_require_a_signed_catalog_without_bundles() {
    let fixture = SignedFixture::new();
    let mut options = fixture.options();
    options.manifest_path = None;
    options.bundle_dirs.clear();
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("unsigned bundle-only installation is not allowed"),
        "{error:#}"
    );
}

#[test]
fn trusted_explicit_bundles_still_reject_a_tampered_catalog() {
    let fixture = SignedFixture::new();
    let mut bytes = std::fs::read(&fixture.catalog_path).unwrap();
    bytes.extend_from_slice(b" ");
    std::fs::write(&fixture.catalog_path, &bytes).unwrap();
    write_checksum(&fixture.catalog_path, &bytes);
    let mut options = fixture.options();
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    assert!(install(options).is_err());
}

/// Serves `files` (path -> bytes) for `requests` connections; unknown paths 404.
fn serve_files(
    files: Vec<(&'static str, Vec<u8>)>,
    requests: usize,
) -> (String, std::thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let worker = std::thread::spawn(move || {
        for _ in 0..requests {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut request = [0; 4096];
            let read = stream.read(&mut request).unwrap();
            let line = String::from_utf8_lossy(&request[..read]);
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            match files.iter().find(|(name, _)| *name == path) {
                Some((_, body)) => {
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .unwrap();
                    stream.write_all(body).unwrap();
                }
                None => write!(
                    stream,
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .unwrap(),
            }
        }
    });
    (base, worker)
}

#[test]
fn trusted_bundles_do_not_fall_back_when_a_remote_catalog_fails_authentication() {
    let fixture = SignedFixture::new();
    let mut catalog = std::fs::read(&fixture.catalog_path).unwrap();
    catalog.extend_from_slice(b" ");
    let checksum = format!(
        "{}  native-runtimes.json\n",
        hex::encode(Sha256::digest(&catalog))
    );
    let signature = std::fs::read(sidecar(&fixture.catalog_path, ".minisig")).unwrap();
    let (base, worker) = serve_files(
        vec![
            ("/native-runtimes.json", catalog),
            ("/native-runtimes.json.sha256", checksum.into_bytes()),
            ("/native-runtimes.json.minisig", signature),
        ],
        3,
    );
    let mut options = fixture.options();
    options.manifest_path = None;
    options.manifest_url = Some(format!("{base}/native-runtimes.json"));
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let error = install(options).unwrap_err();
    worker.join().unwrap();
    assert!(
        format!("{error:#}").contains("failed authentication"),
        "{error:#}"
    );
}

#[test]
fn trusted_bundles_still_fall_back_when_the_remote_catalog_is_unreachable() {
    let fixture = SignedFixture::new();
    let (base, worker) = serve_files(Vec::new(), 1);
    let mut options = fixture.options();
    options.manifest_path = None;
    options.manifest_url = Some(format!("{base}/native-runtimes.json"));
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let outcome = install(options).unwrap();
    worker.join().unwrap();
    assert_bundle_source(&outcome, &fixture);
}

#[test]
fn trusted_bundles_include_packager_discovered_bundle_dirs() {
    // Distro packages point MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR at their
    // runtime; the host passes no bundle_dirs. Every other test here uses
    // the explicit-only install path, which ignores this variable.
    let fixture = SignedFixture::new();
    sign_catalog_without_bundle(&fixture);
    let mut options = fixture.options();
    options.bundle_dirs.clear();
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;
    // SAFETY: no other test in this binary reads this variable.
    unsafe { std::env::set_var("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &fixture.bundle_path) };
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(install_native_runtime(options));
    unsafe { std::env::remove_var("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR") };
    assert_bundle_source(&result.unwrap(), &fixture);
}

#[test]
fn strict_mode_rejects_an_unlisted_file_in_a_bundle() {
    let fixture = SignedFixture::new();
    std::fs::write(fixture.bundle_path.join("lib/injected.dll"), b"evil").unwrap();

    let error = install(fixture.options()).unwrap_err();
    assert!(
        format!("{error:#}").contains("lib/injected.dll, which the signed catalog does not list"),
        "{error:#}"
    );
}

#[test]
fn strict_mode_allows_top_level_docs_in_a_bundle() {
    let fixture = SignedFixture::new();
    std::fs::write(fixture.bundle_path.join("README.md"), b"docs").unwrap();

    let outcome = install(fixture.options()).unwrap();
    assert_bundle_source(&outcome, &fixture);
}

#[test]
fn strict_mode_rejects_a_modified_library_in_a_bundle() {
    let fixture = SignedFixture::new();
    std::fs::write(fixture.bundle_path.join("lib/runtime.bin"), b"tampered").unwrap();

    assert!(install(fixture.options()).is_err());
}

#[test]
fn strict_mode_rehashes_the_cache_on_every_install() {
    let mut fixture = SignedFixture::new();
    let bytes = bundle_archive(&fixture.bundle_path);
    fixture.artifact.sha256 = Some(hex::encode(Sha256::digest(&bytes)));
    let (url, worker) = serve_archive(bytes);
    fixture.artifact.url = Some(url);
    write_signed_catalog(
        &fixture.catalog_path,
        RELEASE,
        &fixture.artifact,
        &fixture.key_pair,
    );
    let mut options = fixture.options();
    options.bundle_dirs.clear();
    options.allow_download = true;
    let result = install(options.clone());
    worker.join().unwrap();
    let installed = result.unwrap();
    std::fs::write(installed.runtime.path.join("lib/extra.so"), b"evil").unwrap();

    options.allow_download = false;
    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("lib/extra.so, which the signed catalog does not list"),
        "{error:#}"
    );
}

#[test]
fn trusted_bundles_skip_the_closed_directory_check() {
    let fixture = SignedFixture::new();
    std::fs::write(fixture.bundle_path.join("lib/distro-extra.so"), b"packager").unwrap();
    let mut options = fixture.options();
    options.verification_policy =
        NativeRuntimeVerificationPolicy::RequireSignatureTrustExplicitBundles;

    let outcome = install(options).unwrap();
    assert_bundle_source(&outcome, &fixture);
}

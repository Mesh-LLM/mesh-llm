use minisign::{KeyPair, PublicKeyBox};
use sha2::{Digest, Sha256};
use skippy_native_runtime::{NativeRuntimeBackend, NativeRuntimePlatform};
use skippy_runtime_install::{
    NativeRuntimeArtifact, NativeRuntimeCache, NativeRuntimeCatalog, NativeRuntimeInstallOptions,
    NativeRuntimeManifest, NativeRuntimeReleaseManifest, NativeRuntimeSource,
    NativeRuntimeVerificationPolicy, RuntimeSelection, host_runtime_profile,
    install_native_runtime_explicit,
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
fn self_consistent_tampered_bundle_cannot_bypass_signed_file_hashes() {
    let fixture = SignedFixture::new();
    let tampered = fixture._root.path().join("tampered-bundle");
    write_bundle(&tampered, b"tampered runtime");
    let mut options = fixture.options();
    options.bundle_dirs = vec![tampered];

    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("authenticated catalog entry"),
        "{error:#}"
    );
}

#[test]
fn self_consistent_tampered_cache_cannot_bypass_signed_file_hashes() {
    let fixture = SignedFixture::new();
    let tampered = fixture._root.path().join("tampered-cache-source");
    write_bundle(&tampered, b"tampered runtime");
    NativeRuntimeCache::new(&fixture.cache_path)
        .install_from_dir(&tampered)
        .unwrap();
    let mut options = fixture.options();
    options.bundle_dirs.clear();

    let error = install(options).unwrap_err();
    assert!(
        format!("{error:#}").contains("authenticated catalog entry"),
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

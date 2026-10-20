//! Real archive download/install/load qualification. No production secrets enter this process.
//! Run explicitly with SIGNED_RUNTIME_DIAGNOSTIC_DIR containing public artifacts.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};
use skippy_runtime_install::{
    NativeRuntimeCatalog, NativeRuntimeInstallOptions, NativeRuntimeReleaseManifest,
    NativeRuntimeVerificationPolicy, RuntimeSelection, install_native_runtime_explicit,
};

struct Server {
    root: PathBuf,
    faults: Arc<Mutex<Fault>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    MissingSignature,
    InvalidSignature,
    Catalog,
    Archive,
    Stop,
}

impl Server {
    fn start(root: &Path) -> Self {
        let listener =
            TcpListener::bind("127.0.0.1:18764").expect("isolated diagnostic port unavailable");
        let faults = Arc::new(Mutex::new(Fault::None));
        let state = faults.clone();
        let files = root.to_path_buf();
        let worker = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let fault = *state.lock().unwrap();
                if matches!(fault, Fault::Stop) {
                    break;
                }
                serve(stream.unwrap(), &files, fault);
            }
        });
        Self {
            root: root.to_path_buf(),
            faults,
            worker: Some(worker),
        }
    }

    fn set(&self, fault: Fault) {
        *self.faults.lock().unwrap() = fault;
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.set(Fault::Stop);
        let _ = TcpStream::connect("127.0.0.1:18764");
        self.worker.take().unwrap().join().unwrap();
    }
}

fn serve(mut stream: TcpStream, root: &Path, fault: Fault) {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut request = [0; 8192];
    let size = stream.read(&mut request).unwrap();
    let request = String::from_utf8_lossy(&request[..size]);
    let path = request.split_whitespace().nth(1).unwrap_or("");
    let name = match path {
        "/native-runtimes.json" => "native-runtimes.json",
        "/native-runtimes.json.sha256" => "native-runtimes.json.sha256",
        "/native-runtimes.json.minisig" if !matches!(fault, Fault::MissingSignature) => {
            "native-runtimes.json.minisig"
        }
        "/runtime.tar.gz" => "runtime.tar.gz",
        _ => {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            return;
        }
    };
    let mut bytes = std::fs::read(root.join(name)).unwrap();
    if matches!(fault, Fault::InvalidSignature) && name == "native-runtimes.json.minisig" {
        bytes = b"invalid detached signature".to_vec();
    }
    if matches!(fault, Fault::Catalog) {
        if name == "native-runtimes.json" {
            bytes.push(b' ');
        } else if name == "native-runtimes.json.sha256" {
            let mut catalog = std::fs::read(root.join("native-runtimes.json")).unwrap();
            catalog.push(b' ');
            bytes = format!(
                "{}  native-runtimes.json\n",
                hex::encode(Sha256::digest(catalog))
            )
            .into_bytes();
        }
    }
    if matches!(fault, Fault::Archive) && name == "runtime.tar.gz" {
        bytes[0] ^= 1;
    }
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    let _ = stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(&bytes));
}

fn options(server: &Server, cache: &Path) -> NativeRuntimeInstallOptions {
    let manifest: NativeRuntimeReleaseManifest =
        serde_json::from_slice(&std::fs::read(server.root.join("native-runtimes.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.artifacts.len(), 1);
    let mut options = NativeRuntimeInstallOptions::new(
        manifest.release_version,
        NativeRuntimeCatalog {
            release_tags: Default::default(),
            releases_url: "http://127.0.0.1:18764".into(),
            rolling_release: None,
        },
    );
    options.skippy_abi_version = Some(skippy_runtime_install::current_skippy_abi_version());
    options.selection = RuntimeSelection::Id(manifest.artifacts[0].id.clone());
    options.manifest_url = Some("http://127.0.0.1:18764/native-runtimes.json".into());
    options.cache_dir = Some(cache.to_path_buf());
    options.verification_policy = NativeRuntimeVerificationPolicy::RequireChecksumAndSignature;
    // Local disposable-key runs use an explicit override. Production diagnostics
    // deliberately omit this file so verification uses the compiled pin.
    let override_path = server.root.join("disposable-test-key.pub");
    if std::env::var_os("SIGNED_RUNTIME_DISPOSABLE_KEY").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        options.trusted_public_key = Some(std::fs::read_to_string(override_path).unwrap());
    } else {
        assert!(
            !override_path.exists(),
            "production qualification forbids a downloaded trust pin"
        );
        assert_eq!(
            options.trusted_public_key.as_deref(),
            Some(include_str!("../trust/native-runtime-catalog.pub"))
        );
    }
    options
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
#[ignore = "requires a real signed runtime archive; run in an isolated diagnostic"]
fn authenticated_http_download_install_tamper_rejection_and_native_load() {
    assert!(std::env::var_os("MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY").is_none());
    assert!(std::env::var_os("MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_PASSWORD").is_none());
    let root = PathBuf::from(
        std::env::var_os("SIGNED_RUNTIME_DIAGNOSTIC_DIR").expect("public artifact directory"),
    );
    let server = Server::start(&root);
    let temp = tempfile::tempdir().unwrap();
    for (fault, expected) in [
        (Fault::MissingSignature, "404"),
        (Fault::InvalidSignature, "Minisign signature"),
        (Fault::Catalog, "Minisign signature"),
        (Fault::Archive, "checksum"),
    ] {
        server.set(fault);
        let error = install(options(&server, &temp.path().join("rejected"))).unwrap_err();
        assert!(format!("{error:#}").contains(expected), "{error:#}");
        assert!(!skippy_ffi::native_runtime_loaded());
    }
    server.set(Fault::None);
    let mut wrong_key = options(&server, &temp.path().join("wrong-key"));
    wrong_key.trusted_public_key = Some(
        minisign::KeyPair::generate_unencrypted_keypair()
            .unwrap()
            .pk
            .to_box()
            .unwrap()
            .into_string(),
    );
    let error = install(wrong_key).unwrap_err();
    assert!(
        format!("{error:#}").contains("Minisign signature"),
        "{error:#}"
    );
    assert!(!skippy_ffi::native_runtime_loaded());

    let cache = temp.path().join("accepted");
    let outcome = install(options(&server, &cache)).unwrap();
    assert_eq!(
        outcome.status,
        skippy_runtime_install::NativeRuntimeInstallStatus::Installed
    );
    let plan = outcome.runtime.load_plan().unwrap();
    let library = &plan.libraries[0];
    let original = std::fs::read(library).unwrap();
    std::fs::write(library, b"tampered cached library").unwrap();
    let mut offline = options(&server, &cache);
    offline.allow_download = false;
    assert!(
        install(offline).is_err(),
        "tampered cache must not be accepted"
    );
    assert!(!skippy_ffi::native_runtime_loaded());
    std::fs::write(library, original).unwrap();
    let verified = install(options(&server, &cache)).unwrap();
    let plan = verified.runtime.load_plan().unwrap();
    // SAFETY: the strict installer authenticated this archive and all loadable
    // files against the independent pin and checked the compiled ABI contract.
    unsafe { skippy_ffi::load_native_runtime_libraries(&plan.libraries) }.unwrap();
    assert!(skippy_ffi::native_runtime_loaded());
}

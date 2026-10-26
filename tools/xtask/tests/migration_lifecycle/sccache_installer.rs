//! Unchanged copied installer and inert local archive; no download/install outside scratch.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use flate2::{Compression, write::GzEncoder};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
const VERSION: &str = "0.16.0";
const ARCHIVE: &str = "sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz";
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("finite installer tool PATH required"))
        .map(|path| path.join(name))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .unwrap_or_else(|| panic!("finite copied installer fixture requires component {name}"))
        .canonicalize()
        .unwrap()
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("installer fixture with spaces");
        for child in ["bin", "source", "cache", "install", "scratch"] {
            fs::create_dir_all(root.join(child)).unwrap();
        }
        for name in [
            "bash",
            "sh",
            "mkdir",
            "awk",
            "sha256sum",
            "rm",
            "mktemp",
            "mv",
            "tar",
            "gzip",
            "install",
            "cp",
        ] {
            std::os::unix::fs::symlink(tool(name), root.join("bin").join(name)).unwrap();
        }
        fs::copy(
            repo().join("scripts/install-sccache.sh"),
            root.join("actual-installer.sh"),
        )
        .unwrap();
        executable(
            &root.join("bin/curl"),
            include_str!("sccache_installer/curl.sh"),
        );
        let fixture = Self { _temp: temp, root };
        fixture.artifact();
        fixture
    }
    fn artifact(&self) {
        let body = b"#!/bin/sh\nprintf '%s\\n' called >> \"$FIXTURE_ROOT/installed.calls\"\nprintf 'sccache 0.16.0\\n'\n";
        let gzip = GzEncoder::new(Vec::new(), Compression::default());
        let mut archive = tar::Builder::new(gzip);
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o755);
        header.set_mtime(0);
        header.set_cksum();
        archive
            .append_data(
                &mut header,
                "sccache-v0.16.0-x86_64-unknown-linux-musl/sccache",
                &body[..],
            )
            .unwrap();
        let bytes = archive.into_inner().unwrap().finish().unwrap();
        let digest = hex::encode(Sha256::digest(&bytes));
        fs::write(self.root.join("source").join(ARCHIVE), &bytes).unwrap();
        fs::write(
            self.root.join("source").join(format!("{ARCHIVE}.sha256")),
            format!("{digest}\n"),
        )
        .unwrap();
    }
    fn corrupt(&self, mismatch: bool) {
        fs::write(
            self.root.join("cache").join(ARCHIVE),
            b"corrupt existing archive",
        )
        .unwrap();
        let checksum = if mismatch {
            "0".repeat(64)
        } else {
            "invalid".into()
        };
        fs::write(
            self.root.join("cache").join(format!("{ARCHIVE}.sha256")),
            checksum,
        )
        .unwrap();
    }
    fn run(&self, failure: &str) -> Output {
        let environment = [
            ("PATH", self.root.join("bin").into_os_string()),
            ("HOME", self.root.clone().into_os_string()),
            ("TMPDIR", self.root.join("scratch").into_os_string()),
            (
                "DOWNLOAD_CACHE_DIR",
                self.root.join("cache").into_os_string(),
            ),
            (
                "SCCACHE_INSTALL_DIR",
                self.root.join("install").into_os_string(),
            ),
            ("FIXTURE_ROOT", self.root.clone().into_os_string()),
            ("SCCACHE_VERSION", VERSION.into()),
            ("TARGETARCH", "amd64".into()),
            ("FAIL_AT", failure.into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value)))
        .collect::<BTreeMap<_, _>>();
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: tool("bash"),
                cwd: self.root.clone(),
                environment,
                arguments: vec![Value::Public(
                    self.root.join("actual-installer.sh").into_os_string(),
                )],
            },
            &Limits {
                execution: Duration::from_secs(15),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(result.process.failure.is_none(), "{:?}", result.process);
        assert!(result.process.cleanup.complete);
        Output {
            code: result.process.status.unwrap().code().unwrap(),
            stdout: String::from_utf8_lossy(result.stdout.unwrap().as_bytes()).into_owned(),
            stderr: String::from_utf8_lossy(result.stderr.unwrap().as_bytes()).into_owned(),
        }
    }
    fn calls(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("curl.calls"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn no_scratch(&self) {
        assert!(
            fs::read_dir(self.root.join("scratch"))
                .unwrap()
                .next()
                .is_none()
        );
        for entry in fs::read_dir(self.root.join("cache")).unwrap() {
            assert!(
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".tmp.")
            );
        }
    }
}
#[test]
fn sccache_installer_corrupt_cache_recovers_then_valid_cache_is_offline() {
    for mismatch in [false, true] {
        let fixture = Fixture::new();
        fixture.corrupt(mismatch);
        let first = fixture.run("");
        assert_eq!(first.code, 0, "{}", first.stderr);
        assert!(first.stdout.contains("sccache 0.16.0"));
        for file in [ARCHIVE.to_owned(), format!("{ARCHIVE}.sha256")] {
            assert_eq!(
                fs::read(fixture.root.join("cache").join(&file)).unwrap(),
                fs::read(fixture.root.join("source").join(&file)).unwrap()
            );
        }
        let installed = fixture.root.join("install/sccache");
        assert!(installed.is_file());
        assert_eq!(
            fs::metadata(&installed).unwrap().permissions().mode() & 0o777,
            0o755
        );
        assert_eq!(
            fixture.calls(),
            [ARCHIVE.to_owned(), format!("{ARCHIVE}.sha256")]
        );
        fixture.no_scratch();
        let offline = fixture.run("offline");
        assert_eq!(offline.code, 0, "{}", offline.stderr);
        assert!(offline.stdout.contains("sccache 0.16.0"));
        assert_eq!(
            fixture.calls().len(),
            2,
            "valid cache must not call curl even when curl would fail"
        );
        assert_eq!(
            fs::read_to_string(fixture.root.join("installed.calls")).unwrap(),
            "called\ncalled\n"
        );
        fixture.no_scratch();
    }
}
#[test]
fn sccache_installer_failed_refresh_discards_corrupt_cache_preserves_unrelated_and_installed() {
    for failure in ["archive", "checksum", "digest"] {
        let fixture = Fixture::new();
        fixture.corrupt(false);
        fs::write(
            fixture.root.join("cache/unrelated-entry"),
            b"unrelated cache retained",
        )
        .unwrap();
        let installed = fixture.root.join("install/sccache");
        executable(&installed, "#!/bin/sh\nprintf 'old installed fixture\\n'\n");
        let original = fs::read(&installed).unwrap();
        let output = fixture.run(failure);
        assert_ne!(output.code, 0, "{failure}: {}", output.stderr);
        assert!(!fixture.root.join("cache").join(ARCHIVE).exists());
        assert!(
            !fixture
                .root
                .join("cache")
                .join(format!("{ARCHIVE}.sha256"))
                .exists()
        );
        assert_eq!(fs::read(&installed).unwrap(), original);
        assert_eq!(
            fs::read(fixture.root.join("cache/unrelated-entry")).unwrap(),
            b"unrelated cache retained"
        );
        assert!(!fixture.root.join("installed.calls").exists());
        assert_eq!(
            fixture.calls().len(),
            if failure == "archive" { 1 } else { 2 }
        );
        fixture.no_scratch();
    }
}

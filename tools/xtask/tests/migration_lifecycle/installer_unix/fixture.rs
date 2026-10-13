//! Private environment and local release artifacts for the actual Unix installer.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessReport, ProcessSpec, Readiness,
    Value,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) const PREFERRED: &str = "mesh-llm-aarch64-apple-darwin.tar.gz";
pub(super) const FALLBACK: &str = "mesh-bundle.tar.gz";

fn tool(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join(name))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .map(|path| path.canonicalize().unwrap())
}

pub(super) struct Fixture {
    _temporary: tempfile::TempDir,
    pub(super) root: PathBuf,
}

impl Fixture {
    pub(super) fn stub(&self, name: &str, body: &str) {
        let path = self.root.join("tools").join(name);
        if fs::symlink_metadata(&path).is_ok() {
            fs::remove_file(&path).unwrap();
        }
        fs::write(&path, body).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    pub(super) fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("installer's fixture with spaces");
        for child in ["tools", "assets", "download", "install", "home", "scratch"] {
            fs::create_dir_all(root.join(child)).unwrap();
        }
        for name in [
            "bash", "curl", "rm", "awk", "grep", "head", "tr", "cp", "mkdir", "mv", "mktemp",
            "find", "basename", "sed", "sort", "cat", "tar", "gzip", "uname",
        ] {
            std::os::unix::fs::symlink(
                tool(name).unwrap_or_else(|| panic!("installer fixture requires {name}")),
                root.join("tools").join(name),
            )
            .unwrap();
        }
        let checksum = if tool("sha256sum").is_some() {
            "sha256sum"
        } else {
            "shasum"
        };
        std::os::unix::fs::symlink(
            tool(checksum).expect("installer checksum tool"),
            root.join("tools").join(checksum),
        )
        .unwrap();
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        fs::copy(
            repository.join("install.sh"),
            root.join("actual-install.sh"),
        )
        .unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }

    pub(super) fn asset(&self, name: &str, bytes: &[u8]) {
        fs::write(self.root.join("assets").join(name), bytes).unwrap();
        fs::write(
            self.root.join("assets").join(format!("{name}.sha256")),
            format!("{}  {name}\n", hex::encode(Sha256::digest(bytes))),
        )
        .unwrap();
    }

    pub(super) fn run(&self, body: &str) -> ProcessReport {
        let url = url::Url::from_directory_path(self.root.join("assets")).unwrap();
        let environment = [
            ("PATH", self.root.join("tools").into_os_string()),
            ("HOME", self.root.join("home").into_os_string()),
            ("TMPDIR", self.root.join("scratch").into_os_string()),
            ("LC_ALL", "C".into()),
            ("FIXTURE_ROOT", self.root.clone().into_os_string()),
            (
                "MESH_LLM_TEST_CUDA_PROBE_ROOT",
                self.root.join("probe").into_os_string(),
            ),
            ("MESH_LLM_INSTALL_URL_BASE", url.to_string().into()),
            ("MESH_LLM_INSTALL_VERBOSE", "1".into()),
            ("MESH_LLM_REQUIRE_CHECKSUM", "1".into()),
            (
                "MESH_LLM_INSTALL_DIR",
                self.root.join("install").into_os_string(),
            ),
            (
                "FIXTURE_INSTALL_SCRIPT",
                self.root.join("actual-install.sh").into_os_string(),
            ),
            (
                "FIXTURE_DOWNLOAD_DIR",
                self.root.join("download").into_os_string(),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value)))
        .collect::<BTreeMap<_, _>>();
        let report = process::supervise(
            &ProcessSpec {
                executable: tool("bash").unwrap(),
                arguments: vec![
                    Value::Public("-c".into()),
                    Value::Public(
                        format!("set -euo pipefail\nsource \"$FIXTURE_INSTALL_SCRIPT\"\n{body}")
                            .into(),
                    ),
                ],
                cwd: self.root.clone(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 32768,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(report.cleanup.complete, "{report:?}");
        report
    }

    pub(super) fn download(&self) -> ProcessReport {
        self.run(&format!(
            "download_release_archive \"$FIXTURE_DOWNLOAD_DIR\" '{PREFERRED}'\nprintf 'asset=%s\\narchive=%s\\n' \"$DOWNLOADED_ASSET\" \"$DOWNLOADED_ARCHIVE\""
        ))
    }
}

pub(super) fn stdout(report: &ProcessReport) -> String {
    String::from_utf8(report.stdout.bytes_retained.clone()).unwrap()
}

pub(super) fn stderr(report: &ProcessReport) -> String {
    String::from_utf8(report.stderr.bytes_retained.clone()).unwrap()
}

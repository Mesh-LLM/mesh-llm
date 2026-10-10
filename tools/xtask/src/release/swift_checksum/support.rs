use super::capture::NativeSwift;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(super) fn fixture() -> PathBuf {
    let executable = std::env::current_exe().unwrap();
    let profile = executable.parent().unwrap().parent().unwrap();
    let fixture = profile.join("examples").join(format!(
        "migration_swift_checksum_fixture{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(fixture.is_file(), "build the inert checksum example first");
    fixture
}

pub(super) fn native(root: &Path, mode: &str) -> NativeSwift {
    std::fs::write(root.join(mode), b"inert artifact").unwrap();
    NativeSwift {
        executable: fixture(),
        cwd: root.to_owned(),
        artifact: mode.into(),
        timeout: Duration::from_secs(3),
        max_bytes: std::num::NonZeroUsize::new(1024).unwrap(),
    }
}

pub(super) fn wait_for(path: &Path) {
    let until = Instant::now() + Duration::from_secs(2);
    while !path.is_file() {
        assert!(Instant::now() < until, "fixture readiness deadline");
        std::thread::park_timeout(Duration::from_millis(2));
    }
}

pub(super) struct Sentinel(std::process::Child);

impl Sentinel {
    pub(super) fn start(root: &Path) -> Self {
        Self(
            std::process::Command::new(fixture())
                .arg("sentinel")
                .current_dir(root)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap(),
        )
    }

    pub(super) fn assert_alive(&mut self) {
        assert!(self.0.try_wait().unwrap().is_none());
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        self.0.kill().unwrap();
        self.0.wait().unwrap();
    }
}

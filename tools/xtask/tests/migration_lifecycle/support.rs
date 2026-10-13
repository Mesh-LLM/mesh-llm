use crate::protocol::{Audit, Behavior, Destination, Plan, Record};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub struct Case {
    pub root: tempfile::TempDir,
    pub binary: PathBuf,
    pub native: PathBuf,
    pub state: PathBuf,
}

impl Case {
    pub fn new(behavior: Behavior, records: Vec<Record>) -> Self {
        let root = tempfile::tempdir().unwrap();
        let location = root.path().canonicalize().unwrap();
        let native = location.join("native");
        let state = location.join("state");
        std::fs::create_dir(&native).unwrap();
        std::fs::create_dir(&state).unwrap();
        let profile = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let fixture = profile.join("examples").join(format!(
            "migration_client_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
        assert!(
            fixture.is_file(),
            "build the migration_client_fixture example before executable tests"
        );
        let binary = location.join(format!("private-client{}", std::env::consts::EXE_SUFFIX));
        std::fs::copy(fixture, &binary).unwrap();
        std::fs::write(
            native.join("fixture.json"),
            serde_json::to_vec(&Plan { behavior, records }).unwrap(),
        )
        .unwrap();
        Self {
            root,
            binary,
            native,
            state,
        }
    }

    pub fn arguments(&self) -> Vec<String> {
        vec![
            "--binary".into(),
            self.binary.to_str().unwrap().into(),
            "--native-runtime-root".into(),
            self.native.to_str().unwrap().into(),
            "--ready-max-wait".into(),
            "1".into(),
            "--shutdown-max-wait".into(),
            "1".into(),
            "--state-parent".into(),
            self.state.to_str().unwrap().into(),
        ]
    }

    pub fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command.current_dir(repository().join("tools/xtask"));
        command
            .args(["automation", "client-readiness"])
            .args(self.arguments());
        for key in [
            "HOME",
            "USERPROFILE",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "MESH_LLM_CONFIG",
            "MESH_LLM_RUNTIME_ROOT",
            "HF_TOKEN",
            "TASK20_AMBIENT",
        ] {
            command.env(key, "ambient-private-value");
        }
        command
    }

    pub fn run(&self) -> Output {
        self.command().output().unwrap()
    }

    pub fn audit(&self) -> Audit {
        serde_json::from_slice(&std::fs::read(self.native.join("audit.json")).unwrap()).unwrap()
    }

    pub fn assert_removed(&self) {
        assert_eq!(std::fs::read_dir(&self.state).unwrap().count(), 0);
        #[cfg(unix)]
        assert_absent(self.audit().pid);
    }
}

pub fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
}

pub fn record(stream: Destination, bytes: &[u8]) -> Record {
    Record {
        stream,
        bytes: bytes.to_vec(),
    }
}

pub fn ready() -> Record {
    record(
        Destination::Stdout,
        b"{\"extra\":42,\"role\":\"client\",\"event\":\"passive_mode\",\"status\":\"ready\"}\n",
    )
}

#[cfg(unix)]
pub fn assert_absent(pid: u32) {
    let pid = i32::try_from(pid).unwrap();
    // SAFETY: signal zero only queries this fixture-recorded positive PID.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}

pub struct Sentinel(pub std::process::Child);

impl Sentinel {
    pub fn new(case: &Case) -> Self {
        let child = Command::new(&case.binary)
            .arg("--sentinel")
            .env("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR", &case.native)
            .spawn()
            .unwrap();
        Self(child)
    }
}

impl Drop for Sentinel {
    fn drop(&mut self) {
        if let Err(error) = self.0.kill() {
            eprintln!("sentinel kill: {error}");
        }
        if let Err(error) = self.0.wait() {
            eprintln!("sentinel wait: {error}");
        }
    }
}

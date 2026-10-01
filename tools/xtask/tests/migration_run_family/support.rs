use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

const MATRIX: &[u8] = include_bytes!("../fixtures/migration/optional_replay/valid.json");

pub(super) struct Fixture {
    pub(super) root: tempfile::TempDir,
    rust_fixture: PathBuf,
    pub(super) marker: PathBuf,
    pub(super) argv: PathBuf,
    cleanup_marker: PathBuf,
    pub(super) json: PathBuf,
    pub(super) env: PathBuf,
}

impl Fixture {
    pub(super) fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let fixture_parent = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/task25-run-family/fixture-roots");
        fs::create_dir_all(&fixture_parent)?;
        let root = tempfile::Builder::new()
            .prefix("run-family ")
            .tempdir_in(fixture_parent)?;
        fs::create_dir_all(root.path().join("tools/xtask"))?;
        fs::create_dir(root.path().join("evals"))?;
        fs::write(root.path().join("Cargo.toml"), "[workspace]\n")?;
        fs::write(root.path().join("tools/xtask/Cargo.toml"), "[package]\n")?;
        fs::write(
            root.path().join("evals/agentic-replay.py"),
            "printf '%s\\n' \"$0\" \"$@\" > \"$REPLAY_ARGV\"\ntest -s \"$REPLAY_JSON\" || exit 90\ntest -s \"$REPLAY_ENV\" || exit 91\ntest \"$1\" = run || exit 92\ntest \"$2\" = --model || exit 93\ntest \"$3\" = bartowski/granite-3.1-2b-instruct-GGUF@e47b8b46c04cede00f9e19d5a846551b14b2efce/granite-3.1-2b-instruct-Q4_K_M.gguf || exit 94\nexport REPLAY_SUPERVISOR_PID=\"$PPID\"\nexec \"$REPLAY_RUST_FIXTURE\" --exact run_family_rust_child_fixture --nocapture\n",
        )?;
        fs::write(root.path().join("matrix.json"), MATRIX)?;
        Ok(Self {
            marker: root.path().join("child-started"),
            cleanup_marker: root.path().join("child-cleanup"),
            argv: root.path().join("argv.txt"),
            json: root.path().join("params.json"),
            env: root.path().join("github.env"),
            rust_fixture: std::env::current_exe()?,
            root,
        })
    }

    pub(super) fn command(&self, family: &str, timeout: Option<u64>) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        let arguments = [
            "--repo-root",
            self.root.path().to_str().unwrap_or_default(),
            "automation",
            "replay-matrix",
            "run-family",
            "--matrix",
            "matrix.json",
            "--run-family",
            family,
            "--ref",
            "main=HEAD",
            "--dataset-file",
            "data.parquet",
            "--output",
            "result",
            "--python",
            "/bin/sh",
            "--json-output",
            self.json.to_str().unwrap_or_default(),
            "--github-env",
            self.env.to_str().unwrap_or_default(),
            "--print-shell",
        ]
        .map(OsString::from);
        command
            .args(arguments)
            .current_dir(self.root.path())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", self.root.path())
            .env("TMPDIR", self.root.path())
            .env("LANG", "C.UTF-8")
            .env("REPLAY_FIXTURE_MODE", "exit")
            .env("REPLAY_FIXTURE_EXIT", "0")
            .env("REPLAY_RUST_CHILD", "1")
            .env("REPLAY_FIXTURE_MARKER", &self.marker)
            .env("REPLAY_CLEANUP_MARKER", &self.cleanup_marker)
            .env("REPLAY_ARGV", &self.argv)
            .env("REPLAY_JSON", &self.json)
            .env("REPLAY_ENV", &self.env)
            .env("REPLAY_RUST_FIXTURE", &self.rust_fixture);
        if let Some(timeout) = timeout {
            command.args(["--timeout", &timeout.to_string()]);
        }
        command
    }

    pub(super) fn run(
        &self,
        family: &str,
        mode: &str,
        child_status: i32,
        timeout: Option<u64>,
    ) -> Result<Output, std::io::Error> {
        let mut command = self.command(family, timeout);
        command
            .env("REPLAY_FIXTURE_MODE", mode)
            .env("REPLAY_FIXTURE_EXIT", child_status.to_string());
        command.output()
    }

    pub(super) fn run_without_dataset(&self) -> Result<Output, Box<dyn std::error::Error>> {
        let command = self.command("granite-3.1-2b", None);
        let tokens: Vec<_> = command.get_args().map(OsString::from).collect();
        let dataset = tokens
            .windows(2)
            .position(|pair| pair[0] == "--dataset-file")
            .ok_or("dataset option")?;
        let mut command = fixture_environment(self);
        command
            .args(&tokens[..dataset])
            .args(&tokens[dataset + 2..]);
        Ok(command.output()?)
    }

    pub(super) fn execute(
        &self,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Output, std::io::Error> {
        let mut command = fixture_environment(self);
        command.args(arguments);
        command.output()
    }
    pub(super) fn cleanup_marker_exists(&self) -> bool {
        self.cleanup_marker.is_file()
    }
}

fn fixture_environment(fixture: &Fixture) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(fixture.root.path())
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", fixture.root.path())
        .env("TMPDIR", fixture.root.path())
        .env("LANG", "C.UTF-8")
        .env("REPLAY_FIXTURE_MODE", "exit")
        .env("REPLAY_FIXTURE_EXIT", "0")
        .env("REPLAY_RUST_CHILD", "1")
        .env("REPLAY_FIXTURE_MARKER", &fixture.marker)
        .env("REPLAY_CLEANUP_MARKER", &fixture.cleanup_marker)
        .env("REPLAY_ARGV", &fixture.argv)
        .env("REPLAY_JSON", &fixture.json)
        .env("REPLAY_ENV", &fixture.env)
        .env("REPLAY_RUST_FIXTURE", &fixture.rust_fixture);
    command
}

pub(super) fn read(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(fs::read_to_string(path)?)
}

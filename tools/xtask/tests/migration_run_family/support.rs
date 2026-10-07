use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "fixture_inputs.rs"]
mod fixture_inputs;

pub(super) const SHELL_LINE: &str = "all\t3\t2\t131072\t1\t1\t100\t1\t1\t1\t2048\t1\n";
pub(super) const MODEL_URI: &str =
    "fixture/model@0123456789abcdef0123456789abcdef01234567/model.gguf";

pub(super) struct Fixture {
    pub(super) root: tempfile::TempDir,
    pub(super) marker: PathBuf,
    pub(super) argv: PathBuf,
    pub(super) json: PathBuf,
    pub(super) env: PathBuf,
    pub(super) model: PathBuf,
    pub(super) worktrees: PathBuf,
    reader: PathBuf,
    rust_fixture: PathBuf,
}

impl Fixture {
    pub(super) fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let root = tempfile::Builder::new().prefix("run-family ").tempdir()?;
        fs::create_dir_all(root.path().join("tools/xtask"))?;
        fs::create_dir_all(root.path().join("mesh/evals"))?;
        fs::write(root.path().join("Cargo.toml"), "[workspace]\n")?;
        fs::write(root.path().join("tools/xtask/Cargo.toml"), "[package]\n")?;
        fs::write(
            root.path()
                .join("mesh/evals/agentic-trajectory-manifest.py"),
            "# Retained reader path fixture; executed by a Rust adapter.\n",
        )?;
        let model = root.path().join("model.gguf");
        fixture_inputs::inputs(root.path(), &model)?;
        let worktrees = root.path().join("worktrees");
        fixture_inputs::builds(root.path(), &worktrees)?;
        let reader = root.path().join("reader-fixture");
        fixture_inputs::executable(
            &reader,
            concat!(
                "#!/bin/sh\n",
                "case \"$1\" in */mesh/evals/agentic-trajectory-manifest.py) ;; *) exit 90;; esac\n",
                "printf '%s\\n' \"$@\" > \"$REPLAY_ARGV\"\n",
                "shift\n",
                "while [ \"$#\" -gt 0 ]; do\n",
                "  if [ \"$1\" = --output ]; then export REPLAY_MANIFEST_TARGET=\"$2\"; fi\n",
                "  shift 2\n",
                "done\n",
                "exec \"$REPLAY_RUST_FIXTURE\" --exact reader::run_family_reader_fixture --nocapture\n",
            ),
        )?;
        Ok(Self {
            marker: root.path().join("reader-started.json"),
            argv: root.path().join("reader-argv.txt"),
            json: root.path().join("params.json"),
            env: root.path().join("github.env"),
            rust_fixture: std::env::current_exe()?,
            root,
            reader,
            model,
            worktrees,
        })
    }

    pub(super) fn command(&self, family: &str, timeout: Option<u64>) -> Command {
        let mut command = fixture_environment(self);
        command.args([
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
            "--model-file",
            self.model.to_str().unwrap_or_default(),
            "--output",
            "result",
            "--python",
            self.reader.to_str().unwrap_or_default(),
            "--json-output",
            self.json.to_str().unwrap_or_default(),
            "--github-env",
            self.env.to_str().unwrap_or_default(),
            "--print-shell",
        ]);
        command.args(["--timeout", &timeout.unwrap_or(30).to_string()]);
        command
    }

    pub(super) fn run(
        &self,
        family: &str,
        mode: &str,
        status: i32,
        timeout: Option<u64>,
    ) -> std::io::Result<Output> {
        self.command(family, timeout)
            .env("REPLAY_FIXTURE_MODE", mode)
            .env("REPLAY_FIXTURE_EXIT", status.to_string())
            .output()
    }

    pub(super) fn run_without_dataset(&self) -> Result<Output, Box<dyn std::error::Error>> {
        self.without("--dataset-file")
    }

    pub(super) fn without(&self, option: &str) -> Result<Output, Box<dyn std::error::Error>> {
        let mut tokens: Vec<_> = self
            .command("granite-3.1-2b", None)
            .get_args()
            .map(OsString::from)
            .collect();
        let index = tokens
            .windows(2)
            .position(|pair| pair[0] == option)
            .ok_or("missing option")?;
        tokens.drain(index..index + 2);
        Ok(self.execute(tokens)?)
    }

    pub(super) fn execute(
        &self,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> std::io::Result<Output> {
        fixture_environment(self).args(arguments).output()
    }

    pub(super) fn execute_at(
        &self,
        arguments: impl IntoIterator<Item = OsString>,
        cwd: &Path,
    ) -> std::io::Result<Output> {
        fixture_environment(self)
            .args(arguments)
            .current_dir(cwd)
            .output()
    }
}

fn fixture_environment(fixture: &Fixture) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(fixture.root.path())
        .env_clear()
        .env(
            "PATH",
            format!(
                "{}:/usr/bin:/bin",
                fixture.root.path().join("bin").display()
            ),
        )
        .env("HOME", fixture.root.path())
        .env("TMPDIR", fixture.root.path())
        .env("LANG", "C.UTF-8")
        .env("AGENTIC_REPLAY_WORKTREE_ROOT", &fixture.worktrees)
        .env("REPLAY_FIXTURE_MODE", "exit")
        .env("REPLAY_FIXTURE_EXIT", "0")
        .env("REPLAY_RUST_READER", "1")
        .env("REPLAY_FIXTURE_MARKER", &fixture.marker)
        .env("REPLAY_ARGV", &fixture.argv)
        .env("REPLAY_JSON", &fixture.json)
        .env("REPLAY_ENV", &fixture.env)
        .env("REPLAY_RUST_FIXTURE", &fixture.rust_fixture)
        .env(
            "REPLAY_MANIFEST_SOURCE",
            fixture.root.path().join("manifest.json"),
        );
    command
}

pub(super) fn read(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(fs::read_to_string(path)?)
}

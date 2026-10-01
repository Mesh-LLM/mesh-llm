use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub type TestResult = Result<(), Box<dyn Error>>;
pub const SHELL: &str = "all\t16\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t1,2,4,8\n";
pub const ROOT_ERROR: &str = "replay matrix input: root must be an object\n";
pub const SAMPLING_ERROR: &str = "replay sampling must be pinned to temperature 0 and seed 42\n";
pub const WAVES_ERROR: &str = "session count does not cover the required worker waves\n";
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/migration/optional_replay")
}

pub fn valid() -> String {
    include_str!("../fixtures/migration/optional_replay/sampling-bools.json")
        .replace("\"temperature\":false", "\"temperature\":0")
        .trim()
        .to_owned()
}

pub fn mutate(changes: &[(&str, &str)]) -> String {
    changes.iter().fold(valid(), |raw, (from, to)| {
        assert_eq!(raw.matches(from).count(), 1, "mutation {from}");
        raw.replacen(from, to, 1)
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub fn assert_outcome(label: &str, actual: &Outcome, expected: (i32, &str, &str)) {
    assert_eq!(actual.code, expected.0, "{label}: {actual:?}");
    assert_eq!(actual.stdout, expected.1.as_bytes(), "{label}: stdout");
    assert_eq!(actual.stderr, expected.2.as_bytes(), "{label}: stderr");
}

pub struct Stage(PathBuf);

impl Stage {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-task25-{}-{sequence}-{nanos}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        let stage = Self(path);
        fs::create_dir(stage.cwd())?;
        Ok(stage)
    }

    pub fn cwd(&self) -> PathBuf {
        self.0.join("unrelated")
    }

    pub fn input(&self, raw: &[u8]) -> Result<Outcome, Box<dyn Error>> {
        fs::write(self.cwd().join("matrix.json"), raw)?;
        self.run(&["--matrix", "matrix.json"])
    }

    pub fn run(&self, args: &[&str]) -> Result<Outcome, Box<dyn Error>> {
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let stdout = self.0.join(format!("{sequence}.stdout"));
        let stderr = self.0.join(format!("{sequence}.stderr"));
        let child = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "validate"])
            .args(args)
            .current_dir(self.cwd())
            .env("PATH", "")
            .stdin(Stdio::null())
            .stdout(File::create_new(&stdout)?)
            .stderr(File::create_new(&stderr)?)
            .spawn()?;
        let mut child = ReapedChild(child);
        let deadline = Instant::now() + Duration::from_secs(15);
        let status = loop {
            if let Some(status) = child.0.try_wait()? {
                break status;
            }
            if Instant::now() >= deadline {
                return Err(format!("task25 CLI deadline exceeded: {args:?}").into());
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        let code = status.code().ok_or("task25 CLI terminated by signal")?;
        assert!(fs::metadata(&stdout)?.len() < 1024 * 1024, "stdout bound");
        assert!(fs::metadata(&stderr)?.len() < 1024 * 1024, "stderr bound");
        Ok(Outcome {
            code,
            stdout: fs::read(stdout)?,
            stderr: fs::read(stderr)?,
        })
    }
}

struct ReapedChild(Child);

impl Drop for ReapedChild {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                if let Err(error) = self.0.kill() {
                    eprintln!("task25 child kill failed: {error}");
                }
                if let Err(error) = self.0.wait() {
                    eprintln!("task25 child reap failed: {error}");
                }
            }
        }
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!("task25 scratch cleanup failed: {error}");
        }
    }
}

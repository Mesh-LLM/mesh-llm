use super::export_cases::ExportCase;
use super::support::{Stage, TestResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Observation {
    pub status: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub files: BTreeMap<String, Option<Vec<u8>>>,
}

pub fn observe(
    case: &ExportCase,
    _legacy: Option<&Path>,
) -> Result<Observation, Box<dyn std::error::Error>> {
    let stage = Stage::new()?;
    let captures = Stage::new()?;
    for directory in &case.directories {
        fs::create_dir(stage.cwd().join(directory))?;
    }
    for (path, bytes) in &case.initial {
        fs::write(stage.cwd().join(path), bytes)?;
    }
    if let Some(bytes) = &case.input {
        fs::write(stage.cwd().join("matrix.json"), bytes)?;
    }
    fs::write(stage.cwd().join("ambient.env"), b"AMBIENT_UNCHANGED\n")?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.args(["automation", "replay-matrix", "export"]);
    let stdout = captures.cwd().join("stdout");
    let stderr = captures.cwd().join("stderr");
    let mut child = Reaped(
        command
            .args(&case.args)
            .current_dir(stage.cwd())
            .env("PATH", "")
            .env("GITHUB_ENV", "ambient.env")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::null())
            .stdout(File::create_new(&stdout)?)
            .stderr(File::create_new(&stderr)?)
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status.code().ok_or("export child terminated by signal")?;
        }
        if Instant::now() >= deadline {
            return Err(format!("export deadline exceeded: {}", case.id).into());
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(fs::metadata(&stdout)?.len() < 1024 * 1024);
    assert!(fs::metadata(&stderr)?.len() < 1024 * 1024);
    let mut files = BTreeMap::new();
    snapshot(&stage.cwd(), &stage.cwd(), &mut files)?;
    Ok(Observation {
        status,
        stdout: fs::read(stdout)?,
        stderr: fs::read(stderr)?,
        files,
    })
}

fn snapshot(
    root: &Path,
    directory: &Path,
    files: &mut BTreeMap<String, Option<Vec<u8>>>,
) -> TestResult {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .strip_prefix(root)?
            .to_str()
            .ok_or("non-UTF8 fixture filename")?
            .to_owned();
        if entry.file_type()?.is_dir() {
            files.insert(name, None);
            snapshot(root, &path, files)?;
        } else {
            assert!(entry.metadata()?.len() < 1024 * 1024);
            files.insert(name, Some(fs::read(path)?));
        }
    }
    Ok(())
}

struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                if let Err(error) = self.0.kill() {
                    eprintln!("export child kill failed: {error}");
                }
                if let Err(error) = self.0.wait() {
                    eprintln!("export child reap failed: {error}");
                }
            }
        }
    }
}

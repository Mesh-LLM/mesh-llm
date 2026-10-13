use serde::Serialize;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

pub const UPSTREAM: &str = "0000000000000000000000000000000000000023";
pub const FIRST_REPORT: &[u8] =
    b"{\"builders\":[{\"file\":\"src/models/a.cpp\",\"verdict\":\"transformable\"}]}\n";
pub const SECOND_REPORT: &[u8] =
    b"{\"builders\":[{\"file\":\"src/models/a.cpp\",\"verdict\":\"already_transformed\"}]}\n";

#[derive(Serialize)]
struct Invocation {
    pid: u32,
    cwd: PathBuf,
    arguments: Vec<OsString>,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var_os("PATCH_FIXTURE_ROOT").ok_or("missing fixture root")?);
    if !root.is_absolute() || root.canonicalize()? != root {
        return Err("fixture root must be canonical and absolute".into());
    }
    let cwd = std::env::current_dir()?;
    if cwd != root.join("source") {
        return Err("unexpected fixture cwd".into());
    }
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let executable = std::env::current_exe()?;
    let name = executable.file_name().ok_or("fixture executable name")?;
    let (label, report) = match name.to_str() {
        Some("git") if arguments == ["status", "--porcelain", "--untracked-files=no"] => {
            ("git-status", None)
        }
        Some("git")
            if arguments
                == [
                    "diff",
                    "--no-ext-diff",
                    "--binary",
                    "--full-index",
                    "HEAD",
                    "--",
                    "src/models",
                ] =>
        {
            fs::metadata(root.join("rewriter-second.json"))?;
            ("git-diff", None)
        }
        Some("rewriter") => {
            let first = arguments.iter().any(|argument| argument == "--apply");
            let report_name = if first {
                "report.json"
            } else {
                "report-second.json"
            };
            let mut expected = vec![
                OsString::from("--source-root"),
                cwd.clone().into_os_string(),
                "--llama-commit".into(),
                UPSTREAM.into(),
                "--report".into(),
                root.join(report_name).into_os_string(),
                "-p".into(),
                root.join("build").into_os_string(),
            ];
            if first {
                expected.push("--apply".into());
            }
            expected.extend([
                cwd.join("src/models/a.cpp").into_os_string(),
                cwd.join("src/models/b.cpp").into_os_string(),
            ]);
            if arguments != expected {
                return Err("unexpected rewriter invocation".into());
            }
            fs::metadata(root.join(if first {
                "git-status.json"
            } else {
                "rewriter-first.json"
            }))?;
            let label = if first {
                "rewriter-first"
            } else {
                "rewriter-second"
            };
            let bytes = if first { FIRST_REPORT } else { SECOND_REPORT };
            (label, Some((root.join(report_name), bytes)))
        }
        _ => return Err("unexpected fixture invocation".into()),
    };
    let invocation = Invocation {
        pid: std::process::id(),
        cwd,
        arguments,
    };
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(format!("{label}.json")))?
        .write_all(&serde_json::to_vec(&invocation)?)?;
    if let Some((path, bytes)) = report {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(bytes)?;
    }
    if label == "git-diff" {
        let path = root.join("input.diff");
        if fs::metadata(&path)?.len() > 4096 {
            return Err("fixture diff exceeds bound".into());
        }
        io::stdout().lock().write_all(&fs::read(path)?)?;
    }
    Ok(())
}

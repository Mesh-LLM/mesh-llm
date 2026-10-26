use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    behavior: Behavior,
    fail_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Behavior {
    Clean,
    Fail,
    Timeout,
    Descendant,
    Overflow,
    StderrOverflow,
    Sensitive,
    SensitiveFail,
}

#[derive(Serialize)]
struct Invocation {
    arguments: Vec<String>,
    cwd: PathBuf,
    stdin_eof: bool,
    ambient_secret_present: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [mode] if mode == "--sentinel" || mode == "--leaf" => {
            fs::write(
                format!("{}.pid", mode.trim_start_matches('-')),
                std::process::id().to_string(),
            )?;
            std::thread::sleep(Duration::from_secs(10));
            return Ok(());
        }
        [flag, _path] if flag == "-lint" => {}
        _ => return Err("fixture requires exactly -lint path".into()),
    }
    let cwd = std::env::current_dir()?;
    let plan: Plan = serde_json::from_slice(&fs::read(cwd.join("lint-plan.json"))?)?;
    let mut byte = [0];
    let stdin_eof = std::io::Read::read(&mut io::stdin(), &mut byte)? == 0;
    let invocation = Invocation {
        arguments: arguments.clone(),
        cwd: cwd.clone(),
        stdin_eof,
        ambient_secret_present: std::env::var_os("SPV_AMBIENT_SECRET").is_some(),
    };
    let mut trace = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("lint-trace.jsonl")?;
    serde_json::to_writer(&mut trace, &invocation)?;
    trace.write_all(b"\n")?;
    trace.flush()?;
    io::stdout().write_all(b"native stdout must be suppressed\n")?;
    match plan.behavior {
        Behavior::Clean => io::stderr().write_all(b"safe lint diagnostic\n")?,
        Behavior::Fail => {
            if plan.fail_path.as_deref() == arguments.get(1).map(Path::new) {
                io::stderr().write_all(b"safe lint failure\n")?;
                std::process::exit(23);
            }
        }
        Behavior::Sensitive => {
            io::stderr().write_all(b"password=synthetic-never-print\nsafe diagnostic\n")?
        }
        Behavior::SensitiveFail => {
            io::stderr().write_all(b"password=synthetic-raw\0\xff\r\nunterminated")?;
            io::stderr().flush()?;
            std::process::exit(23);
        }
        Behavior::Timeout => std::thread::sleep(Duration::from_secs(10)),
        Behavior::Overflow => io::stdout().write_all(&[b'x'; 8192])?,
        Behavior::StderrOverflow => {
            io::stderr().write_all(b"partial diagnostic must not be substituted\n")?;
            io::stderr().write_all(&[b'x'; 8192])?;
        }
        Behavior::Descendant => {
            let mut child = Command::new(std::env::current_exe()?)
                .arg("--leaf")
                .current_dir(&cwd)
                .stdin(Stdio::null())
                .spawn()?;
            std::thread::sleep(Duration::from_secs(10));
            child.wait()?;
        }
    }
    Ok(())
}

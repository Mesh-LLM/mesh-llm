#[path = "../src/cli_output.rs"]
mod cli_output;

use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, ExitCode},
    time::Duration,
};

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            match writeln!(cli_output::stderr(), "fixture: {error}") {
                Ok(()) => {}
                Err(_) => return ExitCode::from(125),
            }
            ExitCode::from(124)
        }
    }
}

fn run() -> Result<u8, Box<dyn std::error::Error>> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [verb, sentinel] if verb == "--descendant" => {
            std::thread::sleep(Duration::from_secs(2));
            fs::write(sentinel, b"descendant survived")?;
            Ok(0)
        }
        [verb, binary] if verb == "-archs" => inspect(Path::new(binary)),
        _ => Ok(123),
    }
}

fn inspect(binary: &Path) -> Result<u8, Box<dyn std::error::Error>> {
    let bytes = fs::read(binary)?;
    let mut stdout = cli_output::stdout();
    let mut stderr = cli_output::stderr();
    match bytes.as_slice() {
        b"!empty" => {}
        b"!nonzero" => {
            stderr.write_all(b"private-error-sentinel\n")?;
            return Ok(7);
        }
        b"!raw" => {
            stdout.write_all(b"token\x1cpassword\x1dsecret\x1eauthorization\x1finvite\r\n")?
        }
        b"!overflow" => stdout.write_all(&vec![b'x'; 8192])?,
        b"!tree" => {
            let sentinel =
                std::env::var_os("XCFRAMEWORK_SENTINEL").ok_or("missing fixture sentinel")?;
            let ready = std::env::var_os("XCFRAMEWORK_READY").ok_or("missing fixture readiness")?;
            let mut child = Command::new(std::env::current_exe()?)
                .arg("--descendant")
                .arg(sentinel)
                .spawn()?;
            fs::write(ready, child.id().to_string())?;
            child.wait()?;
            std::thread::park();
        }
        _ => stdout.write_all(&bytes)?,
    }
    stdout.flush()?;
    stderr.flush()?;
    Ok(0)
}

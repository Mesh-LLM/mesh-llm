use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn main() -> io::Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let cwd = std::env::current_dir()?;
    if args.as_slice() == [std::ffi::OsString::from("sentinel")] {
        return hold(&cwd);
    }
    if args.as_slice() == [std::ffi::OsString::from("descendant")] {
        std::fs::write(cwd.join("descendant.pid"), std::process::id().to_string())?;
        return hold(&cwd);
    }
    let [package, checksum, artifact] = args.as_slice() else {
        return Err(io::Error::other("expected exact Swift argv"));
    };
    if package != "package" || checksum != "compute-checksum" {
        return Err(io::Error::other("wrong Swift subcommand"));
    }
    let mut receipt = Vec::new();
    for argument in &args {
        receipt.extend_from_slice(argument.as_encoded_bytes());
        receipt.push(0);
    }
    std::fs::write(cwd.join("argv.bin"), receipt)?;
    std::fs::write(cwd.join("cwd.bin"), cwd.as_os_str().as_encoded_bytes())?;
    let mode = Path::new(artifact)
        .file_name()
        .and_then(|name| name.to_str());
    match mode {
        Some("empty") => (),
        Some("crlf") => io::stdout().write_all(b"opaque\r\n\n")?,
        Some("invalid") => io::stdout().write_all(b"\xff\n")?,
        Some("overflow") => io::stdout().write_all(&[b'x'; 4096])?,
        Some("failure") => {
            io::stderr().write_all(b"fixture checksum failure\n")?;
            std::process::exit(7);
        }
        Some("tree") => {
            let mut child = Command::new(std::env::current_exe()?)
                .arg("descendant")
                .current_dir(&cwd)
                .stdin(Stdio::null())
                .spawn()?;
            wait_for(&cwd.join("descendant.pid"))?;
            std::fs::write(cwd.join("ready"), b"ready")?;
            hold(&cwd)?;
            child.kill()?;
            child.wait()?;
        }
        Some(_) | None => io::stdout().write_all(b"opaque-checksum\n\n")?,
    }
    Ok(())
}

fn hold(root: &Path) -> io::Result<()> {
    let until = Instant::now() + Duration::from_secs(10);
    while !root.join("release").is_file() {
        if Instant::now() >= until {
            return Err(io::Error::other("fixture hold deadline"));
        }
        std::thread::park_timeout(Duration::from_millis(2));
    }
    Ok(())
}

fn wait_for(path: &Path) -> io::Result<()> {
    let until = Instant::now() + Duration::from_secs(2);
    while !path.is_file() {
        if Instant::now() >= until {
            return Err(io::Error::other("fixture readiness deadline"));
        }
        std::thread::park_timeout(Duration::from_millis(2));
    }
    Ok(())
}

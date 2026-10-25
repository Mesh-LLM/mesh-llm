use super::fixture;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

pub(super) fn leader(root: &Path, mode: &str) -> io::Result<()> {
    let mut child = fixture::spawn("cleanup-leaf")?;
    wait_until(|| root.join("armed").is_file())?;
    match mode {
        "cleanup-leader" => std::process::exit(0),
        "cleanup-ready" => {
            fixture::install_stop()?;
            io::stdout().write_all(b"READY\n")?;
            io::stdout().flush()?;
            wait_until(fixture::stopped)?;
            child.wait()?;
        }
        "cleanup-timeout" | "cleanup-cancel" => {
            child.wait()?;
        }
        _ => return Err(io::Error::other("invalid readiness fixture")),
    }
    Ok(())
}

pub(super) fn leaf(root: &Path) -> io::Result<()> {
    fixture::install_stop()?;
    std::fs::write(root.join("armed"), b"armed")?;
    wait_until(fixture::stopped)?;
    std::fs::write(root.join("shutdown-observed"), b"stopped")?;
    io::stdout().write_all(b"READY\n")?;
    io::stdout().flush()
}

fn wait_until(mut condition: impl FnMut() -> bool) -> io::Result<()> {
    let until = Instant::now() + Duration::from_secs(5);
    while !condition() {
        if Instant::now() >= until {
            return Err(io::Error::other("readiness fixture handshake timeout"));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}

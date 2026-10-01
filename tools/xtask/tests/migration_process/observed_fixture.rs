use super::fixture;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

pub(super) fn run(root: &Path, mode: &str) -> io::Result<()> {
    fixture::install_stop()?;
    for _ in 0..32 {
        io::stdout().write_all(b"harmless output\n")?;
        io::stderr().write_all(b"harmless error\n")?;
    }
    if mode != "observed-timeout" {
        io::stderr().write_all(b"READY\r\n")?;
        io::stderr().flush()?;
    }
    let until = Instant::now() + Duration::from_secs(5);
    while mode == "observed-stubborn" || !fixture::stopped() {
        if Instant::now() >= until {
            return Err(io::Error::other("observed fixture stop deadline"));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    std::fs::write(root.join("observed-stop"), b"handler-observed")?;
    io::stderr().write_all(b"READY\n")?;
    if mode == "observed-nonzero" {
        std::process::exit(23);
    }
    Ok(())
}

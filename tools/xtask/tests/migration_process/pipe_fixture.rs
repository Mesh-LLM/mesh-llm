use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};

pub(super) fn run(root: &Path, mode: &str) -> io::Result<()> {
    match mode {
        "buffered-eof" => io::stdout().write_all(&[b'\n'; 32768])?,
        "capture-fast" => (),
        "capture-held" => {
            std::fs::write(root.join("holding"), b"holding")?;
            let until = Instant::now() + Duration::from_secs(5);
            while !root.join("release").is_file() {
                if Instant::now() >= until {
                    return Err(io::Error::other("held capture fixture deadline"));
                }
                std::thread::park_timeout(Duration::from_millis(1));
            }
        }
        _ => return Err(io::Error::other("invalid pipe fixture")),
    }
    io::stdout().write_all(b"READY")?;
    io::stdout().flush()?;
    std::process::exit(0)
}

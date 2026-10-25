use super::fixture;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant};
pub(super) fn run(root: &Path) -> io::Result<()> {
    fixture::install_stop()?;
    io::stdout().write_all(b"READY\n")?;
    io::stdout().flush()?;
    let until = Instant::now() + Duration::from_secs(5);
    while !fixture::stopped() {
        if root.join("crash-now").is_file() {
            std::process::exit(23);
        }
        if let Some(barrier) = std::env::var_os("RETAINED_SURVIVOR_BARRIER")
            && Path::new(&barrier).is_file()
        {
            io::stdout().write_all(b"SURVIVOR_WINDOW\n")?;
            io::stdout().flush()?;
            wait_file(&root.join("window-observed"), until)?;
            std::process::exit(23);
        }
        if Instant::now() >= until {
            return Err(io::Error::other("retained fixture deadline"));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    if std::env::var_os("RETAINED_STOP_BARRIER").is_some() {
        std::fs::write(root.join("stop-entered"), b"stop")?;
        io::stdout().write_all(b"TARGET_STOP\n")?;
        io::stdout().flush()?;
        wait_file(&root.join("stop-release"), until)?;
    }
    Ok(())
}

fn wait_file(path: &Path, until: Instant) -> io::Result<()> {
    while !path.is_file() {
        if Instant::now() >= until {
            return Err(io::Error::other("retained barrier deadline"));
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
    Ok(())
}

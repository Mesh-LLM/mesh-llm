use std::io::{self, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

#[cfg(windows)]
unsafe extern "system" fn stop(_: u32) -> i32 {
    STOP.store(true, Ordering::SeqCst);
    1
}

pub fn run() -> io::Result<()> {
    let mode = std::env::var("MIGRATION_PROCESS_MODE").map_err(io::Error::other)?;
    let root = std::env::var_os("MIGRATION_PROCESS_ROOT")
        .ok_or_else(|| io::Error::other("fixture root missing"))?;
    let root = std::path::Path::new(&root);
    std::fs::write(
        root.join(format!("{mode}.pid")),
        std::process::id().to_string(),
    )?;
    match mode.as_str() {
        "retained-crash" => return super::retained_fixture::run(root),
        "observed-ready" | "observed-nonzero" | "observed-stubborn" | "observed-timeout" => {
            return super::observed_fixture::run(root, &mode);
        }
        "buffered-eof" | "capture-fast" | "capture-held" => {
            return super::pipe_fixture::run(root, &mode);
        }
        "cleanup-leader" | "cleanup-timeout" | "cleanup-ready" | "cleanup-cancel" => {
            return super::readiness_fixture::leader(root, &mode);
        }
        "cleanup-leaf" => return super::readiness_fixture::leaf(root),
        "exit" => {
            io::stdout().write_all(b"READY\n")?;
            io::stderr().write_all(b"diagnostic\n")?;
            return Ok(());
        }
        "signal-exit" => {
            #[cfg(unix)]
            {
                // SAFETY: SIGKILL is directed at this disposable fixture only.
                unsafe {
                    libc::raise(libc::SIGKILL);
                }
            }
            #[cfg(windows)]
            std::process::exit(23);
        }
        "crash" => {
            io::stderr().write_all(b"early failure\n")?;
            std::process::exit(23);
        }
        "args" => {
            let args: Vec<_> = std::env::args_os().collect();
            std::fs::write(
                root.join("args.json"),
                serde_json::to_vec(
                    &args
                        .iter()
                        .map(|arg| arg.to_string_lossy())
                        .collect::<Vec<_>>(),
                )?,
            )?;
            std::fs::write(
                root.join("cwd"),
                std::env::current_dir()?.as_os_str().as_encoded_bytes(),
            )?;
            std::fs::write(
                root.join("inherited-home"),
                if std::env::var_os("HOME").is_some() {
                    "present"
                } else {
                    "absent"
                },
            )?;
            io::stdout().write_all(
                std::env::var("PAYLOAD")
                    .map_err(io::Error::other)?
                    .as_bytes(),
            )?;
            return Ok(());
        }
        "bytes" => {
            io::stdout().write_all(b"bad\xff\xfe\n")?;
            io::stderr().write_all(b"err\x80\n")?;
            return Ok(());
        }
        "secrets" => {
            let secret = std::env::var("CREDENTIAL").map_err(io::Error::other)?;
            for byte in secret.as_bytes() {
                io::stdout().write_all(&[*byte])?;
                io::stdout().flush()?;
            }
            io::stdout().write_all(b"\ninvite_token=fixture-private\n")?;
            io::stderr().write_all(secret.as_bytes())?;
            return Ok(());
        }
        "flood" => {
            let chunk = [b'x'; 4096];
            for _ in 0..1024 {
                io::stdout().write_all(&chunk)?;
                io::stderr().write_all(&chunk)?;
            }
            for _ in 0..256 {
                io::stdout().write_all(b"bounded output line\n")?;
                io::stderr().write_all(b"bounded error line\n")?;
            }
            io::stdout().write_all(b"\nREADY\n")?;
            return Ok(());
        }
        "forever-flood" => loop {
            io::stdout().write_all(&[b'x'; 4096])?;
            io::stderr().write_all(&[b'y'; 4096])?;
        },
        "tree-exit" | "tree-crash" | "tree-hang" | "tree-graceful" => {
            let mut child = spawn(if mode == "tree-graceful" {
                "branch-graceful"
            } else {
                "branch"
            })?;
            let ready = root.join("leaf.ready");
            wait_file(&ready)?;
            io::stdout().write_all(b"TREE_READY\n")?;
            io::stdout().flush()?;
            if mode == "tree-exit" {
                std::process::exit(0);
            }
            if mode == "tree-crash" {
                std::process::exit(23);
            }
            if mode == "tree-graceful" {
                std::fs::write(root.join("leaf.stop"), b"stop")?;
                child.wait()?;
                return Ok(());
            }
            stubborn();
            child.wait()?;
        }
        "branch" | "branch-graceful" => {
            let mut child = spawn("leaf")?;
            if mode == "branch" {
                stubborn();
            }
            child.wait()?;
        }
        "leaf" => {
            ignore_termination()?;
            std::fs::write(root.join("leaf.ready"), b"ready")?;
            wait_file(&root.join("leaf.stop"))?;
        }
        "ready-hang" => {
            install_stop()?;
            io::stdout().write_all(b"READY\n")?;
            io::stdout().flush()?;
            while !STOP.load(Ordering::SeqCst) {
                thread::park_timeout(Duration::from_millis(5));
            }
        }
        "hang" | "sentinel" => {
            ignore_termination()?;
            stubborn();
        }
        _ => return Err(io::Error::other("unknown fixture mode")),
    }
    Ok(())
}

pub fn spawn(mode: &str) -> io::Result<std::process::Child> {
    let mut command = Command::new(std::env::current_exe()?);
    if std::env::var_os("MIGRATION_PROCESS_DRIVER").is_none() {
        command.args(["--exact", "migration_process_fixture", "--nocapture"]);
    }
    command
        .env("MIGRATION_PROCESS_MODE", mode)
        .stdin(Stdio::null())
        .spawn()
}

fn stubborn() {
    loop {
        thread::park_timeout(Duration::from_secs(60));
    }
}

fn wait_file(path: &std::path::Path) -> io::Result<()> {
    let until = std::time::Instant::now() + Duration::from_secs(5);
    while !path.is_file() {
        if std::time::Instant::now() >= until {
            return Err(io::Error::other("fixture readiness timeout"));
        }
        thread::park_timeout(Duration::from_millis(5));
    }
    Ok(())
}

#[cfg(unix)]
pub(super) fn install_stop() -> io::Result<()> {
    // SAFETY: handler only stores to a lock-free atomic and has C signal ABI.
    if unsafe { libc::signal(libc::SIGTERM, stop as *const () as libc::sighandler_t) }
        == libc::SIG_ERR
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(unix)]
fn ignore_termination() -> io::Result<()> {
    // SAFETY: SIG_IGN is the platform's valid signal-disposition constant.
    if unsafe { libc::signal(libc::SIGTERM, libc::SIG_IGN) } == libc::SIG_ERR {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
pub(super) fn install_stop() -> io::Result<()> {
    // SAFETY: static handler obeys the console ABI and only stores an atomic.
    if unsafe { windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(stop), 1) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn ignore_termination() -> io::Result<()> {
    install_stop()
}

pub(super) fn stopped() -> bool {
    STOP.load(Ordering::SeqCst)
}

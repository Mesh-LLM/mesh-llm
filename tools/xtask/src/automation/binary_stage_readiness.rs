//! Socket readiness for a caller-owned binary stage. Never signals or owns the server.
use crate::{command::DynResult, process::Cancellation, repository::check_args::Grammar};
use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};
use tokio::time::Instant;
const GRAMMAR: Grammar = Grammar {
    usage: "automation binary-stage-readiness --host IP --port PORT --server-pid PID --timeout-secs SECONDS",
    values: &["--host", "--port", "--server-pid", "--timeout-secs"],
    flags: &["--help"],
};
struct Options {
    address: SocketAddr,
    pid: i32,
    timeout: Duration,
}
impl Options {
    fn parse(args: &[String]) -> DynResult<Self> {
        let parsed = GRAMMAR
            .parse(args)
            .map_err(|_| "invalid binary-stage readiness arguments")?;
        if !parsed.positionals.is_empty() || parsed.flag("--help") {
            return Err(GRAMMAR.usage.into());
        }
        let one = |key| -> DynResult<&str> {
            let values = parsed.all(key);
            if values.len() != 1 {
                return Err(format!("exactly one {key} is required").into());
            }
            Ok(values[0])
        };
        let host: IpAddr = one("--host")?.parse()?;
        let port: u16 = one("--port")?.parse()?;
        let pid: i32 = one("--server-pid")?.parse()?;
        let seconds: u64 = one("--timeout-secs")?.parse()?;
        if port == 0 || pid <= 1 || !(1..=3600).contains(&seconds) {
            return Err(
                "port must be positive and PID must exceed 1; timeout must be 1..=3600 seconds"
                    .into(),
            );
        }
        Ok(Self {
            address: SocketAddr::new(host, port),
            pid,
            timeout: Duration::from_secs(seconds),
        })
    }
}
#[cfg(unix)]
fn server_alive(pid: i32) -> DynResult<bool> {
    // SAFETY: signal zero observes existence/permission only. Positive PID validation
    // prevents process-group selection. The server remains wholly caller-owned.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(error.into()),
    }
}
#[cfg(not(unix))]
fn server_alive(_: i32) -> DynResult<bool> {
    Err("binary-stage PID readiness requires Unix".into())
}
fn check(options: &Options, deadline: Instant, cancellation: &Cancellation) -> DynResult<()> {
    if cancellation.is_cancelled() {
        return Err("binary-stage readiness cancelled".into());
    }
    if !server_alive(options.pid)? {
        return Err("binary-stage server exited before readiness".into());
    }
    if Instant::now() >= deadline {
        return Err("binary-stage readiness deadline exceeded".into());
    }
    Ok(())
}
async fn connect(
    options: &Options,
    deadline: Instant,
    cancellation: &Cancellation,
) -> DynResult<bool> {
    let attempt_end = deadline.min(Instant::now() + Duration::from_secs(1));
    let connection = tokio::net::TcpStream::connect(options.address);
    tokio::pin!(connection);
    loop {
        check(options, deadline, cancellation)?;
        tokio::select! {
            result = &mut connection => {
                check(options, deadline, cancellation)?;
                return Ok(result.is_ok());
            }
            () = tokio::time::sleep_until(attempt_end) => return Ok(false),
            () = tokio::time::sleep(Duration::from_millis(100)) => (),
        }
    }
}
async fn wait(options: &Options, cancellation: &Cancellation) -> DynResult<()> {
    let deadline = Instant::now() + options.timeout;
    loop {
        check(options, deadline, cancellation)?;
        if connect(options, deadline, cancellation).await? {
            return Ok(());
        }
        let retry_at = deadline.min(Instant::now() + Duration::from_secs(1));
        while Instant::now() < retry_at {
            check(options, deadline, cancellation)?;
            tokio::time::sleep_until(retry_at.min(Instant::now() + Duration::from_millis(100)))
                .await;
        }
    }
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        use std::io::Write as _;
        writeln!(crate::cli_output::stdout(), "{}", GRAMMAR.usage)?;
        return Ok(());
    }
    let options = Options::parse(args)?;
    let interrupt = crate::command_interrupt::Interrupt::install()?;
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(wait(&options, &interrupt.cancellation()));
    interrupt.finish()?;
    result
}
#[cfg(test)]
#[path = "binary_stage_readiness_tests.rs"]
mod tests;

//! Supervision of a composed standalone CLI serving one model over loopback:
//! spawn with a log file, wait for `/v1/models` readiness, and always reap the
//! child. Also the bounded capture runner for short CLI probes.

use super::http;
use crate::process::{self, Cancellation, Completion, Limits, RawCaptureOptions, Readiness, Value};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A free loopback port, released before the child binds it.
pub(crate) fn free_port() -> Result<u16, String> {
    let listener =
        std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    Ok(listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port())
}

/// Run a short-lived CLI probe, requiring success and returning stdout.
pub(crate) fn run_capture(
    binary: &Path,
    arguments: &[String],
    execution: Duration,
) -> Result<String, String> {
    let spec = process::ProcessSpec {
        executable: binary
            .canonicalize()
            .map_err(|error| format!("{}: {error}", binary.display()))?,
        arguments: arguments
            .iter()
            .map(|value| Value::Public(value.clone().into()))
            .collect(),
        cwd: std::env::current_dir().map_err(|error| error.to_string())?,
        environment: ["PATH", "SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect(),
    };
    let limits = Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 1_048_576,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(1_048_576),
            stderr: NonZeroUsize::new(65_536),
        },
    )
    .map_err(|error| format!("standalone CLI probe failed: {error:?}"))?;
    if !report.process.success() {
        return Err(format!("standalone CLI probe failed: {:?}", report.process));
    }
    let bytes = report.stdout.ok_or("standalone CLI probe output missing")?;
    Ok(String::from_utf8_lossy(bytes.as_bytes()).into_owned())
}

/// A running `skippy serve` child bound to a loopback port.
pub(crate) struct Server {
    child: std::process::Child,
    pub(crate) base: String,
    pub(crate) log_path: PathBuf,
}

impl Server {
    /// Start the composed CLI and wait until `/v1/models` lists the model.
    pub(crate) fn start(
        binary: &Path,
        arguments: &[String],
        model_id: &str,
        log_path: &Path,
    ) -> Result<Self, String> {
        let log = std::fs::File::create(log_path)
            .map_err(|error| format!("{}: {error}", log_path.display()))?;
        let error_log = log.try_clone().map_err(|error| error.to_string())?;
        let child = std::process::Command::new(binary)
            .args(arguments)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(error_log))
            .spawn()
            .map_err(|error| format!("cannot start the composed CLI: {error}"))?;
        let mut server = Self {
            child,
            base: String::new(),
            log_path: log_path.to_path_buf(),
        };
        let base = arguments
            .iter()
            .skip_while(|arg| arg.as_str() != "--bind-addr")
            .nth(1)
            .map(|address| format!("http://{address}"))
            .ok_or("composed CLI arguments lack --bind-addr")?;
        server.base = base.clone();
        let deadline = Instant::now() + Duration::from_secs(240);
        loop {
            if let Some(status) = server.child.try_wait().map_err(|error| error.to_string())? {
                server.stop();
                return Err(format!(
                    "standalone CLI exited during startup ({status}): {}",
                    server.tail()
                ));
            }
            if Instant::now() >= deadline {
                server.stop();
                return Err(format!(
                    "standalone CLI did not become ready: {}",
                    server.tail()
                ));
            }
            if let Ok((200, models)) = http::request_json(
                &format!("{base}/v1/models"),
                None,
                "GET",
                Duration::from_secs(10),
            ) && models
                .get("data")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| {
                    items.iter().any(|item| {
                        item.get("id").and_then(serde_json::Value::as_str) == Some(model_id)
                    })
                })
            {
                return Ok(server);
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    pub(crate) fn tail(&self) -> String {
        let bytes = std::fs::read(&self.log_path).unwrap_or_default();
        let text = String::from_utf8_lossy(&bytes);
        format!("server log: {}", &text[text.len().saturating_sub(4000)..])
    }

    pub(crate) fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The device token the running product actually selected, from the runtime's
/// own `backend_device_selected` log event.
pub(crate) fn selected_backend_device(log_path: &Path) -> Option<String> {
    let bytes = std::fs::read(log_path).ok()?;
    for raw in String::from_utf8_lossy(&bytes).lines() {
        let line = raw.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if event.get("type").and_then(serde_json::Value::as_str) != Some("backend_device_selected")
        {
            continue;
        }
        let detail = match event.get("data") {
            Some(serde_json::Value::Object(_)) => event
                .get("data")
                .and_then(|data| data.get("detail"))
                .cloned(),
            other => other.cloned(),
        };
        if let Some(detail) = detail
            .and_then(|detail| detail.as_str().map(str::to_owned))
            .map(|detail| detail.trim().to_owned())
            .filter(|detail| !detail.is_empty())
        {
            return Some(detail);
        }
    }
    None
}

/// Wait for the runtime's own device event, which can land after the first
/// successful readiness poll on slower platforms.
pub(crate) fn wait_for_backend_device(log_path: &Path, timeout: Duration) -> Option<String> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(device) = selected_backend_device(log_path) {
            return Some(device);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_event_parses_object_and_string_data() {
        let temp = tempfile::tempdir().unwrap();
        let log = temp.path().join("serve.log");
        std::fs::write(
            &log,
            "not json\n{\"type\":\"other\",\"data\":{\"detail\":\"X\"}}\n{\"type\":\"backend_device_selected\",\"data\":{\"detail\":\"MTL0\"}}\n",
        )
        .unwrap();
        assert_eq!(
            super::selected_backend_device(&log).as_deref(),
            Some("MTL0")
        );
        std::fs::write(
            &log,
            "{\"type\":\"backend_device_selected\",\"data\":\"CUDA0\"}\n",
        )
        .unwrap();
        assert_eq!(
            super::selected_backend_device(&log).as_deref(),
            Some("CUDA0")
        );
        assert!(super::selected_backend_device(&temp.path().join("absent")).is_none());
    }
}

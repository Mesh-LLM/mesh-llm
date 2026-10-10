//! Bounded HTTP(S) component execution retaining upstream status, headers and bytes.
use super::request_projection::{BODY_LIMIT, forward_request_header, forward_response_header};
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use http_body_util::Full;
use hyper::{HeaderMap, Method, Response, body::Bytes};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::PathBuf, time::Duration};

#[derive(Clone)]
pub(super) struct Forwarder {
    executable: PathBuf,
    #[cfg(test)]
    fixture_ca: Option<PathBuf>,
    #[cfg(test)]
    local_fixture: bool,
}

#[derive(Deserialize)]
struct Metadata {
    status: u16,
    headers: BTreeMap<String, Vec<String>>,
}

impl Forwarder {
    pub(super) fn discover() -> Result<Self, String> {
        let executable =
            std::env::split_paths(&std::env::var_os("PATH").ok_or("curl PATH unavailable")?)
                .map(|directory| directory.join(if cfg!(windows) { "curl.exe" } else { "curl" }))
                .find(|path| path.is_file())
                .ok_or("recording proxy requires curl 8.4 or newer")?;
        let executable = if executable.is_absolute() {
            executable
        } else {
            std::env::current_dir()
                .map_err(|_| "working directory unavailable")?
                .join(executable)
        };
        let selected = Self {
            executable,
            #[cfg(test)]
            fixture_ca: None,
            #[cfg(test)]
            local_fixture: false,
        };
        let temporary = tempfile::tempdir().map_err(|_| "curl probe directory unavailable")?;
        let result = selected.execute(
            vec![
                Value::Public("--disable".into()),
                Value::Public("--version".into()),
            ],
            temporary.path(),
            Duration::from_secs(5),
            &Cancellation::default(),
        )?;
        let line = std::str::from_utf8(&result).map_err(|_| "curl version output invalid")?;
        let version = line
            .lines()
            .next()
            .and_then(|line| line.strip_prefix("curl "))
            .and_then(|line| line.split_whitespace().next())
            .ok_or("curl version absent")?;
        let parts = version
            .split('.')
            .take(2)
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "curl version invalid")?;
        if parts.len() != 2
            || (parts[0], parts[1]) < (8, 4)
            || !line.lines().any(|line| {
                line.starts_with("Protocols:")
                    && line.split_whitespace().any(|word| word == "https")
            })
        {
            return Err("recording proxy requires curl 8.4 or newer with HTTPS support".into());
        }
        Ok(selected)
    }

    #[cfg(test)]
    pub(super) fn fixture(ca: Option<&std::path::Path>) -> Self {
        let mut selected = Self::discover().unwrap();
        selected.fixture_ca = ca.map(std::path::Path::to_owned);
        selected.local_fixture = true;
        selected
    }

    pub(super) fn forward(
        &self,
        endpoint: String,
        method: Method,
        headers: HeaderMap,
        body: Vec<u8>,
        cancellation: Cancellation,
    ) -> Result<Response<Full<Bytes>>, String> {
        let temporary = tempfile::tempdir().map_err(|_| "forwarding directory unavailable")?;
        let root = temporary.path();
        let output = root.join("response");
        let mut arguments: Vec<Value> = [
            "--disable",
            "--silent",
            "--show-error",
            "--http1.1",
            "--path-as-is",
            "--location",
            "--max-redirs",
            "5",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--max-filesize",
            "33554432",
            "--max-time",
            "360",
            "--connect-timeout",
            "30",
        ]
        .into_iter()
        .map(|value| Value::Public(value.into()))
        .collect();
        arguments.extend([
            Value::Public("--output".into()),
            Value::Public(output.as_os_str().into()),
            Value::Public("--write-out".into()),
            Value::Public(r#"{"status":%{http_code},"headers":%{header_json}}"#.into()),
        ]);
        for (name, value) in &headers {
            if forward_request_header(name.as_str()) && !connection_header(&headers, name.as_str())
            {
                let value = value
                    .to_str()
                    .map_err(|_| "request header is not textual")?;
                arguments.extend([
                    Value::Public("--header".into()),
                    Value::Secret(format!("{name}: {value}").into()),
                ]);
            }
        }
        if method == Method::POST {
            let payload = root.join("request");
            fs::write(&payload, body).map_err(|_| "request body unavailable")?;
            let mut file = std::ffi::OsString::from("@");
            file.push(payload);
            arguments.extend([Value::Public("--data-binary".into()), Value::Public(file)]);
        } else {
            arguments.extend([
                Value::Public("--request".into()),
                Value::Public("GET".into()),
            ]);
        }
        arguments.extend([
            Value::Public("--url".into()),
            Value::Public(endpoint.into()),
        ]);
        let bytes = self.execute(arguments, root, Duration::from_secs(360), &cancellation)?;
        let metadata: Metadata =
            serde_json::from_slice(&bytes).map_err(|_| "upstream response metadata invalid")?;
        if fs::metadata(&output)
            .map_err(|_| "upstream response unavailable")?
            .len()
            > BODY_LIMIT as u64
        {
            return Err("upstream response exceeds 32 MiB".into());
        }
        let body = fs::read(output).map_err(|_| "upstream response unreadable")?;
        let mut response = Response::builder()
            .status(metadata.status)
            .body(Full::new(Bytes::from(body)))
            .map_err(|_| "upstream status invalid")?;
        let headers = response_headers(metadata.headers)?;
        *response.headers_mut() = headers;
        Ok(response)
    }

    fn execute(
        &self,
        arguments: Vec<Value>,
        root: &std::path::Path,
        duration: Duration,
        cancellation: &Cancellation,
    ) -> Result<Vec<u8>, String> {
        let environment: BTreeMap<_, _> = [
            "PATH",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "CURL_CA_BUNDLE",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "https_proxy",
            "http_proxy",
            "all_proxy",
            "no_proxy",
        ]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Secret(value))))
        .collect();
        #[cfg(test)]
        let environment = {
            let mut environment = environment;
            if self.local_fixture {
                environment.insert("NO_PROXY".into(), Value::Public("*".into()));
                environment.insert("no_proxy".into(), Value::Public("*".into()));
            }
            if let Some(ca) = &self.fixture_ca {
                environment.insert(
                    "CURL_CA_BUNDLE".into(),
                    Value::Public(ca.as_os_str().into()),
                );
            }
            environment
        };
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.executable.clone(),
                arguments,
                cwd: root.to_owned(),
                environment,
            },
            &Limits {
                execution: duration,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 1024 * 1024,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancellation,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(1024 * 1024),
                stderr: NonZeroUsize::new(4096),
            },
        )
        .map_err(|_| "upstream transfer supervisor failed")?;
        if report.process.outcome != Outcome::Exited
            || !report.process.success()
            || !report.process.cleanup.complete
        {
            return Err("upstream transfer failed or was cancelled".into());
        }
        Ok(report
            .stdout
            .ok_or("upstream metadata absent")?
            .as_bytes()
            .to_vec())
    }
}

pub(super) fn connection_header(headers: &HeaderMap, name: &str) -> bool {
    headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|value| value.trim().eq_ignore_ascii_case(name))
}

fn response_headers(values: BTreeMap<String, Vec<String>>) -> Result<HeaderMap, String> {
    let mut raw = HeaderMap::new();
    for (name, values) in values {
        let name = hyper::header::HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| "upstream header name invalid")?;
        for value in values {
            raw.append(
                name.clone(),
                value.parse().map_err(|_| "upstream header value invalid")?,
            );
        }
    }
    let mut forwarded = HeaderMap::new();
    for (name, value) in &raw {
        // Preserve content encoding because forwarding retains raw response bytes.
        if (forward_response_header(name.as_str()) || name == "content-encoding")
            && !connection_header(&raw, name.as_str())
        {
            forwarded.append(name.clone(), value.clone());
        }
    }
    Ok(forwarded)
}

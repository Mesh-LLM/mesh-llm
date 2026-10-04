use super::Request;
use crate::process::{
    Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    supervise_raw,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone)]
pub(in super::super) struct Curl {
    executable: PathBuf,
    environment: Vec<(OsString, OsString)>,
    certificate_bundle: Option<PathBuf>,
}

impl Curl {
    #[cfg(test)]
    pub(super) fn fixture(ca: &Path) -> Self {
        let mut curl = Self::discover().unwrap();
        curl.environment.retain(|(key, _)| {
            matches!(
                key.to_str(),
                Some("PATH" | "SystemRoot" | "SYSTEMROOT" | "WINDIR" | "TEMP" | "TMP")
            )
        });
        curl.environment.extend([
            ("CURL_CA_BUNDLE".into(), ca.as_os_str().into()),
            ("NO_PROXY".into(), "*".into()),
        ]);
        curl.certificate_bundle = Some(ca.to_path_buf());
        curl
    }
    pub(in super::super) fn discover() -> Result<Self, String> {
        let executable = find()
            .ok_or("stability HTTPS requires existing curl 8.4 or newer with HTTPS support")?;
        let cwd = std::env::current_dir().map_err(|_| "HTTPS working directory unavailable")?;
        let mut environment = environment();
        normalize_trust_paths(&mut environment, &cwd);
        let certificate_bundle = selected_certificate_bundle(&environment);
        let curl = Self {
            executable,
            environment,
            certificate_bundle,
        };
        let report = supervise_raw(
            &ProcessSpec {
                executable: curl.executable.clone(),
                arguments: vec![
                    Value::Public("--disable".into()),
                    Value::Public("--version".into()),
                ],
                cwd,
                environment: curl.child_environment(),
            },
            &limits(Duration::from_secs(5)),
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(4096),
            },
        )
        .map_err(|_| "stability curl capability probe failed")?;
        let bytes = report
            .stdout
            .as_ref()
            .ok_or("curl version output absent")?
            .as_bytes();
        if !report.process.success() || !supported(bytes) {
            return Err(
                "stability HTTPS requires existing curl 8.4 or newer with HTTPS support".into(),
            );
        }
        Ok(curl)
    }

    pub(super) fn specification(
        &self,
        request: &Request,
        files: Files<'_>,
        budget: Duration,
    ) -> ProcessSpec {
        let mut arguments: Vec<Value> = [
            "--disable",
            "--silent",
            "--show-error",
            "--no-buffer",
            "--http1.1",
            "--suppress-connect-headers",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--max-filesize",
            "16777216",
            "--request",
            request.method.as_str(),
        ]
        .into_iter()
        .map(|value| Value::Public(value.into()))
        .collect();
        if let Some(bundle) = &self.certificate_bundle {
            arguments.extend([
                Value::Public("--cacert".into()),
                Value::Public(bundle.as_os_str().into()),
            ]);
        }
        for (flag, value) in [
            (
                "--max-time",
                OsString::from(budget.as_secs_f64().to_string()),
            ),
            (
                "--connect-timeout",
                OsString::from(budget.as_secs_f64().to_string()),
            ),
            ("--output", files.body.as_os_str().into()),
            ("--dump-header", files.headers.as_os_str().into()),
        ] {
            arguments.extend([Value::Public(flag.into()), Value::Public(value)]);
        }
        if let Some(payload) = files.payload {
            let mut file = OsString::from("@");
            file.push(payload);
            arguments.extend([
                Value::Public("--header".into()),
                Value::Public("content-type: application/json".into()),
                Value::Public("--header".into()),
                Value::Secret(format!("authorization: Bearer {}", request.token).into()),
                Value::Public("--data-binary".into()),
                Value::Public(file),
            ]);
        }
        arguments.extend([
            Value::Public("--url".into()),
            Value::Public(request.endpoint.clone().into()),
        ]);
        ProcessSpec {
            executable: self.executable.clone(),
            arguments,
            cwd: files.directory.to_path_buf(),
            environment: self.child_environment(),
        }
    }

    fn child_environment(&self) -> BTreeMap<OsString, Value> {
        self.environment
            .iter()
            .map(|(key, value)| {
                let is_proxy = key
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .ends_with("proxy")
                    && !key.to_string_lossy().eq_ignore_ascii_case("no_proxy")
                    && !value.is_empty();
                (
                    key.clone(),
                    if is_proxy {
                        Value::Secret(value.clone())
                    } else {
                        Value::Public(value.clone())
                    },
                )
            })
            .collect()
    }
}

pub(super) struct Files<'a> {
    pub directory: &'a Path,
    pub body: &'a Path,
    pub headers: &'a Path,
    pub payload: Option<&'a Path>,
}

pub(super) fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn environment() -> Vec<(OsString, OsString)> {
    [
        "PATH",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CURL_SSL_BACKEND",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), value)))
    .collect()
}

fn normalize_trust_paths(environment: &mut [(OsString, OsString)], cwd: &Path) {
    for (key, value) in environment {
        if matches!(
            key.to_str(),
            Some("CURL_CA_BUNDLE" | "SSL_CERT_FILE" | "SSL_CERT_DIR")
        ) && !value.is_empty()
        {
            let path = PathBuf::from(&*value);
            if !path.is_absolute() {
                *value = cwd.join(path).into_os_string();
            }
        }
    }
}

fn selected_certificate_bundle(environment: &[(OsString, OsString)]) -> Option<PathBuf> {
    ["CURL_CA_BUNDLE", "SSL_CERT_FILE"]
        .into_iter()
        .find_map(|name| {
            environment
                .iter()
                .find(|(key, value)| {
                    key.as_os_str() == std::ffi::OsStr::new(name) && !value.is_empty()
                })
                .map(|(_, value)| PathBuf::from(value))
        })
}

fn find() -> Option<PathBuf> {
    let name = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut paths = std::env::var_os("PATH")
        .map(|value| {
            std::env::split_paths(&value)
                .map(|directory| directory.join(name))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    #[cfg(unix)]
    paths.extend([PathBuf::from("/usr/bin/curl"), PathBuf::from("/bin/curl")]);
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("SystemRoot").or_else(|| std::env::var_os("SYSTEMROOT")) {
        paths.push(PathBuf::from(root).join("System32/curl.exe"));
    }
    paths.into_iter().find_map(|path| {
        let metadata = std::fs::metadata(&path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                return None;
            }
        }
        path.canonicalize().ok()
    })
}

fn supported(bytes: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    let Some(version) = text
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("curl "))
        .and_then(|line| line.split_whitespace().next())
    else {
        return false;
    };
    let numbers = version
        .split('.')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>();
    let Ok(numbers) = numbers else {
        return false;
    };
    numbers.len() == 3
        && (numbers[0], numbers[1]) >= (8, 4)
        && text.lines().any(|line| {
            line.strip_prefix("Protocols:")
                .is_some_and(|value| value.split_whitespace().any(|protocol| protocol == "https"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stability_https_explicit_bundle_resolves_before_private_cwd_and_preserves_arguments() {
        let directory = tempfile::tempdir().unwrap();
        let cwd = directory.path().canonicalize().unwrap();
        let mut environment = vec![
            ("CURL_CA_BUNDLE".into(), "certificates/root ca.pem".into()),
            ("SSL_CERT_FILE".into(), "alternate.pem".into()),
            ("SSL_CERT_DIR".into(), "certificates".into()),
        ];
        normalize_trust_paths(&mut environment, &cwd);
        let bundle = cwd.join("certificates/root ca.pem");
        assert_eq!(
            selected_certificate_bundle(&environment),
            Some(bundle.clone())
        );
        assert_eq!(PathBuf::from(&environment[2].1), cwd.join("certificates"));
        let curl = Curl {
            executable: cwd.join("curl.exe"),
            certificate_bundle: Some(bundle.clone()),
            environment,
        };
        let request = Request {
            endpoint: "https://fixture.invalid/v1/models".into(),
            method: hyper::Method::GET,
            body: None,
            stream: false,
            timeout: Duration::from_secs(1),
            token: "fixture",
            started: std::time::Instant::now(),
        };
        let private = cwd.join("private transfer");
        let body = private.join("body");
        let headers = private.join("headers");
        let spec = curl.specification(
            &request,
            Files {
                directory: &private,
                body: &body,
                headers: &headers,
                payload: None,
            },
            Duration::from_secs(1),
        );
        let arguments = spec
            .arguments
            .iter()
            .map(|value| match value {
                Value::Public(value) | Value::Secret(value) => value.clone(),
            })
            .collect::<Vec<_>>();
        assert_eq!(spec.cwd, private);
        assert_eq!(arguments[0], "--disable");
        assert!(
            arguments
                .windows(2)
                .any(|pair| pair[0] == "--cacert" && pair[1] == bundle.as_os_str())
        );
        assert!(!arguments.iter().any(|value| value == "--insecure"));
        let fallback = vec![
            ("CURL_CA_BUNDLE".into(), OsString::new()),
            (
                "SSL_CERT_FILE".into(),
                cwd.join("alternate.pem").into_os_string(),
            ),
        ];
        assert_eq!(
            selected_certificate_bundle(&fallback),
            Some(cwd.join("alternate.pem"))
        );
        assert_eq!(selected_certificate_bundle(&[]), None);
    }

    #[test]
    fn stability_https_requires_unknown_length_download_limits_and_https_capability() {
        for version in ["8.4.0", "8.7.1", "9.0.0"] {
            assert!(supported(
                format!("curl {version} (fixture)\nProtocols: http https\n").as_bytes()
            ));
        }
        for text in [
            "curl 8.3.9\nProtocols: http https\n",
            "curl 8.7.1\nProtocols: http\n",
            "curl 8.7\nProtocols: https\n",
            "not curl 9.0.0\nProtocols: https\n",
        ] {
            assert!(!supported(text.as_bytes()));
        }
    }
}

pub(crate) struct Request<'a> {
    pub endpoint: String,
    pub method: hyper::Method,
    pub token: &'a str,
}

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
pub(crate) struct Curl {
    executable: PathBuf,
    environment: Vec<(OsString, OsString)>,
    certificate_bundle: Option<PathBuf>,
    #[cfg(test)]
    pinned_fixture_ca: Option<PathBuf>,
}

impl Curl {
    #[cfg(test)]
    pub(crate) fn fixture(ca: &Path) -> Self {
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
        curl.pinned_fixture_ca = Some(ca.to_path_buf());
        curl
    }
    pub(crate) fn discover() -> Result<Self, String> {
        Self::discover_for(Duration::from_secs(8), &Cancellation::default())
    }
    pub(crate) fn discover_for(
        budget: Duration,
        cancellation: &Cancellation,
    ) -> Result<Self, String> {
        if cancellation.is_cancelled() {
            return Err("HTTPS capability discovery cancelled".into());
        }
        let execution = execution_budget(budget)?.min(Duration::from_secs(5));
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
            #[cfg(test)]
            pinned_fixture_ca: None,
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
            &limits(execution),
            cancellation,
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

    pub(crate) fn specification(
        &self,
        request: &Request<'_>,
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

    /// Guardrail corpus GET/JSON POST, retaining TLS verification and owned files.
    /// No redirects or URL credentials; callers supply an admitted HTTP(S) endpoint.
    pub(crate) fn json_specification(
        &self,
        request: &Request<'_>,
        files: Files<'_>,
        budget: Duration,
        maximum: u64,
    ) -> Result<ProcessSpec, String> {
        if request.token.len() > 4096 || request.token.chars().any(char::is_control) {
            return Err("owned JSON authorization must be bounded single-line text".into());
        }
        let uri: hyper::Uri = request
            .endpoint
            .parse()
            .map_err(|_| "invalid corpus endpoint")?;
        if !matches!(uri.scheme_str(), Some("http" | "https"))
            || uri.host().is_none()
            || uri.authority().is_none_or(|a| a.as_str().contains('@'))
            || uri.query().is_some()
            || maximum == 0
            || maximum > 1024 * 1024
            || (request.method != hyper::Method::GET && request.method != hyper::Method::POST)
            || (request.method == hyper::Method::POST) != files.payload.is_some()
        {
            return Err("invalid corpus method/endpoint/body bound".into());
        }
        let mut spec = self.specification(request, files, budget);
        if request.method == hyper::Method::GET {
            spec.arguments.extend([
                Value::Public("--header".into()),
                Value::Secret(format!("authorization: Bearer {}", request.token).into()),
            ]);
        }
        // No-auth JSON consumers must not acquire an unsolicited bearer header.
        if request.token.is_empty() {
            let header = spec.arguments.iter().position(|value|
                matches!(value, Value::Secret(v) if v.to_str().is_some_and(|s| s.starts_with("authorization: Bearer "))))
                .ok_or("owned JSON authorization header absent")?;
            if header == 0
                || !matches!(&spec.arguments[header-1], Value::Public(v) if v == "--header")
            {
                return Err("owned JSON authorization option malformed".into());
            }
            spec.arguments.drain(header - 1..=header);
        }
        for (flag, value) in [
            ("--proto", "=http,https".to_owned()),
            ("--max-filesize", maximum.to_string()),
        ] {
            let index = spec
                .arguments
                .iter()
                .position(|v| matches!(v, Value::Public(s) if s == flag))
                .ok_or("owned corpus option absent")?;
            spec.arguments[index + 1] = Value::Public(value.into());
        }
        let header = spec
            .arguments
            .iter()
            .position(|v| matches!(v,Value::Public(s) if s=="--dump-header"))
            .ok_or("owned corpus header option absent")?;
        spec.arguments.drain(header..header + 2);
        spec.arguments.extend([
            Value::Public("--write-out".into()),
            Value::Public("%{http_code}".into()),
        ]);
        Ok(spec)
    }

    /// Explicit per-hop pins; this mode never inherits proxy, credential or CA settings.
    pub(crate) fn pinned_get(
        &self,
        endpoint: &str,
        resolve: &str,
        files: Files<'_>,
        budget: Duration,
        maximum: u64,
    ) -> ProcessSpec {
        let request = Request {
            endpoint: endpoint.into(),
            method: hyper::Method::GET,
            token: "",
        };
        let mut spec = self.specification(&request, files, budget);
        let index = spec
            .arguments
            .iter()
            .position(|value| matches!(value,Value::Public(v) if v == "--max-filesize"))
            .expect("owned maximum option");
        spec.arguments[index + 1] = Value::Public(maximum.to_string().into());
        spec.arguments.extend([
            Value::Public("--proxy".into()),
            Value::Public("".into()),
            Value::Public("--noproxy".into()),
            Value::Public("*".into()),
            Value::Public("--resolve".into()),
            Value::Public(resolve.into()),
        ]);
        // Signed CDN query strings are secrets even when there are no ambient credentials.
        let url_index = spec
            .arguments
            .iter()
            .position(|value| matches!(value,Value::Public(v) if v == "--url"))
            .expect("owned url option");
        spec.arguments[url_index + 1] = Value::Secret(endpoint.into());
        spec.environment.retain(|key, _| {
            matches!(
                key.to_str(),
                Some("PATH" | "SystemRoot" | "SYSTEMROOT" | "WINDIR" | "TEMP" | "TMP")
            )
        });
        // Remove explicit ambient trust paths; OS default certificate trust remains enabled.
        if let Some(index) = spec
            .arguments
            .iter()
            .position(|value| matches!(value,Value::Public(v) if v == "--cacert"))
        {
            spec.arguments.drain(index..index + 2);
        }
        #[cfg(test)]
        if let Some(ca) = &self.pinned_fixture_ca {
            spec.arguments.extend([
                Value::Public("--cacert".into()),
                Value::Public(ca.as_os_str().into()),
            ]);
        }
        spec
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

pub(crate) struct Files<'a> {
    pub directory: &'a Path,
    pub body: &'a Path,
    pub headers: &'a Path,
    pub payload: Option<&'a Path>,
}

/// Reserve graceful, forced, and owned EOF-drain allowances before spawning.
pub(crate) fn execution_budget(total: Duration) -> Result<Duration, String> {
    let execution = total.saturating_sub(Duration::from_secs(3));
    if execution.is_zero() {
        return Err("HTTPS budget cannot reserve owned cleanup".into());
    }
    Ok(execution)
}

pub(crate) fn limits(execution: Duration) -> Limits {
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
            pinned_fixture_ca: None,
            environment,
        };
        let request = Request {
            endpoint: "https://fixture.invalid/v1/models".into(),
            method: hyper::Method::GET,
            token: "fixture",
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
        assert_eq!(
            execution_budget(Duration::from_secs(8)).unwrap(),
            Duration::from_secs(5)
        );
        assert!(execution_budget(Duration::from_secs(3)).is_err());
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

#[cfg(test)]
mod pinned_tests {
    use super::*;
    #[test]
    fn hf_projector_pinned_command_disables_proxy_and_redacts_signed_url() {
        let cancellation = Cancellation::default();
        cancellation.cancel();
        let refused = Curl::discover_for(Duration::from_secs(8), &cancellation);
        assert!(matches!(refused,Err(error) if error == "HTTPS capability discovery cancelled"));
        let directory = tempfile::tempdir().unwrap();
        let ca = directory.path().join("owned-ca");
        std::fs::write(&ca, b"owned fixture trust input").unwrap();
        let mut curl = Curl::fixture(&ca);
        curl.environment.extend([
            (
                "HTTPS_PROXY".into(),
                "https://user:secret@proxy.invalid".into(),
            ),
            ("SSL_CERT_DIR".into(), "ambient-trust".into()),
        ]);
        let endpoint = "https://hf.co/model?signature=owned-secret";
        let spec = curl.pinned_get(
            endpoint,
            "hf.co:443:13.33.88.1",
            Files {
                directory: directory.path(),
                body: &directory.path().join("body"),
                headers: Path::new("-"),
                payload: None,
            },
            Duration::from_secs(1),
            32,
        );
        assert!(
            !spec
                .environment
                .contains_key(std::ffi::OsStr::new("HTTPS_PROXY"))
        );
        assert!(
            !spec
                .environment
                .contains_key(std::ffi::OsStr::new("SSL_CERT_DIR"))
        );
        assert!(
            spec.arguments
                .iter()
                .any(|v| matches!(v,Value::Secret(s) if s==endpoint))
        );
        for (flag, value) in [
            ("--proxy", ""),
            ("--noproxy", "*"),
            ("--resolve", "hf.co:443:13.33.88.1"),
            ("--max-filesize", "32"),
        ] {
            assert!(spec.arguments.windows(2).any(|p|matches!((&p[0],&p[1]),(Value::Public(a),Value::Public(b)) if a==flag && b==value)));
        }
        assert!(!spec.arguments.iter().any(
            |v| matches!(v,Value::Public(s) if s=="--insecure"||s=="--location"||s=="--netrc")
        ));
        directory.close().unwrap();
    }
}
#[cfg(test)]
mod corpus_tests {
    use super::*;
    #[test]
    fn guardrail_json_projection_preserves_tls_secret_post_payload_and_download_caps() {
        let root = tempfile::tempdir().unwrap();
        let curl = Curl {
            executable: root.path().join("curl"),
            environment: vec![],
            certificate_bundle: None,
            pinned_fixture_ca: None,
        };
        let body = root.path().join("body");
        let headers = root.path().join("headers");
        let payload = root.path().join("payload");
        for endpoint in [
            "https://fixture.invalid/v1/chat/completions",
            "http://localhost:1234/v1/chat/completions",
        ] {
            let request = Request {
                endpoint: endpoint.into(),
                method: hyper::Method::POST,
                token: "owned-token",
            };
            let spec = curl
                .json_specification(
                    &request,
                    Files {
                        directory: root.path(),
                        body: &body,
                        headers: &headers,
                        payload: Some(&payload),
                    },
                    Duration::from_secs(2),
                    1024,
                )
                .unwrap();
            assert!(
                !spec
                    .arguments
                    .iter()
                    .any(|v| matches!(v,Value::Public(a) if a=="--dump-header"))
            );
            assert!(spec.arguments.windows(2).any(|p|matches!((&p[0],&p[1]),(Value::Public(a),Value::Public(b)) if a=="--write-out"&&b=="%{http_code}")));
            assert!(matches!(&spec.arguments[0],Value::Public(v) if v=="--disable"));
            assert!(
                spec.arguments.iter().any(
                    |v| matches!(v,Value::Secret(v) if v=="authorization: Bearer owned-token")
                )
            );
            for (flag, value) in [
                ("--proto", "=http,https"),
                ("--proto-redir", "=https"),
                ("--max-filesize", "1024"),
                ("--request", "POST"),
            ] {
                assert!(spec.arguments.windows(2).any(|p|matches!((&p[0],&p[1]),(Value::Public(a),Value::Public(b)) if a==flag&&b==value)));
            }
            assert!(!spec.arguments.iter().any(
                |v| matches!(v,Value::Public(a) if a=="--location"||a=="--insecure"||a=="--netrc")
            ));
        }
        for endpoint in [
            "file:///etc/hosts",
            "https://user:secret@fixture.invalid/v1",
            "https://fixture.invalid/v1?token=secret",
        ] {
            assert!(
                curl.json_specification(
                    &Request {
                        endpoint: endpoint.into(),
                        method: hyper::Method::GET,
                        token: ""
                    },
                    Files {
                        directory: root.path(),
                        body: &body,
                        headers: &headers,
                        payload: None
                    },
                    Duration::from_secs(2),
                    1024
                )
                .is_err()
            );
        }
        let get = curl
            .json_specification(
                &Request {
                    endpoint: "https://fixture.invalid/v1/models".into(),
                    method: hyper::Method::GET,
                    token: "owned-token",
                },
                Files {
                    directory: root.path(),
                    body: &body,
                    headers: &headers,
                    payload: None,
                },
                Duration::from_secs(2),
                1024,
            )
            .unwrap();
        assert!(
            get.arguments
                .iter()
                .any(|v| matches!(v,Value::Secret(v) if v=="authorization: Bearer owned-token"))
        );
        root.close().unwrap();
    }
}
#[cfg(test)]
mod suffix_json_tests {
    use super::*;
    #[test]
    fn suffix_json_no_auth_keeps_guardrail_auth_and_existing_tls_body_status_contract() {
        let root = tempfile::tempdir().unwrap();
        let curl = Curl {
            executable: root.path().join("curl"),
            environment: vec![],
            certificate_bundle: None,
            pinned_fixture_ca: None,
        };
        let body = root.path().join("body");
        let headers = root.path().join("headers");
        let payload = root.path().join("payload");
        for token in ["", "mesh-llm-ci"] {
            for method in [hyper::Method::GET, hyper::Method::POST] {
                let post = method == hyper::Method::POST;
                let spec = curl
                    .json_specification(
                        &Request {
                            endpoint: "https://fixture.invalid/v1/chat/completions".into(),
                            method,
                            token,
                        },
                        Files {
                            directory: root.path(),
                            body: &body,
                            headers: &headers,
                            payload: post.then_some(payload.as_path()),
                        },
                        Duration::from_secs(1),
                        1048576,
                    )
                    .unwrap();
                let args: Vec<_> = spec
                    .arguments
                    .iter()
                    .map(|v| v.value().to_string_lossy().into_owned())
                    .collect();
                assert_eq!(
                    args.iter()
                        .filter(|s| s.starts_with("authorization:"))
                        .count(),
                    usize::from(!token.is_empty())
                );
                assert!(
                    args.windows(2)
                        .any(|v| v == ["--write-out", "%{http_code}"])
                );
                assert!(args.windows(2).any(|v| v == ["--proto", "=http,https"]));
                assert!(
                    !args
                        .iter()
                        .any(|v| v == "--insecure" || v == "--location" || v == "--dump-header")
                );
                assert_eq!(
                    args.iter()
                        .filter(|s| s.as_str() == "--data-binary")
                        .count(),
                    usize::from(post)
                );
            }
        }
        root.close().unwrap();
    }
}

#[cfg(test)]
mod borrowed_authorization_tests {
    use super::*;
    #[test]
    fn borrowed_json_authorization_becomes_owned_secret_spec_after_request_owner_drops() {
        let root = tempfile::tempdir().unwrap();
        let curl = Curl {
            executable: root.path().join("inert-curl-not-launched"),
            environment: vec![],
            certificate_bundle: None,
            pinned_fixture_ca: None,
        };
        for control in ['\r', '\n', '\0'] {
            let owned = format!("owned{control}value");
            let body = root.path().join("body");
            let headers = root.path().join("headers");
            assert!(
                curl.json_specification(
                    &Request {
                        endpoint: "https://fixture.invalid/v1/models".into(),
                        method: hyper::Method::GET,
                        token: &owned
                    },
                    Files {
                        directory: root.path(),
                        body: &body,
                        headers: &headers,
                        payload: None
                    },
                    Duration::from_secs(1),
                    1024
                )
                .is_err()
            );
        }
        for method in [hyper::Method::GET, hyper::Method::POST] {
            let spec = {
                let owned = ["bounded", "borrowed", "authorization"].join("-");
                let body = root.path().join("body");
                let headers = root.path().join("headers");
                let payload = root.path().join("payload");
                let post = method == hyper::Method::POST;
                let request = Request {
                    endpoint: "https://fixture.invalid/v1/chat/completions".into(),
                    method,
                    token: owned.as_str(),
                };
                curl.json_specification(
                    &request,
                    Files {
                        directory: root.path(),
                        body: &body,
                        headers: &headers,
                        payload: post.then_some(payload.as_path()),
                    },
                    Duration::from_secs(1),
                    1024,
                )
                .unwrap()
            };
            assert_eq!(spec.arguments.iter().filter(|v| matches!(v,Value::Secret(s) if s=="authorization: Bearer bounded-borrowed-authorization")).count(),1);
            assert!(!spec.arguments.iter().any(|v| matches!(v,Value::Public(s) if s.to_string_lossy().contains("bounded-borrowed-authorization"))));
            assert!(
                !spec
                    .arguments
                    .iter()
                    .any(|v| matches!(v,Value::Public(s) if s=="--insecure"||s=="--location"))
            );
        }
        root.close().unwrap();
    }
}

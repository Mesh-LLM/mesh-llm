//! Bounded raw GitHub protocol transport with explicit credentials and no auth fallback.
use super::Request;
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
};
use serde::{Deserialize, Deserializer};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const RESPONSE_LIMIT: usize = 2 * 1024 * 1024;
const TRANSACTION_LIMIT: usize = 32 * 1024 * 1024;

// Deliberately no Debug/Serialize. Mint replies may contain a new secret token.
pub(crate) struct ApiResponse {
    pub status: u16,
    pub body: Vec<u8>,
}
pub(crate) trait GithubRead {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse>;
}

pub(crate) struct SecretToken(String);
impl SecretToken {
    pub(crate) fn checked(value: String) -> DynResult<Self> {
        if value.is_empty() || value.len() > 16 * 1024 || value.chars().any(char::is_control) {
            return Err("GitHub credential must be nonempty bounded single-line text".into());
        }
        Ok(Self(value))
    }
}
impl<'de> Deserialize<'de> for SecretToken {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::checked(value)
            .map_err(|_| serde::de::Error::custom("invalid opaque GitHub credential"))
    }
}

pub(crate) struct GithubClient {
    executable: PathBuf,
    root: PathBuf,
    state: tempfile::TempDir,
    environment: BTreeMap<OsString, Value>,
    deadline: Instant,
    cancellation: Cancellation,
    received: usize,
    requests: u32,
    jwt_header: Option<PathBuf>,
}
impl GithubClient {
    pub(crate) fn installation(
        root: &Path,
        cancellation: &Cancellation,
        credential: SecretToken,
    ) -> DynResult<Self> {
        Self::authenticated(root, cancellation, credential, false)
    }
    pub(crate) fn app_jwt(
        root: &Path,
        cancellation: &Cancellation,
        credential: SecretToken,
    ) -> DynResult<Self> {
        Self::authenticated(root, cancellation, credential, true)
    }
    fn authenticated(
        root: &Path,
        cancellation: &Cancellation,
        credential: SecretToken,
        jwt: bool,
    ) -> DynResult<Self> {
        let root = root.canonicalize()?;
        let executable = find_executable(if jwt { "curl" } else { "gh" }, &root)?;
        let state = tempfile::Builder::new()
            .prefix("ci-governance-gh-")
            .tempdir()?;
        let mut environment = child_environment();
        environment.insert(
            "GH_CONFIG_DIR".into(),
            Value::Public(state.path().as_os_str().into()),
        );
        environment.insert("GH_PROMPT_DISABLED".into(), Value::Public("1".into()));
        environment.insert("GH_NO_UPDATE_NOTIFIER".into(), Value::Public("1".into()));
        environment.insert(
            "GH_NO_EXTENSION_UPDATE_NOTIFIER".into(),
            Value::Public("1".into()),
        );
        environment.insert("GH_HOST".into(), Value::Public("github.com".into()));
        let jwt_header = authenticate(state.path(), &mut environment, credential, jwt)?;
        Ok(Self {
            executable,
            root,
            state,
            environment,
            deadline: Instant::now() + Duration::from_secs(180),
            cancellation: cancellation.clone(),
            received: 0,
            requests: 0,
            jwt_header,
        })
    }
    fn arguments(
        &self,
        encoded: &super::request::Encoded,
        execution: Duration,
    ) -> DynResult<Vec<Value>> {
        let mut args: Vec<Value> = if let Some(header) = &self.jwt_header {
            let mut args: Vec<Value> = [
                "--disable",
                "--silent",
                "--show-error",
                "--include",
                "--suppress-connect-headers",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--max-redirs",
                "0",
                "--request",
                encoded.method,
            ]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect();
            let mut header_file = OsString::from("@");
            header_file.push(header);
            for (flag, value) in [
                ("--header", header_file),
                ("--max-time", execution.as_secs_f64().to_string().into()),
                (
                    "--connect-timeout",
                    execution
                        .min(Duration::from_secs(15))
                        .as_secs_f64()
                        .to_string()
                        .into(),
                ),
                ("--max-filesize", RESPONSE_LIMIT.to_string().into()),
                (
                    "--url",
                    format!("https://api.github.com/{}", encoded.endpoint).into(),
                ),
            ] {
                args.extend([Value::Public(flag.into()), Value::Public(value)]);
            }
            args
        } else {
            [
                "api",
                "--hostname",
                "github.com",
                "--include",
                "--method",
                encoded.method,
                &encoded.endpoint,
            ]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect()
        };
        for header in [
            format!("Accept: {}", encoded.accept),
            "Content-Type: application/json".into(),
            "X-GitHub-Api-Version: 2022-11-28".into(),
            "Cache-Control: no-cache".into(),
        ] {
            args.extend([
                Value::Public("--header".into()),
                Value::Public(header.into()),
            ]);
        }
        if let Some(body) = &encoded.body {
            let path = self.state.path().join("request.json");
            std::fs::write(&path, body)?;
            let (flag, value) = if self.jwt_header.is_some() {
                let mut file = OsString::from("@");
                file.push(&path);
                ("--data-binary", file)
            } else {
                ("--input", path.into_os_string())
            };
            args.extend([Value::Public(flag.into()), Value::Public(value)]);
        }
        Ok(args)
    }
}

fn authenticate(
    state: &Path,
    environment: &mut BTreeMap<OsString, Value>,
    credential: SecretToken,
    jwt: bool,
) -> DynResult<Option<PathBuf>> {
    if !jwt {
        environment.insert("GH_TOKEN".into(), Value::Secret(credential.0.into()));
        return Ok(None);
    }
    use std::io::Write;
    let path = state.join("authorization.header");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path)?;
    writeln!(file, "Authorization: Bearer {}", credential.0)?;
    Ok(Some(path))
}
impl GithubRead for GithubClient {
    fn send(&mut self, request: &Request) -> DynResult<ApiResponse> {
        if self.cancellation.is_cancelled() {
            return Err("governance API transaction cancelled".into());
        }
        self.requests += 1;
        if self.requests > 512 {
            return Err("governance API transaction request bound exceeded".into());
        }
        let execution = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("governance API transaction exceeded 180 seconds")?;
        let encoded = request.encode()?;
        let args = self.arguments(&encoded, execution)?;
        let spec = ProcessSpec {
            executable: self.executable.clone(),
            arguments: args,
            cwd: self.root.clone(),
            environment: self.environment.clone(),
        };
        let report = process::supervise_raw(
            &spec,
            &limits(execution),
            &self.cancellation,
            RawCaptureOptions {
                stdout: NonZeroUsize::new(RESPONSE_LIMIT),
                stderr: None,
            },
        )?;
        if report.process.outcome != process::Outcome::Exited
            || !report.process.cleanup.complete
            || report.process.failure.is_some()
        {
            return Err(
                "governance GitHub transport failed, cancelled, overflowed or did not clean up"
                    .into(),
            );
        }
        let raw = report
            .stdout
            .ok_or("governance GitHub response incomplete")?;
        self.received = self
            .received
            .checked_add(raw.as_bytes().len())
            .ok_or("governance response byte bound exceeded")?;
        if self.received > TRANSACTION_LIMIT {
            return Err("governance responses exceed 32 MiB transaction bound".into());
        }
        let mut response = parse_response(raw.as_bytes())?;
        if !report.process.success() && (200..300).contains(&response.status) {
            return Err("GitHub failed while reading a successful response".into());
        }
        if !(200..300).contains(&response.status) {
            // Never expose a provider error body that might reflect authentication.
            response.body.clear();
        }
        Ok(response)
    }
}

fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn child_environment() -> BTreeMap<OsString, Value> {
    let mut environment = BTreeMap::new();
    for name in [
        "PATH",
        "SYSTEMROOT",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CURL_CA_BUNDLE",
    ] {
        if let Some(value) = std::env::var_os(name) {
            // Proxy URLs can carry credentials; none are rendered as diagnostics.
            let value = if name.to_ascii_lowercase().contains("proxy") {
                Value::Secret(value)
            } else {
                Value::Public(value)
            };
            environment.insert(name.into(), value);
        }
    }
    environment
}

fn parse_response(bytes: &[u8]) -> DynResult<ApiResponse> {
    let (end, separator) = if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n")
    {
        (end, 4)
    } else if let Some(end) = bytes.windows(2).position(|part| part == b"\n\n") {
        (end, 2)
    } else {
        return Err("GitHub response headers are incomplete".into());
    };
    if end > 64 * 1024 {
        return Err("GitHub response headers exceed supported bound".into());
    }
    let first = bytes[..end]
        .split(|byte| *byte == b'\n')
        .next()
        .ok_or("GitHub status line missing")?;
    let first = std::str::from_utf8(first).map_err(|_| "GitHub status line is not UTF-8")?;
    let fields: Vec<_> = first.split_whitespace().collect();
    if fields.len() < 2 || !matches!(fields[0], "HTTP/1.0" | "HTTP/1.1" | "HTTP/2" | "HTTP/2.0") {
        return Err("invalid GitHub HTTP status line".into());
    }
    let status = fields[1]
        .parse::<u16>()
        .map_err(|_| "invalid GitHub HTTP status")?;
    if !(100..=599).contains(&status) {
        return Err("invalid GitHub HTTP status".into());
    }
    Ok(ApiResponse {
        status,
        body: bytes[end + separator..].to_vec(),
    })
}

fn find_executable(name: &str, root: &Path) -> DynResult<PathBuf> {
    let path = std::env::var_os("PATH").ok_or("governance executable PATH unavailable")?;
    for directory in std::env::split_paths(&path) {
        let directory = if directory.is_absolute() {
            directory
        } else {
            root.join(directory)
        };
        let name = if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.into()
        };
        let candidate = directory.join(name);
        let Ok(metadata) = candidate.metadata() else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o111 == 0 {
                continue;
            }
        }
        return Ok(candidate);
    }
    Err("governance required executable unavailable on PATH".into())
}

pub(crate) fn checkout_head(root: &Path, cancellation: &Cancellation) -> DynResult<String> {
    let raw = git_bytes(
        root,
        cancellation,
        &["rev-parse", "--verify", "HEAD^{commit}"],
        128,
    )?;
    let head = std::str::from_utf8(&raw)
        .map_err(|_| "controller HEAD is not UTF-8")?
        .trim_end_matches(['\r', '\n']);
    if !crate::ci_governance::policy::valid_sha(head) {
        return Err("invalid controller checkout SHA".into());
    }
    Ok(head.into())
}

pub(crate) fn checkout_activation(
    root: &Path,
    revision: &str,
    cancellation: &Cancellation,
) -> DynResult<Vec<u8>> {
    if !crate::ci_governance::policy::valid_sha(revision) {
        return Err("invalid activation source revision".into());
    }
    if checkout_head(root, cancellation)? != revision {
        return Err("activation revision must equal controller checkout HEAD".into());
    }
    let tree = git_bytes(
        root,
        cancellation,
        &[
            "ls-tree",
            "--full-tree",
            revision,
            "--",
            "ci/governance/activation.json",
        ],
        512,
    )?;
    let tree = std::str::from_utf8(&tree)
        .map_err(|_| "activation tree entry invalid")?
        .trim_end_matches(['\r', '\n']);
    let fields: Vec<_> = tree.split_whitespace().collect();
    if fields.len() != 4
        || fields[0] != "100644"
        || fields[1] != "blob"
        || !crate::ci_governance::policy::valid_sha(fields[2])
        || fields[3] != "ci/governance/activation.json"
    {
        return Err(
            "activation must be a regular tracked100644 blob at protected controller revision"
                .into(),
        );
    }
    git_bytes(
        root,
        cancellation,
        &[
            "show",
            "--no-textconv",
            &format!("{revision}:ci/governance/activation.json"),
        ],
        64 * 1024,
    )
}

fn git_bytes(
    root: &Path,
    cancellation: &Cancellation,
    arguments: &[&str],
    bound: usize,
) -> DynResult<Vec<u8>> {
    let root = root.canonicalize()?;
    let executable = find_executable("git", &root)?;
    let mut environment = BTreeMap::new();
    for name in ["PATH", "SYSTEMROOT", "SystemRoot"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Value::Public(value));
        }
    }
    environment.insert("GIT_TERMINAL_PROMPT".into(), Value::Public("0".into()));
    environment.insert("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into()));
    let global_config = tempfile::NamedTempFile::new()?;
    environment.insert(
        "GIT_CONFIG_GLOBAL".into(),
        Value::Public(global_config.path().as_os_str().into()),
    );
    environment.insert("GIT_CONFIG_COUNT".into(), Value::Public("0".into()));
    environment.insert("GIT_NO_REPLACE_OBJECTS".into(), Value::Public("1".into()));
    environment.insert("GIT_NO_LAZY_FETCH".into(), Value::Public("1".into()));
    environment.insert("GIT_PAGER".into(), Value::Public("".into()));
    let mut argv: Vec<Value> = [
        "--no-replace-objects",
        "-c",
        "core.hooksPath=",
        "-c",
        "core.fsmonitor=false",
    ]
    .into_iter()
    .map(|value| Value::Public(value.into()))
    .collect();
    argv.extend(arguments.iter().map(|value| Value::Public((*value).into())));
    let spec = ProcessSpec {
        executable,
        cwd: root,
        environment,
        arguments: argv,
    };
    let report = process::supervise_raw(
        &spec,
        &limits(Duration::from_secs(15)),
        cancellation,
        RawCaptureOptions {
            stdout: NonZeroUsize::new(bound),
            stderr: None,
        },
    )?;
    if !report.process.success()
        || report.process.failure.is_some()
        || !report.process.cleanup.complete
    {
        return Err("cannot resolve protected controller checkout HEAD".into());
    }
    let raw = report
        .stdout
        .ok_or("controller checkout identity incomplete")?;
    Ok(raw.as_bytes().to_vec())
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;

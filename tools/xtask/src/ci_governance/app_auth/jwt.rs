//! Ephemeral RS256 App JWT creation; neither credentials nor JWTs are printed.
use crate::ci_governance::collector::SecretToken;
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
};
use std::{
    collections::BTreeMap,
    io::Write,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn credential(
    root: &Path,
    app_id: u64,
    cancellation: &Cancellation,
    now: u64,
) -> DynResult<SecretToken> {
    let key = std::env::var_os("CI_GOVERNANCE_APP_PRIVATE_KEY");
    let jwt = std::env::var_os("CI_GOVERNANCE_APP_JWT");
    match (key,jwt) {
        (None,Some(jwt))=>SecretToken::checked(jwt.into_string().map_err(|_|"invalid explicit App JWT encoding")?),
        (Some(key),None)=>sign(root,app_id,cancellation,now,key.into_string().map_err(|_|"invalid App private key encoding")?),
        (None,None)=>Err("explicit protected App private key or ephemeral App JWT is unavailable".into()),
        (Some(_),Some(_))=>Err("choose one explicit App credential source; private key and JWT cannot both be supplied".into()),
    }
}

fn sign(
    root: &Path,
    app_id: u64,
    cancellation: &Cancellation,
    now: u64,
    key: String,
) -> DynResult<SecretToken> {
    if app_id == 0
        || key.is_empty()
        || key.len() > 64 * 1024
        || key
            .bytes()
            .any(|ch| ch == 0 || (!ch.is_ascii_graphic() && !ch.is_ascii_whitespace()))
    {
        return Err("App private key must be bounded PEM text for the reviewed App".into());
    }
    let state = tempfile::Builder::new()
        .prefix("ci-governance-jwt-")
        .tempdir()?;
    let configuration = state.path().join("openssl.cnf");
    std::fs::write(&configuration, b"")?;
    let mut private = tempfile::NamedTempFile::new_in(state.path())?;
    private.write_all(key.as_bytes())?;
    private.flush()?;
    drop(key);
    let claims = serde_json::to_vec(
        &serde_json::json!({"iat":now.saturating_sub(60),"exp":now.checked_add(540).ok_or("App JWT clock overflow")?,"iss":app_id}),
    )?;
    let signing_input = format!(
        "{}.{}",
        encode(br#"{"alg":"RS256","typ":"JWT"}"#),
        encode(&claims)
    );
    let mut input = tempfile::NamedTempFile::new_in(state.path())?;
    input.write_all(signing_input.as_bytes())?;
    input.flush()?;
    let executable = openssl(root)?;
    invoke(
        root,
        &executable,
        vec![
            "rsa".into(),
            "-in".into(),
            private.path().as_os_str().to_owned(),
            "-check".into(),
            "-noout".into(),
            "-passin".into(),
            "pass:".into(),
        ],
        &configuration,
        cancellation,
        false,
    )?;
    let signature = invoke(
        root,
        &executable,
        vec![
            "dgst".into(),
            "-sha256".into(),
            "-sign".into(),
            private.path().as_os_str().to_owned(),
            "-passin".into(),
            "pass:".into(),
            "-binary".into(),
            input.path().as_os_str().to_owned(),
        ],
        &configuration,
        cancellation,
        true,
    )?;
    SecretToken::checked(format!("{signing_input}.{}", encode(&signature)))
}

fn invoke(
    root: &Path,
    executable: &Path,
    args: Vec<std::ffi::OsString>,
    configuration: &Path,
    cancellation: &Cancellation,
    capture: bool,
) -> DynResult<Vec<u8>> {
    let mut environment = BTreeMap::new();
    for name in ["PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Value::Public(value));
        }
    }
    environment.insert(
        "OPENSSL_CONF".into(),
        Value::Public(configuration.as_os_str().to_owned()),
    );
    let spec = ProcessSpec {
        executable: executable.to_owned(),
        arguments: args.into_iter().map(Value::Public).collect(),
        cwd: root.to_owned(),
        environment,
    };
    let report = process::supervise_raw(
        &spec,
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 0,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancellation,
        RawCaptureOptions {
            stdout: if capture {
                NonZeroUsize::new(4096)
            } else {
                None
            },
            stderr: None,
        },
    )?;
    if report.process.outcome != process::Outcome::Exited
        || !report.process.success()
        || !report.process.cleanup.complete
        || report.process.failure.is_some()
    {
        return Err(
            "bounded OpenSSL RSA JWT signing failed; credential diagnostics are suppressed".into(),
        );
    }
    if !capture {
        return Ok(Vec::new());
    }
    let signature = report
        .stdout
        .ok_or("App JWT signature capture incomplete")?;
    if signature.as_bytes().is_empty() {
        return Err("App JWT RSA signature is empty".into());
    }
    Ok(signature.as_bytes().to_vec())
}

fn openssl(root: &Path) -> DynResult<PathBuf> {
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let directory = if directory.is_absolute() {
            directory
        } else {
            root.join(directory)
        };
        let candidate = directory.join(if cfg!(windows) {
            "openssl.exe"
        } else {
            "openssl"
        });
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err("governance App signing requires provisioned OpenSSL on PATH".into())
}

fn encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut output = String::new();
    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);
        output.push(char::from(ALPHABET[usize::from(first >> 2)]));
        output.push(char::from(
            ALPHABET[usize::from((first & 3) << 4 | second >> 4)],
        ));
        if chunk.len() > 1 {
            output.push(char::from(
                ALPHABET[usize::from((second & 15) << 2 | third >> 6)],
            ));
        }
        if chunk.len() > 2 {
            output.push(char::from(ALPHABET[usize::from(third & 63)]));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    #[test]
    fn jwt_base64url_has_no_padding_and_uses_url_alphabet() {
        for (bytes, encoded) in [
            (b"".as_slice(), ""),
            (b"f", "Zg"),
            (b"fo", "Zm8"),
            (b"foo", "Zm9v"),
            (&[251, 255, 255], "-___"),
        ] {
            assert_eq!(super::encode(bytes), encoded);
        }
    }
}

//! Blocking OS DNS belongs exclusively to the bounded owned worker process.
use crate::{command::DynResult, process};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    io::{Read, Write},
    net::{IpAddr, ToSocketAddrs},
    path::Path,
    time::Duration,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    host: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    request_sha256: String,
    host: String,
    addresses: Vec<IpAddr>,
    failure: Option<String>,
}

pub(super) fn resolve(
    host: &str,
    budget: Duration,
    cancellation: &process::Cancellation,
) -> DynResult<Vec<IpAddr>> {
    resolve_using(
        host,
        &std::env::current_exe()?,
        &["models", "projector-download", "resolve-worker"],
        budget,
        cancellation,
    )
}

fn resolve_using(
    host: &str,
    executable: &Path,
    prefix: &[&str],
    budget: Duration,
    cancellation: &process::Cancellation,
) -> DynResult<Vec<IpAddr>> {
    let execution = process::curl_https::execution_budget(budget)?.min(Duration::from_secs(5));
    let directory = tempfile::tempdir()?;
    let input = directory.path().join("request.json");
    let output = directory.path().join("receipt.json");
    let bytes = serde_json::to_vec(&Request {
        schema_version: 1,
        host: host.into(),
    })?;
    std::fs::write(&input, &bytes)?;
    let result = process::supervise(
        &process::ProcessSpec {
            executable: executable.into(),
            cwd: directory.path().into(),
            arguments: prefix
                .iter()
                .copied()
                .chain(["--input"])
                .map(|s| process::Value::Public(s.into()))
                .chain([
                    process::Value::Public(input.into_os_string()),
                    process::Value::Public("--output".into()),
                    process::Value::Public(output.clone().into_os_string()),
                ])
                .collect(),
            environment: ["SYSTEMROOT", "WINDIR"]
                .into_iter()
                .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), process::Value::Public(v))))
                .collect(),
        },
        &process::curl_https::limits(execution),
        cancellation,
        process::OutputFiles::default(),
    );
    let result: DynResult<Vec<IpAddr>> = (|| {
        let report = result?;
        if !super::transfer::clean(&report) {
            return Err(format!("projector DNS worker failed: {:?}", report.outcome).into());
        }
        let receipt: Receipt = serde_json::from_slice(&bounded_read(&output, 65536)?)?;
        if receipt.schema_version != 1
            || receipt.request_sha256 != hex::encode(Sha256::digest(&bytes))
            || receipt.host != host
            || receipt.addresses.len() > 64
            || receipt.failure.is_some()
        {
            return Err("projector DNS receipt invalid or resolution failed".into());
        }
        Ok(receipt.addresses)
    })();
    let close = directory.close();
    match (result, close) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (result, Err(error)) => Err(format!(
            "DNS result {}; private-state cleanup failed: {error}",
            result
                .as_ref()
                .err()
                .map_or("successful".into(), ToString::to_string)
        )
        .into()),
    }
}

pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [input_flag, input, output_flag, output] = args else {
        return Err("resolve-worker --input ABS --output ABS".into());
    };
    if input_flag != "--input" || output_flag != "--output" {
        return Err("invalid resolver worker options".into());
    }
    let input = Path::new(input);
    let output = Path::new(output);
    if !input.is_absolute() || !output.is_absolute() {
        return Err("resolver paths must be absolute".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("resolver output must be fresh".into()),
    }
    let bytes = bounded_read(input, 16384)?;
    let request: Request = serde_json::from_slice(&bytes)?;
    if request.schema_version != 1 || request.host.len() > 253 {
        return Err("invalid resolver request".into());
    }
    let url = super::url_policy::trusted(&format!("https://{}/", request.host))?;
    if url.host_str() != Some(request.host.as_str()) {
        return Err("resolver host must be canonical trusted host".into());
    }
    let resolved = (request.host.as_str(), 443).to_socket_addrs();
    let (addresses, failure) = match resolved {
        Ok(values) => {
            let mut unique = BTreeSet::new();
            for value in values {
                unique.insert(value.ip());
                if unique.len() > 64 {
                    break;
                }
            }
            let values = unique;
            if values.is_empty() || values.len() > 64 {
                (Vec::new(), Some("DNS answer empty or over limit".into()))
            } else {
                (values.into_iter().collect(), None)
            }
        }
        Err(_) => (Vec::new(), Some("OS DNS resolution failed".into())),
    };
    let receipt = Receipt {
        schema_version: 1,
        request_sha256: hex::encode(Sha256::digest(&bytes)),
        host: request.host,
        addresses,
        failure,
    };
    let mut temporary =
        tempfile::NamedTempFile::new_in(output.parent().ok_or("resolver parent absent")?)?;
    temporary.write_all(&serde_json::to_vec(&receipt)?)?;
    temporary.flush()?;
    temporary.persist_noclobber(output)?;
    if receipt.failure.is_some() {
        return Err("DNS resolution failed; partial receipt retained".into());
    }
    Ok(())
}

pub(super) fn bounded_read(path: &Path, maximum: u64) -> DynResult<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("projector input must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened projector input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("bounded input exceeds limit".into());
    }
    Ok(bytes)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt as _;
    #[test]
    fn hf_projector_dns_owned_child_receipt_binding_and_deadline() {
        for mode in ["valid", "wrong-hash", "failure", "deadline"] {
            let directory = tempfile::tempdir().unwrap();
            let executable = directory.path().join("worker");
            let bytes = serde_json::to_vec(&Request {
                schema_version: 1,
                host: "hf.co".into(),
            })
            .unwrap();
            let receipt = Receipt {
                schema_version: 1,
                host: "hf.co".into(),
                request_sha256: if mode == "wrong-hash" {
                    "wrong".into()
                } else {
                    hex::encode(Sha256::digest(&bytes))
                },
                addresses: vec!["13.33.88.1".parse().unwrap()],
                failure: None,
            };
            let body = if mode == "deadline" {
                "#!/bin/bash\n/bin/sleep 5\n".into()
            } else if mode == "failure" {
                "#!/bin/bash\nexit 37\n".into()
            } else {
                format!(
                    "#!/bin/bash\n[[ $# == 4 && $1 == --input && $3 == --output ]] || exit 91\n/bin/cat > \"$4\" <<'RECEIPT'\n{}\nRECEIPT\n",
                    serde_json::to_string(&receipt).unwrap()
                )
            };
            std::fs::write(&executable, body).unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
            let result = resolve_using(
                "hf.co",
                &executable,
                &[],
                if mode == "deadline" {
                    Duration::from_millis(3100)
                } else {
                    Duration::from_secs(8)
                },
                &process::Cancellation::default(),
            );
            assert_eq!(result.is_ok(), mode == "valid", "{mode}");
            if let Ok(addresses) = result {
                assert_eq!(addresses, vec!["13.33.88.1".parse::<IpAddr>().unwrap()]);
            }
            directory.close().unwrap();
        }
    }
}

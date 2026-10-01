//! Read the immutable producer's build contract before composing product bytes.
use super::python_object::{display, item};
use crate::ci_plan::document::Json;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Duration;

pub(super) fn read(host: &Path) -> Result<Json, String> {
    let spec = ProcessSpec {
        executable: host.canonicalize().map_err(|error| error.to_string())?,
        arguments: ["--log-format", "json", "--print-build-contract"]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
        cwd: std::env::current_dir().map_err(|error| error.to_string())?,
        environment: ["PATH", "SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect(),
    };
    let limits = Limits {
        execution: Duration::from_secs(30),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(2),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .map_err(|error| format!("host build contract probe failed: {error:?}"))?;
    if !report.process.success() {
        return Err(format!(
            "host build contract probe failed: {:?}",
            report.process
        ));
    }
    let bytes = report.stdout.ok_or("host build contract output missing")?;
    let contract = Json::parse(bytes.as_bytes()).map_err(|error| error.to_string())?;
    check(&contract)?;
    Ok(contract)
}

fn check(contract: &Json) -> Result<(), String> {
    if contract.get("schema_version").and_then(Json::as_int) != Some(1) {
        return Err("ValueError: unsupported host build contract schema".to_owned());
    }
    for field in ["product_version", "runtime_release", "skippy_abi"] {
        if contract
            .get(field)
            .and_then(Json::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(format!(
                "ValueError: host build contract is missing {field}"
            ));
        }
    }
    Ok(())
}

pub(super) fn validate(
    contract: &Json,
    runtime: &Json,
    id: &Json,
    version: &str,
) -> Result<(), String> {
    check(contract)?;
    let product = item(contract, "product_version")?;
    if product.as_str() != Some(version) {
        return Err(format!(
            "ValueError: host build contract product version {} does not match requested {version}",
            display(product)
        ));
    }
    let abi = item(runtime, "skippy_abi")?;
    let required = item(contract, "skippy_abi")?;
    if abi != required {
        return Err(format!(
            "ValueError: native runtime {} ABI {} does not match host-required ABI {}",
            display(id),
            display(abi),
            display(required)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_runtime_release_preserves_exact_host_abi_requirement() {
        let contract = Json::parse(br#"{"schema_version":1,"product_version":"2.0.0","runtime_release":"1.0.0","skippy_abi":"7"}"#).unwrap();
        let runtime = Json::parse(br#"{"release_version":"9.0.0","skippy_abi":"7"}"#).unwrap();
        let id = Json::String("producer-runtime".to_owned());
        assert!(validate(&contract, &runtime, &id, "2.0.0").is_ok());
        assert!(
            validate(&contract, &runtime, &id, "3.0.0")
                .unwrap_err()
                .contains("product version")
        );
        let incompatible = Json::parse(br#"{"release_version":"9.0.0","skippy_abi":"8"}"#).unwrap();
        assert!(
            validate(&contract, &incompatible, &id, "2.0.0")
                .unwrap_err()
                .contains("host-required ABI")
        );
    }
}

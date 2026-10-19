//! Record and admit the accelerator used by the native AgentX runner.
use super::{Prepared, command, write};
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn capture(input: &Prepared) -> DynResult<()> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let (executable, args) = probe(os, arch)?;
    let spec = command::spec(&input.root, executable, &command::strings(args))?;
    let log = input.output.join("hardware");
    command::run(&spec, 60, &log)?;
    let output = std::fs::read_to_string(log.with_extension("stdout.log"))?;
    let backend = validate(os, &output)?;
    write(
        &input.output.join("hardware.json"),
        &json!({"os":os,"arch":arch,"backend":backend}),
    )
}

fn probe(os: &str, arch: &str) -> DynResult<(&'static str, &'static [&'static str])> {
    match (os, arch) {
        ("linux", "aarch64") => Ok((
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
        )),
        ("macos", "aarch64") => Ok((
            "/usr/sbin/system_profiler",
            &["SPDisplaysDataType", "-json"],
        )),
        _ => Err("AgentX requires a Linux aarch64 DGX Spark or an Apple Silicon Mac".into()),
    }
}

fn validate(os: &str, output: &str) -> DynResult<&'static str> {
    if os == "linux" && output.contains("GB10") {
        return Ok("cuda");
    }
    if os == "macos" {
        let report: Value = serde_json::from_str(output)?;
        let supported = report["SPDisplaysDataType"]
            .as_array()
            .is_some_and(|gpus| gpus.iter().any(apple_metal));
        if supported {
            return Ok("metal");
        }
    }
    Err("AgentX requires an NVIDIA GB10 or an Apple GPU with Metal support".into())
}

fn apple_metal(gpu: &Value) -> bool {
    gpu["spdisplays_vendor"] == "sppci_vendor_Apple"
        && ["spdisplays_mtlgpufamilysupport", "spdisplays_metal"]
            .iter()
            .any(|field| {
                gpu[field].as_str().is_some_and(|value| {
                    value == "spdisplays_supported"
                        || value
                            .strip_prefix("spdisplays_metal")
                            .and_then(|version| version.parse::<u32>().ok())
                            .is_some_and(|version| version > 0)
                })
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_recorded_apple_metal_and_spark_reports() {
        for field in ["spdisplays_mtlgpufamilysupport", "spdisplays_metal"] {
            let report = json!({"SPDisplaysDataType":[{
                "spdisplays_vendor":"sppci_vendor_Apple",field:"spdisplays_metal4"}]});
            assert_eq!(validate("macos", &report.to_string()).unwrap(), "metal");
        }
        assert_eq!(validate("linux", "NVIDIA GB10, [N/A]").unwrap(), "cuda");
        assert!(probe("macos", "aarch64").is_ok());
        assert!(probe("linux", "aarch64").is_ok());
    }

    #[test]
    fn rejects_unsupported_devices_and_missing_accelerator_evidence() {
        for (os, report) in [
            ("macos", "{}"),
            (
                "macos",
                r#"{"SPDisplaysDataType":[{"spdisplays_vendor":"sppci_vendor_Apple"}]}"#,
            ),
            ("linux", "NVIDIA A100"),
            (
                "macos",
                r#"{"SPDisplaysDataType":[{"spdisplays_vendor":"sppci_vendor_Apple","spdisplays_metal":"spdisplays_metal_unsupported"}]}"#,
            ),
            (
                "macos",
                r#"{"SPDisplaysDataType":[{"spdisplays_vendor":"Intel","spdisplays_metal":"spdisplays_metal4"}]}"#,
            ),
        ] {
            assert!(validate(os, report).is_err());
        }
        assert!(probe("linux", "x86_64").is_err());
        assert!(probe("windows", "aarch64").is_err());
    }
}

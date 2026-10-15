//! `qualify availability`: materialize the closed hardware policy for one
//! planned core row. The product source and plan are identities, not
//! authorities for runner selection.

use super::contract;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::path::Path;

const GRAMMAR: Grammar = Grammar {
    usage: "qualify availability --source-sha <sha> --plan-digest <digest> --row-id <row> --output <file>",
    values: &["--source-sha", "--plan-digest", "--row-id", "--output"],
    flags: &[],
};

/// Rows the protected hosted graph always provides.
const ALWAYS_AVAILABLE: [(&str, &str); 3] = [
    ("linux-cpu", "linux-hosted"),
    ("macos-metal", "macos-hosted"),
    ("windows-cpu", "windows-hosted"),
];

/// Accelerator rows, available only when the protected enablement variable is
/// exactly `true` (case-sensitive), so an absent GPU runner is recorded as
/// hardware-unavailable rather than silently downgraded or faked on CPU.
const CONDITIONAL: [(&str, &str, &str); 3] = [
    (
        "linux-cuda",
        "gpu-nvidia",
        "MESH_CUDA_INFERENCE_RUNNER_ENABLED",
    ),
    (
        "linux-rocm",
        "gpu-amd",
        "MESH_ROCM_INFERENCE_RUNNER_ENABLED",
    ),
    (
        "linux-vulkan",
        "gpu-nvidia",
        "MESH_VULKAN_INFERENCE_RUNNER_ENABLED",
    ),
];

const UNAVAILABLE: [(&str, &str); 3] = [
    (
        "windows-cuda",
        "no approved ephemeral Windows CUDA inference runner",
    ),
    (
        "windows-rocm",
        "no approved ephemeral Windows ROCm inference runner",
    ),
    (
        "windows-vulkan",
        "no approved ephemeral Windows Vulkan inference runner",
    ),
];

fn availability(
    root: &Path,
    source_sha: &str,
    plan_digest: &str,
    row_id: &str,
    enablement: &dyn Fn(&str) -> String,
) -> Result<serde_json::Value, String> {
    contract::require(contract::is_git_sha(source_sha), "invalid source SHA")?;
    contract::require(contract::is_sha256(plan_digest), "invalid plan digest")?;
    contract::catalog_row(root, row_id)?;
    let mut result = serde_json::json!({
        "schema_version": 1,
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "row_id": row_id,
        "policy_source": "protected-ci",
    });
    if let Some((_, runner)) = ALWAYS_AVAILABLE.iter().find(|(row, _)| *row == row_id) {
        result["state"] = "available".into();
        result["runner"] = (*runner).into();
    } else if let Some((_, runner, variable)) = CONDITIONAL.iter().find(|(row, ..)| *row == row_id)
    {
        if enablement(variable) == "true" {
            result["state"] = "available".into();
            result["runner"] = (*runner).into();
        } else {
            result["state"] = "hardware-unavailable".into();
            result["reason"] = format!("{variable} is not exactly true").into();
        }
    } else if let Some((_, reason)) = UNAVAILABLE.iter().find(|(row, _)| *row == row_id) {
        result["state"] = "hardware-unavailable".into();
        result["reason"] = (*reason).into();
    } else {
        return Err(format!("unknown core row: {row_id}"));
    }
    Ok(result)
}

pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    let missing = GRAMMAR
        .values
        .iter()
        .filter(|name| parsed.last(name).is_none())
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return GRAMMAR.error(&format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error(&format!(
            "unrecognized arguments: {}",
            parsed.positionals.join(" ")
        ));
    }
    let enablement = |name: &str| std::env::var(name).unwrap_or_default();
    let output = std::path::PathBuf::from(parsed.last("--output").unwrap_or_default());
    let result = availability(
        root,
        parsed.last("--source-sha").unwrap_or_default(),
        parsed.last("--plan-digest").unwrap_or_default(),
        parsed.last("--row-id").unwrap_or_default(),
        &enablement,
    )
    .and_then(|result| {
        let text = contract::render(&result)?;
        contract::write_atomic(&output, text.as_bytes())
    });
    match result {
        Ok(()) => CheckReport::success(String::new()),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("Skippy availability rejected: {error}\n"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::availability;
    use super::contract;

    fn root() -> tempfile::TempDir {
        let temp = tempfile::tempdir().unwrap();
        let ci = temp.path().join("ci");
        std::fs::create_dir_all(&ci).unwrap();
        let rows: Vec<serde_json::Value> = contract::CORE_ROWS
            .iter()
            .map(|id| serde_json::json!({"id": id}))
            .collect();
        std::fs::write(
            ci.join("slices.yml"),
            serde_json::json!({"runtime_rows": rows}).to_string(),
        )
        .unwrap();
        temp
    }

    fn decide(
        root: &std::path::Path,
        row: &str,
        cuda: &str,
        rocm: &str,
        vulkan: &str,
    ) -> serde_json::Value {
        availability(
            root,
            &"a".repeat(40),
            &"b".repeat(64),
            row,
            &|name| match name {
                "MESH_CUDA_INFERENCE_RUNNER_ENABLED" => cuda.to_owned(),
                "MESH_ROCM_INFERENCE_RUNNER_ENABLED" => rocm.to_owned(),
                _ => vulkan.to_owned(),
            },
        )
        .expect("availability")
    }

    #[test]
    fn hosted_rows_are_always_available() {
        let root = root();
        for (row, runner) in [
            ("linux-cpu", "linux-hosted"),
            ("macos-metal", "macos-hosted"),
            ("windows-cpu", "windows-hosted"),
        ] {
            let result = decide(root.path(), row, "", "", "");
            assert_eq!(result["state"], "available");
            assert_eq!(result["runner"], runner);
        }
    }

    #[test]
    fn accelerator_rows_require_exact_lowercase_true() {
        let root = root();
        let enabled = decide(root.path(), "linux-cuda", "true", "", "");
        assert_eq!(enabled["state"], "available");
        assert_eq!(enabled["runner"], "gpu-nvidia");
        for value in ["", "TRUE", "True", "1", "yes"] {
            let result = decide(root.path(), "linux-cuda", value, "", "");
            assert_eq!(result["state"], "hardware-unavailable", "{value}");
            assert_eq!(
                result["reason"],
                "MESH_CUDA_INFERENCE_RUNNER_ENABLED is not exactly true"
            );
        }
        assert_eq!(
            decide(root.path(), "linux-rocm", "", "true", "")["state"],
            "available"
        );
        assert_eq!(
            decide(root.path(), "linux-rocm", "", "true", "")["runner"],
            "gpu-amd"
        );
        assert_eq!(
            decide(root.path(), "linux-vulkan", "", "", "true")["runner"],
            "gpu-nvidia"
        );
    }

    #[test]
    fn windows_accelerators_never_queue() {
        let root = root();
        for (row, reason) in [
            (
                "windows-cuda",
                "no approved ephemeral Windows CUDA inference runner",
            ),
            (
                "windows-rocm",
                "no approved ephemeral Windows ROCm inference runner",
            ),
            (
                "windows-vulkan",
                "no approved ephemeral Windows Vulkan inference runner",
            ),
        ] {
            let result = decide(root.path(), row, "true", "true", "true");
            assert_eq!(result["state"], "hardware-unavailable");
            assert_eq!(result["reason"], reason);
        }
    }

    #[test]
    fn identity_and_row_are_closed() {
        let root = root();
        assert!(
            availability(root.path(), "bad", &"b".repeat(64), "linux-cpu", &|_| {
                String::new()
            })
            .is_err()
        );
        assert!(
            availability(root.path(), &"a".repeat(40), "bad", "linux-cpu", &|_| {
                String::new()
            })
            .is_err()
        );
        assert!(
            availability(
                root.path(),
                &"a".repeat(40),
                &"b".repeat(64),
                "plan9-cpu",
                &|_| String::new()
            )
            .is_err()
        );
    }
}

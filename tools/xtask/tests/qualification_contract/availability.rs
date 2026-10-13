//! `qualify availability`: closed policy per core row.
use super::support::{self, Checkout, xtool};

fn availability(root: &std::path::Path, row: &str, env: &[(&str, &str)]) -> std::process::Output {
    let output = root.join("availability.json");
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .current_dir(root)
        .args([
            "qualify",
            "availability",
            "--source-sha",
            support::SOURCE_SHA,
            "--plan-digest",
            support::PLAN_DIGEST,
            "--row-id",
            row,
            "--output",
            output.to_str().unwrap(),
        ])
        .env_remove("MESH_CUDA_INFERENCE_RUNNER_ENABLED")
        .env_remove("MESH_ROCM_INFERENCE_RUNNER_ENABLED")
        .env_remove("MESH_VULKAN_INFERENCE_RUNNER_ENABLED");
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().unwrap()
}

#[test]
fn hosted_rows_are_always_available() {
    let checkout = Checkout::new();
    checkout.write_full_catalog();
    for (row, runner) in [
        ("linux-cpu", "linux-hosted"),
        ("macos-metal", "macos-hosted"),
        ("windows-cpu", "windows-hosted"),
    ] {
        let result = availability(checkout.root(), row, &[]);
        assert!(result.status.success(), "{}", support::stderr(&result));
        let record: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkout.root().join("availability.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(record["state"], "available");
        assert_eq!(record["runner"], runner);
        assert_eq!(record["policy_source"], "protected-ci");
        assert_eq!(record["row_id"], row);
    }
}

#[test]
fn accelerator_rows_require_exact_lowercase_true() {
    let checkout = Checkout::new();
    checkout.write_full_catalog();
    let root = checkout.root();
    let unavailable = availability(root, "linux-cuda", &[]);
    assert!(unavailable.status.success());
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("availability.json")).unwrap()).unwrap();
    assert_eq!(record["state"], "hardware-unavailable");
    assert_eq!(
        record["reason"],
        "MESH_CUDA_INFERENCE_RUNNER_ENABLED is not exactly true"
    );
    for value in ["TRUE", "True", "1"] {
        let result = availability(
            root,
            "linux-cuda",
            &[("MESH_CUDA_INFERENCE_RUNNER_ENABLED", value)],
        );
        let record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("availability.json")).unwrap())
                .unwrap();
        assert_eq!(record["state"], "hardware-unavailable", "{value}");
    }
    let enabled = availability(
        root,
        "linux-cuda",
        &[("MESH_CUDA_INFERENCE_RUNNER_ENABLED", "true")],
    );
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("availability.json")).unwrap()).unwrap();
    assert_eq!(record["state"], "available");
    assert_eq!(record["runner"], "gpu-nvidia");
    let rocm = availability(
        root,
        "linux-rocm",
        &[("MESH_ROCM_INFERENCE_RUNNER_ENABLED", "true")],
    );
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("availability.json")).unwrap()).unwrap();
    assert_eq!(record["runner"], "gpu-amd");
}

#[test]
fn windows_accelerators_never_queue() {
    let checkout = Checkout::new();
    checkout.write_full_catalog();
    let result = availability(
        checkout.root(),
        "windows-vulkan",
        &[("MESH_VULKAN_INFERENCE_RUNNER_ENABLED", "true")],
    );
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkout.root().join("availability.json")).unwrap())
            .unwrap();
    assert_eq!(record["state"], "hardware-unavailable");
    assert_eq!(
        record["reason"],
        "no approved ephemeral Windows Vulkan inference runner"
    );
}

#[test]
fn catalog_drift_fails_closed() {
    let checkout = Checkout::new();
    checkout.write_catalog(&[
        "linux-cpu",
        "linux-cuda",
        "linux-rocm",
        "linux-vulkan",
        "macos-metal",
        "windows-cpu",
        "windows-cuda",
        "windows-rocm",
    ]);
    let result = availability(checkout.root(), "linux-cpu", &[]);
    assert!(!result.status.success());
    assert!(
        support::stderr(&result).contains("catalog drift"),
        "{}",
        support::stderr(&result)
    );
}

#[test]
fn identity_and_row_are_closed() {
    let checkout = Checkout::new();
    checkout.write_full_catalog();
    let root = checkout.root();
    let bad_sha = xtool(
        root,
        &[
            "qualify",
            "availability",
            "--source-sha",
            "not-a-sha",
            "--plan-digest",
            support::PLAN_DIGEST,
            "--row-id",
            "linux-cpu",
            "--output",
            root.join("o.json").to_str().unwrap(),
        ],
    );
    assert!(!bad_sha.status.success());
    let unknown_row = availability(root, "plan9-cpu", &[]);
    assert!(!unknown_row.status.success());
    assert!(support::stderr(&unknown_row).contains("unknown core row"));
}

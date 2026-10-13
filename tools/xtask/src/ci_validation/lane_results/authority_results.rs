//! Required native authority infrastructure and same-run product-source automation producers
//! follow the selected callers.
use super::results::Lane;

fn includes(jobs: &[&str], consumers: &[&str]) -> bool {
    consumers.iter().any(|consumer| jobs.contains(consumer))
}

pub(super) fn extend(lane: Lane, jobs: &mut Vec<&'static str>) {
    let linux = match lane {
        Lane::Quality => includes(jobs, &["quality"]),
        Lane::Website => includes(jobs, &["web"]),
        Lane::Linux => includes(
            jobs,
            &[
                "ui_artifact",
                "static_abi",
                "rust_tests",
                "hosts",
                "native_runtimes",
                "runtime_product",
                "kotlin_sdk_input",
            ],
        ),
        Lane::Macos | Lane::Windows => includes(jobs, &["ui_artifact"]),
    };
    let macos = lane == Lane::Macos
        && includes(
            jobs,
            &[
                "hosts",
                "native_runtimes",
                "runtime_product",
                "platform_checks",
                "swift_sdk_input",
            ],
        );
    let windows = lane == Lane::Windows
        && includes(
            jobs,
            &[
                "hosts",
                "native_runtimes",
                "runtime_product",
                "platform_checks",
            ],
        );
    let product_linux = lane == Lane::Linux
        && includes(
            jobs,
            &[
                "static_abi",
                "rust_tests",
                "hosts",
                "native_runtimes",
                "runtime_product",
            ],
        );
    let product_native = matches!(lane, Lane::Macos | Lane::Windows)
        && includes(jobs, &["hosts", "native_runtimes", "runtime_product"]);
    for (required, job) in [
        (linux || macos || windows, "authority_source"),
        (linux, "authority_linux_x64"),
        (macos, "authority_macos_arm64"),
        (windows, "authority_windows_x64"),
        (product_linux, "product_automation_linux_x64"),
        (
            product_native && lane == Lane::Macos,
            "product_automation_macos_arm64",
        ),
        (
            product_native && lane == Lane::Windows,
            "product_automation_windows_x64",
        ),
    ] {
        if required && !jobs.contains(&job) {
            jobs.push(job);
        }
    }
}

#[test]
fn every_audited_topic_requires_its_native_producer_and_protected_source() {
    for (lane, consumer, producer) in [
        (Lane::Quality, "quality", "authority_linux_x64"),
        (Lane::Website, "web", "authority_linux_x64"),
        (Lane::Macos, "platform_checks", "authority_macos_arm64"),
        (Lane::Windows, "platform_checks", "authority_windows_x64"),
    ] {
        let mut jobs = vec![consumer];
        extend(lane, &mut jobs);
        assert_eq!(jobs, [consumer, "authority_source", producer]);
    }
    let mut jobs = vec!["rust_tests"];
    extend(Lane::Linux, &mut jobs);
    assert_eq!(
        jobs,
        [
            "rust_tests",
            "authority_source",
            "authority_linux_x64",
            "product_automation_linux_x64"
        ]
    );
}

#[test]
fn product_source_automation_follows_only_the_jobs_that_restore_it() {
    for (lane, consumer, producer) in [
        (Lane::Linux, "static_abi", "product_automation_linux_x64"),
        (
            Lane::Linux,
            "native_runtimes",
            "product_automation_linux_x64",
        ),
        (
            Lane::Macos,
            "native_runtimes",
            "product_automation_macos_arm64",
        ),
        (
            Lane::Windows,
            "runtime_product",
            "product_automation_windows_x64",
        ),
    ] {
        let mut jobs = vec![consumer];
        extend(lane, &mut jobs);
        assert!(jobs.contains(&producer), "{consumer}: {jobs:?}");
    }
    for (lane, consumer) in [
        (Lane::Linux, "sdk"),
        (Lane::Linux, "product_smoke"),
        (Lane::Macos, "platform_checks"),
        (Lane::Windows, "platform_checks"),
        (Lane::Quality, "quality"),
        (Lane::Website, "web"),
    ] {
        let mut jobs = vec![consumer];
        extend(lane, &mut jobs);
        assert!(
            !jobs
                .iter()
                .any(|job| job.starts_with("product_automation_")),
            "{consumer}: {jobs:?}"
        );
    }
}

#[test]
fn macos_and_windows_ui_artifacts_use_native_linux_automation_independently_of_product_target() {
    for (lane, native) in [
        (Lane::Macos, "authority_macos_arm64"),
        (Lane::Windows, "authority_windows_x64"),
    ] {
        let mut jobs = vec!["ui_artifact", "hosts"];
        extend(lane, &mut jobs);
        let product = format!("product_automation_{}", &native["authority_".len()..]);
        assert_eq!(
            jobs,
            [
                "ui_artifact",
                "hosts",
                "authority_source",
                "authority_linux_x64",
                native,
                product.as_str()
            ]
        );
        assert!(!jobs.contains(&"authority_linux_arm64"));
    }
}

#[test]
fn no_audit_caller_adds_no_authority_jobs_and_runner_contract_remains_independent() {
    for lane in Lane::ALL {
        let mut jobs = Vec::new();
        extend(lane, &mut jobs);
        assert!(jobs.is_empty());
    }
    let mut jobs = vec!["runner_contract"];
    extend(Lane::Quality, &mut jobs);
    assert_eq!(jobs, ["runner_contract"]);
}

#[test]
fn shared_audit_consumers_add_each_required_native_producer_once() {
    let mut jobs = vec![
        "ui_artifact",
        "hosts",
        "native_runtimes",
        "runtime_product",
        "kotlin_sdk_input",
    ];
    extend(Lane::Linux, &mut jobs);
    let first = jobs.clone();
    extend(Lane::Linux, &mut jobs);
    assert_eq!(jobs, first);
    assert_eq!(
        jobs.iter()
            .filter(|job| **job == "authority_source")
            .count(),
        1
    );
    assert_eq!(
        jobs.iter()
            .filter(|job| **job == "authority_linux_x64")
            .count(),
        1
    );
    assert_eq!(
        jobs.iter()
            .filter(|job| **job == "product_automation_linux_x64")
            .count(),
        1
    );
}

//! Required native authority infrastructure follows the selected audit callers.
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
    for (required, job) in [
        (linux || macos || windows, "authority_source"),
        (linux, "authority_linux_x64"),
        (macos, "authority_macos_arm64"),
        (windows, "authority_windows_x64"),
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
        (Lane::Linux, "rust_tests", "authority_linux_x64"),
        (Lane::Macos, "platform_checks", "authority_macos_arm64"),
        (Lane::Windows, "platform_checks", "authority_windows_x64"),
    ] {
        let mut jobs = vec![consumer];
        extend(lane, &mut jobs);
        assert_eq!(jobs, [consumer, "authority_source", producer]);
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
        assert_eq!(
            jobs,
            [
                "ui_artifact",
                "hosts",
                "authority_source",
                "authority_linux_x64",
                native
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
}

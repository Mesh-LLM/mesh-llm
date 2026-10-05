use super::{document, input, job, text};
use std::collections::BTreeSet;

#[test]
fn artifact_platform_products_depend_only_on_matching_host_and_runtime_producers() {
    for (platform, authority) in [
        ("linux", "linux_x64"),
        ("macos", "macos_arm64"),
        ("windows", "windows_x64"),
    ] {
        let document = document(&format!("ci-{platform}-lane.yml"));
        let product = job(&document, "runtime_product");
        let needs = product
            .get("needs")
            .unwrap()
            .list()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let authority = format!("authority_{authority}");
        assert_eq!(
            needs,
            BTreeSet::from(["hosts", "native_runtimes", authority.as_str()])
        );
        assert_eq!(
            text(product, "uses"),
            Some(format!("./.github/workflows/ci-{platform}-product-slice.yml").as_str())
        );
        for producer in ["hosts", "native_runtimes"] {
            let uses = text(job(&document, producer), "uses").unwrap();
            assert!(uses.starts_with(&format!("./.github/workflows/ci-{platform}-")));
        }
    }
}

#[test]
fn artifact_sdk_producers_keep_parallel_start_and_static_abi_identity() {
    let linux = document("ci-linux-lane.yml");
    let kotlin = job(&linux, "kotlin_sdk_input");
    assert_eq!(
        kotlin.get("needs").unwrap().list(),
        ["static_abi", "authority_linux_x64"]
    );
    assert_eq!(
        job(&linux, "sdk").get("needs").unwrap().list(),
        ["runtime_product", "kotlin_sdk_input"]
    );
    let macos = document("ci-macos-lane.yml");
    let swift = job(&macos, "swift_sdk_input");
    assert_eq!(
        swift.get("needs").unwrap().list(),
        ["authority_macos_arm64"]
    );
    assert_eq!(
        input(swift, "max_parallel"),
        Some("${{ fromJson(inputs.lane_plan_json).budgets.macos_max_parallel }}")
    );
    assert_eq!(
        input(swift, "fail_fast"),
        Some("${{ inputs.original_event_name == 'pull_request' }}")
    );
    assert_eq!(
        job(&macos, "sdk").get("needs").unwrap().list(),
        ["validate_plan", "runtime_product", "swift_sdk_input"]
    );
    let name = "ci-static-abi-${{ github.run_id }}";
    assert_eq!(
        input(job(&linux, "static_abi"), "artifact_name"),
        Some(name)
    );
    for consumer in ["rust_tests", "kotlin_sdk_input"] {
        assert_eq!(
            input(job(&linux, consumer), "static_abi_artifact_name"),
            Some(name)
        );
    }
    assert_eq!(
        input(job(&linux, "rust_tests"), "static_abi_toolchain_epoch"),
        Some("${{ needs.static_abi.outputs.toolchain_epoch }}")
    );
}

#[test]
fn artifact_lanes_reuse_the_existing_typed_slice_catalog() {
    let used = ["quality", "website", "linux", "macos", "windows"]
        .into_iter()
        .flat_map(|lane| {
            let document = document(&format!("ci-{lane}-lane.yml"));
            document
                .get("jobs")
                .unwrap()
                .entries()
                .iter()
                .filter_map(|(_, job)| text(job, "uses").map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    for slice in [
        "quality",
        "web",
        "ui-artifact",
        "rust-tests",
        "linux-host",
        "macos-host",
        "windows-host",
        "linux-runtime",
        "linux-product",
        "macos-runtime",
        "macos-product",
        "windows-runtime",
        "windows-product",
        "platform-checks",
        "linux-product-smoke",
        "macos-product-smoke",
        "linux-sdk",
        "macos-sdk",
    ] {
        assert!(
            used.contains(&format!("./.github/workflows/ci-{slice}-slice.yml")),
            "slice {slice}"
        );
    }
}

#[test]
fn artifact_static_abi_epoch_flows_from_producer_to_both_rust_consumers() {
    let producer = document("static-abi-artifact.yml");
    let output = producer
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("outputs")
        .unwrap()
        .get("toolchain_epoch")
        .unwrap();
    assert_eq!(
        text(output, "value"),
        Some("${{ jobs.static_abi_artifact.outputs.toolchain_epoch }}")
    );
    let rust = document("ci-rust-tests-slice.yml");
    for consumer in ["rust_tests", "safetensors_runtime_smoke"] {
        let resolver = super::named(job(&rust, consumer), "Resolve static ABI toolchain epoch");
        assert_eq!(
            input(resolver, "pinned_epoch"),
            Some("${{ inputs.static_abi_toolchain_epoch }}")
        );
    }
}

#[test]
fn artifact_main_compatibility_filename_is_inert_and_has_no_event_ingress() {
    let compatibility = document("ci.yml");
    let events = compatibility
        .get("on")
        .unwrap()
        .entries()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(events, ["workflow_call"]);
    let jobs = compatibility.get("jobs").unwrap().entries();
    assert_eq!(jobs.len(), 1);
    assert!(jobs[0].1.get("uses").is_none());
    let steps = super::steps(&jobs[0].1);
    assert_eq!(steps.len(), 1);
    assert!(text(&steps[0], "run").unwrap().starts_with("echo "));
    assert!(steps[0].get("uses").is_none());
}

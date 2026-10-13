//! Parsed producer/caller policy declarations; local source proof only.
use super::{Node, document, input, job, named, source, steps, text};
const LANES: [&str; 5] = ["quality", "website", "linux", "macos", "windows"];
fn call_inputs(doc: &Node) -> &Node {
    doc.get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("inputs")
        .unwrap()
}
fn local_workflow(uses: &str) -> Option<&str> {
    uses.strip_prefix("./.github/workflows/")
}
#[test]
fn graph_entrypoint_plans_are_single_and_pr_cancellation_keys_are_independent() {
    let mut groups = std::collections::BTreeSet::new();
    for lane in LANES {
        for prefix in ["pr", "main"] {
            let doc = document(&format!("{prefix}_{lane}.yml"));
            let plan = job(&doc, "plan");
            let calls = steps(plan)
                .iter()
                .filter(|s| text(s, "uses") == Some("./.github/actions/plan-ci"))
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            let native = job(&doc, "lane");
            assert_eq!(
                text(native, "uses"),
                Some(
                    if prefix == "pr" {
                        format!("Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main")
                    } else {
                        format!("./.github/workflows/ci-{lane}-lane.yml")
                    }
                    .as_str()
                )
            );
            assert!(
                !super::source(&format!("{prefix}_{lane}.yml")).contains("createWorkflowDispatch")
            );
            if prefix == "pr" {
                let concurrency = doc.get("concurrency").unwrap();
                let group = text(concurrency, "group").unwrap();
                assert!(group.ends_with("${{ github.event.pull_request.number }}"));
                assert!(groups.insert(group.to_owned()));
                assert_eq!(
                    input(native, "supersession_key"),
                    Some("pr-${{ github.event.pull_request.number }}")
                );
            }
        }
    }
    let controller = document("ci-control.yml");
    assert_eq!(
        steps(job(&controller, "plan"))
            .iter()
            .filter(|s| text(s, "uses") == Some("./.github/actions/plan-ci"))
            .count(),
        1
    );
    assert!(controller.get("concurrency").is_some());
    let shim = document("ci.yml");
    assert!(shim.get("concurrency").is_none());
}
#[test]
fn graph_lane_source_identity_and_protected_reporting_follow_typed_consumers() {
    for lane in LANES {
        let doc = document(&format!("ci-{lane}-lane.yml"));
        let triggers = doc.get("on").unwrap();
        assert!(triggers.get("workflow_call").is_some());
        assert!(triggers.get("workflow_dispatch").is_some());
        assert!(call_inputs(&doc).get("lane_plan_json").is_some());
        for (_, caller) in doc.get("jobs").unwrap().entries() {
            if let Some(file) = text(caller, "uses").and_then(local_workflow) {
                let target = document(file);
                if call_inputs(&target).get("source_sha").is_some() {
                    assert_eq!(
                        input(caller, "source_sha"),
                        Some("${{ inputs.source_sha }}")
                    );
                }
            }
        }
        let summary = job(&doc, "summary");
        let report = steps(summary)
            .iter()
            .filter(|s| text(s, "uses") == Some("./.github/actions/report-ci-lane"))
            .collect::<Vec<_>>();
        assert_eq!(report.len(), 1);
        assert_eq!(
            input(report[0], "source_sha"),
            Some("${{ inputs.source_sha }}")
        );
        let checkout = steps(summary)
            .iter()
            .filter(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@")))
            .collect::<Vec<_>>();
        assert_eq!(checkout.len(), 1);
        assert_eq!(
            input(checkout[0], "ref"),
            Some("${{ github.event.repository.default_branch }}")
        );
        let concurrency = doc.get("concurrency").unwrap();
        assert!(
            text(concurrency, "group")
                .unwrap()
                .contains("inputs.supersession_key || inputs.source_sha")
        );
    }
}
#[test]
fn graph_runner_contract_and_original_event_follow_all_current_typed_build_calls() {
    for lane in LANES {
        let doc = document(&format!("ci-{lane}-lane.yml"));
        for (_, caller) in doc.get("jobs").unwrap().entries() {
            if let Some(file) = text(caller, "uses").and_then(local_workflow) {
                let target = document(file);
                let inputs = call_inputs(&target);
                if inputs.get("force_hosted").is_some() {
                    assert_eq!(
                        input(caller, "force_hosted"),
                        Some(
                            "${{ fromJson(inputs.lane_plan_json).signals.runner_contract_required }}"
                        )
                    );
                }
                if inputs.get("original_event_name").is_some() {
                    assert_eq!(
                        input(caller, "original_event_name"),
                        Some("${{ inputs.original_event_name }}")
                    );
                }
            }
        }
    }
    for file in ["static-abi-artifact.yml", "native-sdk-artifact.yml"] {
        let doc = document(file);
        let selectors = doc
            .get("jobs")
            .unwrap()
            .entries()
            .iter()
            .filter_map(|(_, j)| j.get("steps").map(|_| j))
            .flat_map(steps)
            .filter(|s| text(s, "uses") == Some("./.github/actions/select-ci-runners"))
            .collect::<Vec<_>>();
        assert!(!selectors.is_empty());
        for selector in selectors {
            assert_eq!(
                input(selector, "original_event_name"),
                Some("${{ inputs.original_event_name }}")
            );
            assert_eq!(
                input(selector, "repository"),
                Some("${{ github.repository }}")
            );
            assert_eq!(
                input(selector, "head_repository"),
                Some("${{ github.event.pull_request.head.repo.full_name }}")
            );
            assert_eq!(
                input(selector, "depot_pr_enabled"),
                Some("${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}")
            );
        }
    }
    assert!(
        super::support::step(&super::support::action("select-ci-runners"), "run")
            .contains("\"$INPUT_FORCE_HOSTED\" == \"false\"")
    );
}
#[test]
fn graph_critical_matrix_failfast_follows_pr_profile_and_web_control_changes() {
    for platform in ["linux", "macos", "windows"] {
        for component in ["host", "runtime", "product"] {
            let doc = document(&format!("ci-{platform}-{component}-slice.yml"));
            assert!(call_inputs(&doc).get("fail_fast").is_some());
            for (_, j) in doc.get("jobs").unwrap().entries() {
                if let Some(strategy) = j.get("strategy") {
                    assert_eq!(text(strategy, "fail-fast"), Some("${{ inputs.fail_fast }}"));
                }
            }
        }
        let lane = document(&format!("ci-{platform}-lane.yml"));
        for (_, caller) in lane.get("jobs").unwrap().entries() {
            if let Some(file) = text(caller, "uses").and_then(local_workflow) {
                let target = document(file);
                if call_inputs(&target).get("fail_fast").is_some() {
                    assert_eq!(
                        input(caller, "fail_fast"),
                        Some("${{ inputs.original_event_name == 'pull_request' }}")
                    );
                }
            }
        }
    }
    for file in ["ci-rust-tests-slice.yml", "ci-platform-checks-slice.yml"] {
        let doc = document(file);
        assert!(call_inputs(&doc).get("fail_fast").is_some());
        for (_, j) in doc.get("jobs").unwrap().entries() {
            if let Some(strategy) = j.get("strategy") {
                assert_eq!(text(strategy, "fail-fast"), Some("${{ inputs.fail_fast }}"));
            }
        }
    }
    let doc = document("ci-website-lane.yml");
    for signal in ["ui", "website"] {
        assert_eq!(input(job(&doc,"web"),&format!("{signal}_changed")),Some(format!("${{{{ fromJson(inputs.lane_plan_json).signals.{signal}_changed || contains(fromJson(inputs.lane_plan_json).domains, 'ci-control') }}}}").as_str()));
    }
}
#[test]
fn graph_controller_optional_summary_and_trusted_hf_secret_boundaries_are_declared() {
    let controller = source("ci-control.yml");
    assert!(!controller.contains("HF_TOKEN"));
    for matrix in [
        "hosts",
        "runtime_products",
        "rust_tests",
        "smoke",
        "sdk",
        "platform_checks",
    ] {
        assert!(controller.contains(&format!("(.matrices.{matrix} // [])[]")));
    }
    for lane in ["linux", "macos"] {
        let doc = document(&format!("ci-{lane}-lane.yml"));
        for (_, j) in doc.get("jobs").unwrap().entries() {
            if let Some(secrets) = j.get("secrets")
                && let Some(token) = text(secrets, "HF_TOKEN")
            {
                assert_eq!(
                    token,
                    "${{ inputs.original_event_name == 'push' && secrets.HF_TOKEN || '' }}"
                );
            }
        }
    }
    let doc = document("ci-quality-slice.yml");
    for (_, j) in doc.get("jobs").unwrap().entries() {
        if let Some(strategy) = j.get("strategy") {
            assert_eq!(text(strategy, "fail-fast"), Some("false"));
        }
    }
}
#[test]
fn graph_runtime_product_and_macos_consumers_keep_architecture_in_artifact_identity() {
    for platform in ["linux", "macos", "windows"] {
        for component in ["runtime", "product"] {
            let doc = document(&format!("ci-{platform}-{component}-slice.yml"));
            let names = doc
                .get("jobs")
                .unwrap()
                .entries()
                .iter()
                .filter_map(|(_, j)| j.get("steps").map(|_| j))
                .flat_map(steps)
                .filter_map(|s| input(s, "name"))
                .collect::<Vec<_>>();
            let runtime = format!(
                "ci-runtime-{platform}-${{{{ matrix.runtime.architecture }}}}-${{{{ matrix.runtime.backend }}}}"
            );
            assert!(names.contains(&runtime.as_str()));
            if component == "product" {
                let product = format!(
                    "ci-product-{platform}-${{{{ matrix.runtime.architecture }}}}-${{{{ matrix.runtime.backend }}}}"
                );
                assert!(names.contains(&product.as_str()));
            }
        }
    }
    let lane = document("ci-macos-lane.yml");
    for name in ["product_smoke", "sdk"] {
        assert_eq!(
            input(job(&lane, name), "architecture"),
            Some(
                "${{ fromJson(inputs.lane_plan_json).matrices.runtime_products[0].architecture }}"
            )
        );
        assert!(
            job(&lane, name)
                .get("needs")
                .unwrap()
                .list()
                .contains(&"validate_plan")
        );
    }
    let run = text(
        named(
            job(&lane, "validate_plan"),
            "Enforce one consumer architecture",
        ),
        "run",
    )
    .unwrap();
    assert!(run.contains("unique"));
}
#[test]
fn graph_main_base_identity_and_manual_depot_forwarding_remain_owned() {
    for lane in LANES {
        let doc = document(&format!("main_{lane}.yml"));
        let plan = job(&doc, "plan");
        assert_eq!(
            text(plan.get("outputs").unwrap(), "base_sha"),
            Some("${{ steps.identity.outputs.base_sha }}")
        );
        let identity = steps(plan)
            .iter()
            .find(|s| text(s, "id") == Some("identity"))
            .unwrap();
        let run = text(identity, "run").unwrap();
        assert!(run.contains("[[ \"$BASE_SHA\" =~ ^0+$ ]]"));
        assert!(run.contains("echo \"base_sha=\""));
        assert!(run.contains("git cat-file -e \"$BASE_SHA^{commit}\""));
        for uses in [
            "./.github/actions/compute-changes",
            "./.github/actions/plan-ci",
        ] {
            let calls = steps(plan)
                .iter()
                .filter(|s| text(s, "uses") == Some(uses))
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1);
            assert_eq!(
                input(calls[0], "base_sha"),
                Some("${{ steps.identity.outputs.base_sha }}")
            );
        }
    }
    for (lane, consumer) in [
        ("quality", "quality"),
        ("linux", "hosts"),
        ("linux", "native_runtimes"),
    ] {
        let doc = document(&format!("ci-{lane}-lane.yml"));
        assert_eq!(
            input(job(&doc, consumer), "use_depot"),
            Some("${{ inputs.use_depot }}")
        );
    }
    for file in [
        "ci-quality-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
        "static-abi-artifact.yml",
        "native-sdk-artifact.yml",
    ] {
        let doc = document(file);
        assert_eq!(
            text(call_inputs(&doc).get("use_depot").unwrap(), "default"),
            Some("false")
        );
        let mut ordinary_selectors = 0;
        let mut cpu_selectors = 0;
        for (_, j) in doc.get("jobs").unwrap().entries() {
            if j.get("steps").is_some() {
                for s in steps(j) {
                    if text(s, "uses") == Some("./.github/actions/select-ci-runners") {
                        if text(s, "id") == Some("sentinel_policy") {
                            assert_eq!(input(s, "manual_use_depot"), Some("false"));
                            continue;
                        }
                        if text(s, "id") == Some("cpu_policy") {
                            assert_eq!(file, "ci-linux-runtime-slice.yml");
                            assert_eq!(input(s, "force_hosted"), Some("true"));
                            assert!(input(s, "manual_use_depot").is_none());
                            cpu_selectors += 1;
                            continue;
                        }
                        ordinary_selectors += 1;
                        assert_eq!(
                            input(s, "manual_use_depot"),
                            Some("${{ inputs.use_depot }}")
                        );
                    }
                }
            }
        }
        assert_eq!(ordinary_selectors, 1, "{file}");
        assert_eq!(
            cpu_selectors,
            usize::from(file == "ci-linux-runtime-slice.yml")
        );
        if cpu_selectors == 1 {
            cpu_runtime_projection(&doc);
        }
    }
    let doc = document("ci-control.yml");
    let resolve = steps(job(&doc, "plan"))
        .iter()
        .find(|s| text(s, "id") == Some("resolve"))
        .unwrap();
    let script = input(resolve, "script").unwrap();
    assert!(script.contains("context.payload.inputs?.use_depot === true"));
    assert!(script.contains("context.payload.inputs?.use_depot === 'true'"));
    assert!(script.contains("core.setOutput('use_depot', String(useDepot))"));
    assert_eq!(
        text(doc.get("permissions").unwrap(), "contents"),
        Some("read")
    );
    assert!(!source("ci-control.yml").contains("SOURCE_REF"));
}
fn cpu_runtime_projection(doc: &Node) {
    let outputs = job(doc, "runner_policy").get("outputs").unwrap();
    assert_eq!(
        text(outputs, "runner_cpu"),
        Some("${{ steps.cpu_policy.outputs.runner_16 }}")
    );
    assert_eq!(
        text(outputs, "allow_native_github_cache_cpu"),
        Some("${{ steps.cpu_policy.outputs.allow_native_github_cache }}")
    );
    assert_eq!(
        text(job(doc, "linux_runtime"), "runs-on"),
        Some(
            "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16 }}"
        )
    );
}
#[test]
fn graph_product_source_checkouts_bind_declared_identity_and_windows_refuses_fallback() {
    for file in [
        "ci-quality-slice.yml",
        "ci-runner-contract-slice.yml",
        "ci-web-slice.yml",
        "ci-ui-artifact-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-macos-host-slice.yml",
        "ci-windows-host-slice.yml",
        "ci-linux-runtime-slice.yml",
        "ci-linux-product-slice.yml",
        "ci-macos-runtime-slice.yml",
        "ci-macos-product-slice.yml",
        "ci-windows-runtime-slice.yml",
        "ci-windows-product-slice.yml",
        "ci-windows-product-smoke-slice.yml",
        "ci-platform-checks-slice.yml",
        "static-abi-artifact.yml",
        "native-sdk-artifact.yml",
        "swift-sdk-artifact.yml",
        "smoke.yml",
        "scripted-binary-smoke.yml",
        "sdk-smoke.yml",
        "hf-download-smoke.yml",
    ] {
        let doc = document(file);
        assert!(call_inputs(&doc).get("source_sha").is_some());
        let strict = matches!(
            file,
            "ci-windows-runtime-slice.yml" | "ci-windows-product-smoke-slice.yml"
        );
        let expected = if strict {
            "${{ inputs.source_sha }}"
        } else {
            "${{ inputs.source_sha || github.sha }}"
        };
        let bound = doc
            .get("jobs")
            .unwrap()
            .entries()
            .iter()
            .filter_map(|(_, j)| j.get("steps").map(|_| j))
            .flat_map(steps)
            .any(|s| {
                text(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@"))
                    && input(s, "ref") == Some(expected)
            });
        assert!(bound, "{file}");
        if strict {
            assert!(!source(file).contains("inputs.source_sha || github.sha"));
        }
    }
    let doc = document("ci-windows-runtime-slice.yml");
    let validation = named(
        job(&doc, "windows_runtime"),
        "Validate immutable source SHA",
    );
    assert_eq!(
        text(validation.get("env").unwrap(), "SOURCE_SHA"),
        Some("${{ inputs.source_sha }}")
    );
    assert!(
        text(validation, "run")
            .unwrap()
            .contains("if ($env:SOURCE_SHA -notmatch '^[0-9a-f]{40}$')")
    );
}

//! Declared graph intent only; this does not execute a hosted/platform job.
use super::{Node, document, input, job, named, source, steps, support, text};
use std::fs;
fn runs(node: &Node) -> String {
    steps(node)
        .iter()
        .filter_map(|s| text(s, "run"))
        .collect::<Vec<_>>()
        .join("\n")
}
fn requires(run: &str, needles: &[&str]) {
    for needle in needles {
        assert!(run.contains(needle), "missing {needle}");
    }
}
#[test]
fn graph_docs_preserve_five_lane_visibility_and_manual_controller_boundary() {
    let docs = fs::read_to_string(support::root().join("ci/ci.md")).unwrap();
    requires(
        &docs,
        &[
            "The five-way split is a hard CI architecture invariant",
            "optional, non-required diagnostic exception",
            "`dispatched`, with the real work detached",
            "Do not add another all-lanes PR",
            "Do not funnel main pushes through `ci-control.yml`",
        ],
    );
}
#[test]
fn graph_platform_privacy_tests_are_declared_on_windows() {
    let doc = document("ci-platform-checks-slice.yml");
    let jobs = doc.get("jobs").unwrap().entries();
    for (name, method) in [
        (
            "Test Windows log artifact privacy ACL",
            "windows_artifact_paths_have_current_owner_and_exact_user_only_dacl",
        ),
        (
            "Test Windows log SQLite storage ACL",
            "sqlite_root_database_and_sidecars_have_only_current_user_acl",
        ),
    ] {
        let owners = jobs
            .iter()
            .filter_map(|(_, j)| j.get("steps").map(|_| j))
            .filter(|j| steps(j).iter().any(|s| text(s, "name") == Some(name)))
            .collect::<Vec<_>>();
        assert_eq!(owners.len(), 1);
        assert!(runs(owners[0]).contains(method));
        let step = named(owners[0], name);
        assert_eq!(text(step, "shell"), Some("pwsh"));
        assert_eq!(
            text(step, "if"),
            Some("${{ matrix.check.kind == 'log-store' }}")
        );
    }
}
#[test]
fn graph_platform_producers_and_consumers_keep_separate_responsibilities() {
    for platform in ["linux", "macos", "windows"] {
        let lane = document(&format!("ci-{platform}-lane.yml"));
        let product = job(&lane, "runtime_product");
        requires(
            text(product, "if").unwrap(),
            &[
                "needs.hosts.result == 'success'",
                "needs.native_runtimes.result == 'success'",
            ],
        );
        let runtime = job(&lane, "native_runtimes");
        assert!(!runtime.get("needs").unwrap().list().contains(&"hosts"));
        for (component, required, forbidden) in [
            (
                "runtime",
                "./.github/actions/prepare-native-runtime-input",
                "./.github/actions/compose-product-input",
            ),
            (
                "product",
                "./.github/actions/compose-product-input",
                "./.github/actions/prepare-native-runtime-input",
            ),
        ] {
            let doc = document(&format!("ci-{platform}-{component}-slice.yml"));
            let calls = doc
                .get("jobs")
                .unwrap()
                .entries()
                .iter()
                .filter_map(|(_, j)| j.get("steps").map(|_| j))
                .flat_map(steps)
                .filter_map(|s| text(s, "uses"))
                .collect::<Vec<_>>();
            assert!(calls.contains(&required));
            assert!(!calls.contains(&forbidden));
            if component == "product" {
                assert!(
                    !source(&format!("ci-{platform}-{component}-slice.yml"))
                        .contains("cargo build")
                );
            }
            assert!(
                !source(&format!("ci-{platform}-{component}-slice.yml"))
                    .contains("compose_products")
            );
        }
    }
}
#[test]
fn graph_host_artifact_names_and_ui_inputs_are_platform_pure() {
    for (platform, label, prefix, others) in [
        (
            "linux",
            "Linux",
            "ci-host-linux-",
            ["macOS host", "Windows host"],
        ),
        (
            "macos",
            "macOS",
            "ci-host-macos-",
            ["Linux host", "Windows host"],
        ),
        (
            "windows",
            "Windows",
            "ci-host-windows-",
            ["Linux host", "macOS host"],
        ),
    ] {
        let doc = document(&format!("ci-{platform}-host-slice.yml"));
        let jobs = doc.get("jobs").unwrap().entries();
        assert!(jobs.iter().any(|(_, j)| {
            text(j, "name").is_some_and(|n| n.starts_with(&format!("{label} host (")))
        }));
        let all = jobs.iter().flat_map(|(_, j)| steps(j)).collect::<Vec<_>>();
        assert!(
            all.iter()
                .any(|s| input(s, "name").is_some_and(|n| n.starts_with(prefix)))
        );
        assert!(
            all.iter()
                .any(|s| input(s, "name") == Some("${{ inputs.ui_artifact_name }}"))
        );
        for (_, j) in jobs {
            for other in others {
                assert!(!text(j, "name").unwrap_or("").contains(other));
            }
        }
    }
    let windows = source("ci-windows-host-slice.yml");
    assert!(windows.contains("./.github/actions/prepare-windows-host-input"));
    assert!(!windows.contains("scripts/build-windows.ps1"));
}
#[test]
fn graph_smoke_gates_parse_ids_and_catalog_domains_have_consumers() {
    let catalog: serde_json::Value =
        serde_json::from_slice(&fs::read(support::root().join("ci/slices.yml")).unwrap()).unwrap();
    let mut admitted = std::collections::BTreeSet::new();
    let declared = catalog["smoke_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    for platform in ["linux", "macos", "windows"] {
        let doc = document(&format!("ci-{platform}-product-smoke-slice.yml"));
        for (_, j) in doc.get("jobs").unwrap().entries() {
            if let Some(condition) = text(j, "if") {
                assert!(!condition.contains("contains(inputs.smoke_matrix,"));
                let prefix = "contains(fromJson(inputs.smoke_matrix).*.id, '";
                if let Some((_, tail)) = condition.split_once(prefix) {
                    admitted.insert(tail.split_once('\'').unwrap().0.to_owned());
                }
            }
        }
    }
    for ids in catalog["smoke_domain_rows"].as_object().unwrap().values() {
        for id in ids.as_array().unwrap() {
            assert!(admitted.contains(id.as_str().unwrap()));
            assert!(declared.contains(id.as_str().unwrap()));
        }
    }
    for id in [
        "core",
        "two-node-client",
        "two-node-split",
        "core-cuda",
        "model-download",
        "metal-model-load",
    ] {
        assert!(admitted.contains(id));
    }
}
#[test]
fn graph_cold_swift_budget_and_isolated_rust_batch_invocation_remain_declared() {
    let macos = document("ci-macos-lane.yml");
    assert_eq!(
        input(job(&macos, "swift_sdk_input"), "timeout_minutes"),
        Some("90")
    );
    let rust = source("ci-rust-tests-slice.yml");
    requires(
        &rust,
        &["for crate in", "cargo test --locked -p \"$crate\""],
    );
    assert!(!rust.contains("cargo test --locked \"${crate_args[@]}\""));
}
#[test]
fn graph_macos_lld_precedes_native_tools_preparation() {
    let doc = document("ci-macos-runtime-slice.yml");
    let runtime = job(&doc, "macos_runtime");
    let calls = steps(runtime)
        .iter()
        .filter_map(|s| text(s, "uses"))
        .collect::<Vec<_>>();
    let lld = calls
        .iter()
        .position(|s| *s == "./.github/actions/setup-macos-lld")
        .unwrap();
    let prepare = calls
        .iter()
        .position(|s| *s == "./.github/actions/prepare-native-runtime-input")
        .unwrap();
    assert!(lld < prepare);
}
#[test]
fn graph_release_dispatch_permissions_and_supported_node_targets_are_bounded() {
    let doc = document("release.yml");
    assert_eq!(
        doc.get("on")
            .unwrap()
            .entries()
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        ["workflow_dispatch"]
    );
    let grant = doc.get("permissions").unwrap();
    assert_eq!(text(grant, "contents"), Some("read"));
    assert_eq!(text(grant, "packages"), Some("read"));
    for name in ["metadata", "publish"] {
        let grant = job(&doc, name).get("permissions").unwrap();
        assert_eq!(text(grant, "contents"), Some("write"));
        assert_ne!(text(grant, "packages"), Some("write"));
    }
    let node = job(&doc, "build_node_sdk_addon");
    let Node::Seq(rows) = node
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("include")
        .unwrap()
    else {
        panic!("node matrix")
    };
    assert_eq!(
        rows.iter()
            .map(|r| text(r, "target").unwrap())
            .collect::<Vec<_>>(),
        ["darwin-arm64", "linux-arm64", "linux-x64", "win32-x64"]
    );
    assert!(!source("node-sdk-addon-artifact.yml").contains("darwin-x64"));
}
#[test]
fn graph_release_publication_keeps_cancellation_tag_and_token_boundaries() {
    let doc = document("release.yml");
    let publish = job(&doc, "publish");
    let condition = text(publish, "if").unwrap();
    assert!(condition.contains("!cancelled()"));
    assert!(!condition.contains("always()"));
    let pushes = steps(publish)
        .iter()
        .filter(|s| {
            s.get("env")
                .is_some_and(|env| text(env, "GITHUB_TOKEN") == Some("${{ secrets.GITHUB_TOKEN }}"))
        })
        .collect::<Vec<_>>();
    assert_eq!(pushes.len(), 1);
    assert_eq!(text(pushes[0], "name"), Some("Push dispatched release tag"));
    requires(
        text(pushes[0], "run").unwrap(),
        &["git push \"$release_remote\" \"refs/tags/$RELEASE_TAG\""],
    );
    let prepare = named(publish, "Prepare dispatched release tag");
    let run = text(prepare, "run").unwrap();
    requires(run, &["Release tag already exists and cannot be reused"]);
    assert!(!run.contains("git push"));
    assert!(!run.contains("release_remote="));
    assert_eq!(
        input(named(publish, "Publish GitHub release"), "overwrite_files"),
        Some("false")
    );
    for s in steps(publish)
        .iter()
        .filter(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@")))
    {
        assert_eq!(input(s, "persist-credentials"), Some("false"));
    }
}
#[test]
fn graph_packaging_dispatch_checks_write_access_and_excludes_prereleases() {
    let doc = document("release.yml");
    let dispatch = job(&doc, "dispatch_packaging_release");
    requires(
        text(dispatch, "if").unwrap(),
        &[
            "needs.metadata.outputs.prerelease != 'true'",
            "needs.metadata.outputs.skip_gpu_bundles != 'true'",
        ],
    );
    let step = named(dispatch, "Dispatch verified release to mesh-packaging");
    let env = step.get("env").unwrap();
    assert_eq!(
        text(env, "GH_TOKEN"),
        Some("${{ secrets.MESH_AGENT_IMAGES_DISPATCH_TOKEN }}")
    );
    assert_eq!(
        text(env, "TARGET_REPOSITORY"),
        Some("Mesh-LLM/mesh-packaging")
    );
    let run = text(step, "run").unwrap();
    requires(
        run,
        &[
            ".permissions.push // false",
            "dry_run: false",
            "publish_images: true",
            "publish_release_assets: true",
            "publish_npm: true",
        ],
    );
    assert!(
        run.find(".permissions.push // false").unwrap()
            < run
                .find("\"repos/${TARGET_REPOSITORY}/dispatches\"")
                .unwrap()
    );
}

#[test]
fn graph_actual_pr_and_main_event_census_keeps_exact_five_lane_entries() {
    use std::collections::BTreeSet;
    let lanes = ["quality", "website", "linux", "macos", "windows"];
    let expected_pr = lanes
        .map(|l| format!("pr_{l}.yml"))
        .into_iter()
        .collect::<BTreeSet<_>>();
    let exceptions = BTreeSet::from([
        "pr_auto_assign.yml".to_owned(),
        "pr_cleanup.yml".to_owned(),
        "pr_ci_canary.yml".to_owned(),
    ]);
    let expected_main = lanes
        .map(|l| format!("main_{l}.yml"))
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut pr = BTreeSet::new();
    let mut main = BTreeSet::new();
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if !matches!(
            path.extension().and_then(|x| x.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let doc = super::workflow_yaml::parse(&fs::read_to_string(&path).unwrap()).unwrap();
        let trigger = doc.get("on").unwrap();
        let names = match trigger {
            Node::Map(rows) => rows.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            _ => trigger.list(),
        };
        let filename = entry.file_name().to_str().unwrap().to_owned();
        if names
            .iter()
            .any(|n| matches!(*n, "pull_request" | "pull_request_target"))
        {
            pr.insert(filename.clone());
        }
        if filename.starts_with("main_") && names.contains(&"push") {
            main.insert(filename);
        }
    }
    assert!(exceptions.is_subset(&pr));
    assert_eq!(
        pr.difference(&exceptions).cloned().collect::<BTreeSet<_>>(),
        expected_pr
    );
    assert_eq!(main, expected_main);
    for lane in lanes {
        for (prefix, expected) in [
            (
                "pr",
                format!("Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"),
            ),
            ("main", format!("./.github/workflows/ci-{lane}-lane.yml")),
        ] {
            let doc = document(&format!("{prefix}_{lane}.yml"));
            let calls = doc
                .get("jobs")
                .unwrap()
                .entries()
                .iter()
                .filter_map(|(_, j)| text(j, "uses"))
                .collect::<Vec<_>>();
            assert_eq!(calls, [expected]);
            let trigger = doc
                .get("on")
                .unwrap()
                .get(if prefix == "pr" {
                    "pull_request"
                } else {
                    "push"
                })
                .unwrap();
            assert!(trigger.get("paths").is_none());
            assert!(trigger.get("paths-ignore").is_none());
        }
    }
}

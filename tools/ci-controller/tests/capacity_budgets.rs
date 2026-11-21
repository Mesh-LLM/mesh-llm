//! Plans built from the checked-in `ci/slices.yml` keep every selected slice
//! and row while the per-matrix allowances lanes consume stay within each
//! platform's declared per-run worker ceiling.

use ci_controller::lane_results::workflow_yaml;
use ci_controller::plan::{build_for_validation, matrix_budgets::peak_workers};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BASE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn catalog() -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(
        root().join("ci/slices.yml"),
    )?)?)
}

fn plan(
    profile: &str,
    event: &str,
    base: &str,
    changed: &[&str],
) -> Result<Value, Box<dyn std::error::Error>> {
    let input = json!({
        "profile": profile,
        "event_name": event,
        "source_sha": SOURCE,
        "base_sha": base,
        "changed_files": changed,
        "affected_crates": ["mesh-llm"],
        "workspace_packages": [
            {"name": "mesh-llm", "path": "mesh/crates/mesh-llm"},
            {"name": "mesh-llm-host-runtime", "path": "mesh/crates/mesh-llm-host-runtime"},
            {"name": "skippy-runtime", "path": "skippy/crates/skippy-runtime"}
        ]
    });
    build_for_validation(&root(), &serde_json::to_vec(&input)?)
}

fn ids(plan: &Value, matrix: &str) -> Vec<String> {
    plan["matrices"][matrix]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["id"].as_str().map(str::to_owned))
        .collect()
}

fn catalog_ids(catalog: &Value, field: &str) -> Vec<String> {
    catalog[field]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["id"].as_str().map(str::to_owned))
        .collect()
}

fn assert_within_ceilings(plan: &Value) {
    let budgets = &plan["budgets"];
    let ceilings = ["linux", "macos", "windows"].map(|os| {
        budgets[format!("{os}_max_workers")]
            .as_u64()
            .expect("ceiling")
    });
    let peak = peak_workers(plan);
    for (index, os) in ["linux", "macos", "windows"].into_iter().enumerate() {
        assert!(
            peak[index] <= ceilings[index],
            "{} {os}: peak {} exceeds ceiling {}",
            plan["profile"],
            peak[index],
            ceilings[index]
        );
    }
    assert!(peak.iter().sum::<u64>() <= budgets["total_max_workers"].as_u64().expect("total"));
    for (key, allowance) in budgets["matrix_max_parallel"]
        .as_object()
        .expect("allowances")
    {
        let os = key.split('_').next().expect("platform prefix");
        let limit = budgets[format!("{os}_max_parallel")]
            .as_u64()
            .expect("limit");
        let value = allowance.as_u64().expect("integer allowance");
        assert!(
            (1..=limit).contains(&value),
            "{key}={value} outside 1..={limit}"
        );
    }
}

#[test]
fn exhaustive_main_keeps_every_catalog_row_within_the_declared_ceilings() -> TestResult {
    let catalog = catalog()?;
    for profile in ["main", "manual-full"] {
        let event = if profile == "main" {
            "push"
        } else {
            "workflow_dispatch"
        };
        let plan = plan(profile, event, "", &["mesh/crates/mesh-llm/src/lib.rs"])?;
        let all_slices = catalog_ids(&catalog, "slices");
        assert_eq!(
            ids(&plan, "runtime_products"),
            catalog_ids(&catalog, "runtime_rows")
        );
        assert_eq!(
            ids(&plan, "platform_checks"),
            catalog_ids(&catalog, "platform_rows")
        );
        assert_eq!(ids(&plan, "sdk"), catalog_ids(&catalog, "sdk_rows"));
        assert_eq!(ids(&plan, "smoke"), catalog_ids(&catalog, "smoke_rows"));
        assert_eq!(plan["required_slices"], json!(all_slices));
        assert_within_ceilings(&plan);
    }
    Ok(())
}

#[test]
fn exhaustive_main_keeps_todays_concurrency_instead_of_throttling() -> TestResult {
    let plan = plan("main", "push", "", &["mesh/crates/mesh-llm/src/lib.rs"])?;
    let allowances = &plan["budgets"]["matrix_max_parallel"];
    assert_eq!(
        allowances,
        &json!({
            "linux_clippy": 3, "linux_rust_tests": 3, "linux_hosts": 1,
            "macos_hosts": 1, "macos_runtimes": 1, "macos_products": 1,
            "macos_platform_checks": 2, "macos_swift_sdk": 4,
            "windows_hosts": 1, "windows_runtimes": 2, "windows_products": 2,
            "windows_platform_checks": 2
        })
    );
    assert_eq!(peak_workers(&plan), [11, 8, 5]);
    Ok(())
}

#[test]
fn control_plane_pr_overlaps_every_platform_matrix_within_its_ceiling() -> TestResult {
    let plan = plan(
        "pr-ready",
        "pull_request",
        BASE,
        &[".github/workflows/pr_linux.yml"],
    )?;
    let catalog = catalog()?;
    assert_eq!(
        ids(&plan, "runtime_products"),
        catalog_ids(&catalog, "runtime_rows")
    );
    assert_eq!(
        ids(&plan, "platform_checks"),
        catalog_ids(&catalog, "platform_rows")
    );
    assert_within_ceilings(&plan);
    assert_eq!(peak_workers(&plan), [7, 4, 3]);
    Ok(())
}

#[test]
fn every_lane_matrix_consumes_exactly_its_own_allowance() -> TestResult {
    let plan = plan("main", "push", "", &["mesh/crates/mesh-llm/src/lib.rs"])?;
    let emitted = plan["budgets"]["matrix_max_parallel"]
        .as_object()
        .ok_or("allowances")?
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let prefix = "${{ fromJson(inputs.lane_plan_json).budgets.matrix_max_parallel.";
    let mut consumed = Vec::new();
    for lane in ["quality", "linux", "macos", "windows"] {
        let path = root().join(format!(".github/workflows/ci-{lane}-lane.yml"));
        let tree = workflow_yaml::parse(&std::fs::read_to_string(path)?)?;
        for (job, definition) in tree
            .get("jobs")
            .map(|jobs| jobs.entries())
            .unwrap_or_default()
        {
            for (input, value) in definition
                .get("with")
                .map(|with| with.entries())
                .unwrap_or_default()
            {
                if !matches!(input.as_str(), "max_parallel" | "clippy_max_parallel") {
                    continue;
                }
                let text = value.text().unwrap_or_default();
                if text == "1" {
                    assert!(
                        job.starts_with("native_runtime_") || job.starts_with("product_"),
                        "{lane}.{job}"
                    );
                    continue;
                }
                let key = text
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.strip_suffix(" }}"))
                    .unwrap_or_else(|| {
                        panic!("{lane}.{job} max_parallel bypasses its allowance: {text}")
                    });
                consumed.push(key.to_owned());
            }
        }
    }
    consumed.sort();
    let unique = consumed
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        consumed.len(),
        "an allowance is shared by two matrices"
    );
    assert_eq!(unique, emitted);
    Ok(())
}

#[test]
fn ordinary_and_draft_prs_stay_within_their_ceilings() -> TestResult {
    for (profile, changed) in [
        ("pr-ready", "mesh/crates/mesh-llm/src/lib.rs"),
        ("pr-ready", "scripts/build-windows.ps1"),
        ("pr-draft", "mesh/crates/mesh-llm/src/lib.rs"),
    ] {
        let plan = plan(profile, "pull_request", BASE, &[changed])?;
        assert_within_ceilings(&plan);
    }
    Ok(())
}

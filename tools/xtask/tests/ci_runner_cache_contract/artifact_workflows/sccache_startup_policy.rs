//! Every sccache startup applies the central runner cache policy before compilation.
use super::{Node, document, input, job, source, steps, support::Fixture, text, workflow_yaml};
use serde_json::Value;
use std::fs;

const CONFIGURE: &str = "./.github/actions/configure-sccache-gha";
const MACOS_LLD: &str = "./.github/actions/setup-macos-lld";
// Hosted-literal jobs outside the selector; release_compilation, the Windows
// runtime-input guard and prepare-automation own their compiler-cache policy.
const EXEMPT: &[(&str, &str)] = &[
    ("release.yml", "build_linux_arm64"),
    ("release.yml", "windows_host_input"),
    ("release.yml", "build_native_runtime_windows_cpu"),
    ("release.yml", "build_native_runtime_windows_gpu"),
    ("windows-warm-caches.yml", "warm_windows_cpu"),
    ("windows-warm-caches.yml", "warm_windows_gpu"),
    ("claude-live-model-gate.yml", "live_claude"),
];
const SELECTOR_BOUND: &[(&str, &str, &str)] = &[
    ("ci-macos-host-slice.yml", "macos_host", "runner_policy"),
    (
        "ci-macos-runtime-slice.yml",
        "macos_runtime",
        "runner_policy",
    ),
    ("release.yml", "smoke_linux_arm64_artifact", "metadata"),
];

fn starts_sccache(step: &Node) -> bool {
    let uses = text(step, "uses").unwrap_or("");
    uses.starts_with("mozilla-actions/sccache-action@")
        || uses == MACOS_LLD
        || text(step, "run").is_some_and(|run| run.contains("sccache --start-server"))
}

fn unrouted(name: &str, workflow: &Node) -> Vec<String> {
    let mut missing = Vec::new();
    for (job_name, job) in workflow.get("jobs").unwrap().entries() {
        let Some(Node::Seq(steps)) = job.get("steps") else {
            continue;
        };
        if EXEMPT.contains(&(name, job_name.as_str())) {
            continue;
        }
        let Some(first) = steps.iter().position(starts_sccache) else {
            continue;
        };
        let routed = steps.iter().any(|step| {
            text(step, "uses") == Some(CONFIGURE)
                && input(step, "allow_native_github_cache").is_some_and(|v| !v.is_empty())
                && input(step, "allow_depot_remote_cache") != Some("")
        });
        let applied_before_work = steps[first..]
            .iter()
            .take_while(|s| {
                text(s, "uses").is_some_and(|u| {
                    u == CONFIGURE
                        || u == MACOS_LLD
                        || u.starts_with("mozilla-actions/sccache-action@")
                })
            })
            .any(|s| text(s, "uses") == Some(CONFIGURE))
            || steps[..first]
                .iter()
                .any(|s| text(s, "uses") == Some(CONFIGURE));
        if !routed || !applied_before_work {
            missing.push(format!("{name}/{job_name}"));
        }
    }
    missing
}

fn all_workflows() -> Vec<(String, Node)> {
    let mut names = fs::read_dir(super::support::root().join(".github/workflows"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".yml"))
        .collect::<Vec<_>>();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let doc = document(&n);
            (n, doc)
        })
        .collect()
}

#[test]
fn every_sccache_startup_path_is_routed_through_the_selector_policy() {
    let mut missing = Vec::new();
    for (name, workflow) in all_workflows() {
        missing.extend(unrouted(&name, &workflow));
    }
    assert_eq!(missing, Vec::<String>::new());
    for (file, job_name) in EXEMPT {
        assert!(
            steps(job(&document(file), job_name))
                .iter()
                .any(starts_sccache),
            "stale exemption {file}/{job_name}"
        );
    }
}

#[test]
fn selector_placed_startups_bind_exact_selector_outputs() {
    for (file, job_name, owner) in SELECTOR_BOUND {
        let doc = document(file);
        let configure = steps(job(&doc, job_name))
            .iter()
            .filter(|s| text(s, "uses") == Some(CONFIGURE))
            .collect::<Vec<_>>();
        assert_eq!(configure.len(), 1, "{file}/{job_name}");
        assert!(text(configure[0], "if").is_none());
        for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
            assert_eq!(
                input(configure[0], flag),
                Some(format!("${{{{ needs.{owner}.outputs.{flag} }}}}").as_str()),
                "{file}/{job_name}/{flag}"
            );
            assert!(
                job(&doc, owner)
                    .get("outputs")
                    .and_then(|o| o.get(flag))
                    .is_some(),
                "{file}: {owner} must export {flag}"
            );
        }
    }
}

#[test]
fn reverting_a_macos_slice_to_the_lld_only_bypass_is_rejected() {
    for file in ["ci-macos-host-slice.yml", "ci-macos-runtime-slice.yml"] {
        let original = source(file);
        let start = original
            .find("      - name: Apply runner cache policy to the macOS compiler cache\n")
            .expect("macOS policy step");
        let end = original[start..]
            .find("allow_native_github_cache: ${{ needs.runner_policy.outputs.allow_native_github_cache }}\n")
            .map(|i| start + i + "allow_native_github_cache: ${{ needs.runner_policy.outputs.allow_native_github_cache }}\n".len())
            .unwrap();
        let bypass = format!("{}{}", &original[..start], &original[end..]);
        let doc = workflow_yaml::parse(&bypass).unwrap();
        assert_eq!(unrouted(file, &doc).len(), 1, "{file}");
    }
    let release = source("release.yml").replacen(
        "          allow_native_github_cache: ${{ needs.metadata.outputs.allow_native_github_cache }}\n",
        "          allow_native_github_cache: \"\"\n",
        1,
    );
    assert_eq!(
        unrouted("release.yml", &workflow_yaml::parse(&release).unwrap()),
        ["release.yml/smoke_linux_arm64_artifact"]
    );
}

#[test]
fn release_arm64_smoke_applies_policy_before_compiling_its_verifier() {
    let doc = document("release.yml");
    let smoke = steps(job(&doc, "smoke_linux_arm64_artifact"));
    let install = smoke
        .iter()
        .position(|s| {
            text(s, "uses").is_some_and(|u| u.starts_with("mozilla-actions/sccache-action@"))
        })
        .unwrap();
    let configure = smoke
        .iter()
        .position(|s| text(s, "uses") == Some(CONFIGURE))
        .unwrap();
    let cargo = smoke
        .iter()
        .position(|s| text(s, "run").is_some_and(|r| r.contains("cargo xtool")))
        .unwrap();
    assert_eq!(configure, install + 1);
    assert!(configure < cargo);
    let outputs = job(&doc, "metadata").get("outputs").unwrap();
    for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
        assert_eq!(
            text(outputs, flag),
            Some(format!("${{{{ steps.runners.outputs.{flag} }}}}").as_str())
        );
    }
}

fn macos_job_env(file: &str) -> String {
    text(document(file).get("env").unwrap(), "SCCACHE_GHA_ENABLED")
        .unwrap()
        .to_owned()
}

fn assert_disk_only(value: &Value) {
    assert_eq!(value["exports"]["SCCACHE_MULTILEVEL_CHAIN"], "disk");
    assert_eq!(value["failures"], serde_json::json!([]));
    let starts = value["calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["args"][0] == "--start-server")
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 1);
    for key in [
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_CACHE_URL",
        "SCCACHE_WEBDAV_ENDPOINT",
        "SCCACHE_WEBDAV_TOKEN",
    ] {
        assert_eq!(starts[0]["env"][key], "", "{key}");
    }
}

#[test]
fn macos_denied_callers_with_populated_endpoints_stay_disk_only_and_hosted_keeps_disk() {
    for file in ["ci-macos-host-slice.yml", "ci-macos-runtime-slice.yml"] {
        let gha = macos_job_env(file);
        assert_eq!(gha, "false", "{file}: macOS compiler cache stays job-local");
        // Denied (false) and hosted (true) authority, with and without ambient endpoints.
        for native in ["false", "true"] {
            for event in ["push", "pull_request"] {
                for (endpoint, token) in
                    [("http://webdav.fixture", "fixture-depot-token"), ("", "")]
                {
                    let value = Fixture::new().cache(&[
                        ("INPUT_ALLOW_DEPOT_REMOTE_CACHE", "false"),
                        ("INPUT_ALLOW_NATIVE_GITHUB_CACHE", native),
                        ("GITHUB_EVENT_NAME", event),
                        ("SCCACHE_GHA_ENABLED", gha.as_str()),
                        ("SCCACHE_WEBDAV_ENDPOINT", endpoint),
                        ("DEPOT_CACHE_TOKEN", token),
                    ]);
                    assert_disk_only(&value);
                }
            }
        }
    }
}

fn selected(date: &str, gate: &str) -> Option<Value> {
    let fixture = Fixture::new();
    let output = fixture.selector(&[("INPUT_DEPOT_PR_ENABLED", gate), ("FIXTURE_DATE", date)]);
    if !output.status.success() {
        assert!(!fixture.path().join("outputs").exists());
        return None;
    }
    Some(fixture.outputs())
}

#[test]
fn pr_exception_gate_expires_at_the_checked_in_date_and_rejects_malformed_values() {
    for (date, depot) in [
        ("2027-09-12", true),
        ("2027-09-13", true),
        ("2027-09-14", false),
        ("2027-09-15", false),
        ("2028-01-01", false),
    ] {
        let values = selected(date, "true").unwrap();
        assert_eq!(values["depot_enabled"], depot.to_string(), "{date}");
        assert_eq!(values["allow_native_github_cache"], "true", "{date}");
        assert_eq!(values["allow_depot_remote_cache"], "false", "{date}");
        assert_eq!(
            values["allow_trusted_sccache_seed"],
            (!depot).to_string(),
            "{date}"
        );
        assert_eq!(
            values["runner_macos"],
            if depot { "depot-macos-15" } else { "macos-15" }
        );
    }
    for malformed in ["TRUE", "True", "1", "yes", "", " true"] {
        assert!(selected("2027-09-13", malformed).is_none(), "{malformed:?}");
    }
}

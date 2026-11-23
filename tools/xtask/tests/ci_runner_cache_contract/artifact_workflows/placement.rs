//! Real planner output, checked lane expressions and executable selector policy.
use super::{document, input, job, steps, support, text};
use serde_json::{Value, json};
use std::process::Command;

fn plan(profile: &str, changed: &str) -> Value {
    let request = json!({
        "profile": profile,
        "event_name": if profile == "main" { "push" } else { "workflow_dispatch" },
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "changed_files": [changed],
        "workspace_packages": [
            {"name": "mesh-llm", "path": "mesh/crates/mesh-llm"},
            {"name": "xtask", "path": "tools/xtask"}
        ]
    });
    ci_controller::plan::build_for_validation(
        &support::root(),
        &serde_json::to_vec(&request).unwrap(),
    )
    .unwrap()
}

fn forwarded_hosted(expression: &str, plan: &Value) -> String {
    let expression = expression
        .strip_prefix("${{ ")
        .unwrap()
        .strip_suffix(" }}")
        .unwrap();
    let fixture = support::Fixture::new();
    let mut command = Command::new("node");
    command.env("LANE_PLAN", plan.to_string()).args([
        "-e",
        &format!(
            "const inputs={{lane_plan_json:process.env.LANE_PLAN}}; \
             const fromJson=JSON.parse,toJson=v=>JSON.stringify(v??null); \
             console.log({expression});"
        ),
    ]);
    let output = fixture.run(command);
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[test]
fn exhaustive_placement_preserves_coverage_and_resolves_every_lane_caller() {
    let baseline = plan("main", "README.md");
    for (profile, changed, hosted) in [
        ("main", "mesh/crates/mesh-llm/src/lib.rs", false),
        ("main", "README.md", false),
        ("manual-full", "README.md", false),
        ("main", ".github/workflows/main_linux.yml", true),
        ("manual-full", ".github/workflows/main_linux.yml", true),
        ("main", "__force_all__", true),
        ("manual-full", "__force_all__", true),
    ] {
        let plan = plan(profile, changed);
        assert_eq!(plan["signals"]["runner_contract_required"], true);
        assert_eq!(plan["signals"]["hosted_runner_required"], hosted);
        for field in [
            "required_slices",
            "matrices",
            "dependencies",
            "runner_roles",
            "budgets",
        ] {
            assert_eq!(plan[field], baseline[field], "{profile} {changed}: {field}");
        }
        let mut callers = 0;
        for lane in ["quality", "website", "linux", "macos", "windows"] {
            let workflow = document(&format!("ci-{lane}-lane.yml"));
            for (name, caller) in workflow.get("jobs").unwrap().entries() {
                let Some(expression) = input(caller, "force_hosted") else {
                    continue;
                };
                callers += 1;
                let forwarded = forwarded_hosted(expression, &plan);
                assert_eq!(forwarded, hosted.to_string(), "{lane}/{name}");
                resolve_caller(caller, &plan, &forwarded);
            }
        }
        assert_eq!(callers, 26);
    }
}

fn resolve_caller(caller: &super::Node, plan: &Value, hosted: &str) {
    let file = text(caller, "uses")
        .unwrap()
        .strip_prefix("./.github/workflows/")
        .unwrap();
    let target = document(file);
    let policy = job(&target, "runner_policy");
    for selector in steps(policy).iter().filter(|s| {
        text(s, "uses") == Some("./.github/actions/select-ci-runners")
            && text(s, "id") != Some("sentinel_policy")
    }) {
        let force = match input(selector, "force_hosted").unwrap() {
            "${{ inputs.force_hosted }}" => hosted,
            "true" => "true",
            other => panic!("unexpected hosted forwarding: {file}: {other}"),
        };
        let eligible = input(selector, "depot_main_enabled")
            == Some("${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}");
        let fixture = support::Fixture::new();
        let event = plan["event_name"].as_str().unwrap();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", event),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_FORCE_HOSTED", force),
            (
                "INPUT_DEPOT_MAIN_ENABLED",
                if eligible { "true" } else { "false" },
            ),
            ("FIXTURE_DATE", "2026-11-21"),
        ]);
        assert!(output.status.success(), "{file}: {output:?}");
        let resolved = fixture.outputs();
        let depot = eligible && force == "false";
        assert_eq!(resolved["depot_enabled"], depot.to_string(), "{file}");
        assert_eq!(resolved["allow_depot_remote_cache"], "false");
        assert_eq!(resolved["allow_native_github_cache"], (!depot).to_string());
        assert_eq!(resolved["allow_trusted_sccache_seed"], (!depot).to_string());
    }
}

#[test]
fn placement_rejects_missing_or_malformed_signal_and_preserves_explicit_override() {
    let lane = document("ci-linux-lane.yml");
    let expression = input(job(&lane, "hosts"), "force_hosted").unwrap();
    for value in [Value::Null, json!("false"), json!(0), json!(true)] {
        let mut incomplete = plan("main", "README.md");
        incomplete["signals"]["hosted_runner_required"] = value;
        assert_eq!(forwarded_hosted(expression, &incomplete), "true");
    }
    for (gate, manual, force, depot) in [
        ("false", "false", "false", false),
        ("false", "true", "false", true),
        ("true", "true", "true", false),
    ] {
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_ORIGINAL_EVENT_NAME", "workflow_dispatch"),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", gate),
            ("INPUT_MANUAL_USE_DEPOT", manual),
            ("INPUT_FORCE_HOSTED", force),
            ("FIXTURE_DATE", "2026-11-21"),
        ]);
        assert!(output.status.success(), "{output:?}");
        assert_eq!(fixture.outputs()["depot_enabled"], depot.to_string());
    }
}

//! Composed routing: real planner output through every lane boundary into the
//! real selector, checked against each role's `ci/runner-roles.json` decision.
use super::hosted_precedence::{Decision, vars, walk};
use super::{Node, support, text};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs};

const LANES: [&str; 5] = [
    "ci-quality-lane.yml",
    "ci-website-lane.yml",
    "ci-linux-lane.yml",
    "ci-macos-lane.yml",
    "ci-windows-lane.yml",
];
const CI_POLICY: &str = ".github/workflows/main_linux.yml";
const SOURCE: &str = "mesh/crates/mesh-llm/src/lib.rs";

fn plan(profile: &str, changed: &str) -> Value {
    let event = match profile {
        "main" => "push",
        "pr-ready" | "pr-draft" => "pull_request",
        _ => "workflow_dispatch",
    };
    let request = json!({
        "profile": profile,
        "event_name": event,
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

/// Selector id (`file#job/step`) -> role row.
fn roles() -> BTreeMap<String, Value> {
    let path = support::root().join("ci/runner-roles.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for row in doc["roles"].as_array().unwrap() {
        for selector in row["selectors"].as_array().unwrap() {
            map.insert(selector.as_str().unwrap().to_owned(), row.clone());
        }
    }
    map
}

struct Case {
    profile: &'static str,
    changed: &'static str,
    event: &'static str,
    use_depot: bool,
    gate: bool,
    trust: Value,
}

fn route(case: &Case) -> Vec<Decision> {
    let mut trust = json!({
        "INPUT_EVENT_NAME": case.event,
        "INPUT_REF": "refs/heads/main",
    });
    for (key, value) in case.trust.as_object().unwrap() {
        trust[key] = value.clone();
    }
    let context = json!({
        "inputs": {
            "lane_plan_json": plan(case.profile, case.changed).to_string(),
            "use_depot": case.use_depot,
            "original_event_name": case.event,
        },
        "vars": vars(case.gate),
        "github": {"event_name": case.event},
        "needs": {},
        "trust": trust,
    });
    let mut decisions = Vec::new();
    for lane in LANES {
        walk(lane, &context, &mut decisions);
    }
    decisions
}

/// `column` names the runner-roles field whose `depot-gated` rows may take
/// Depot; `None` means every selector must stay hosted.
fn assert_routes(case: &Case, column: Option<&str>) {
    let roles = roles();
    let decisions = route(case);
    assert!(decisions.len() >= 10, "walk reached {}", decisions.len());
    for decision in &decisions {
        let row = roles
            .get(&decision.selector)
            .unwrap_or_else(|| panic!("{}: no runner-roles row", decision.selector));
        let depot = column.is_some_and(|field| row[field] == "depot-gated");
        assert_eq!(
            decision.depot, depot,
            "{} ({}) via {}: {} {} gate={}",
            decision.selector, row["id"], decision.at, case.profile, case.changed, case.gate
        );
    }
    if column.is_some() {
        assert!(decisions.iter().any(|d| d.depot), "no Depot selector");
    }
}

fn case(profile: &'static str, changed: &'static str, event: &'static str) -> Case {
    Case {
        profile,
        changed,
        event,
        use_depot: false,
        gate: true,
        trust: json!({}),
    }
}

#[test]
fn ci_policy_change_and_force_all_stay_hosted_at_every_nested_selector() {
    for changed in [CI_POLICY, "__force_all__"] {
        assert_routes(&case("main", changed, "push"), None);
    }
}

#[test]
fn main_source_change_takes_depot_only_where_the_role_is_depot_gated() {
    assert_routes(&case("main", SOURCE, "push"), Some("main"));
    let closed = Case {
        gate: false,
        ..case("main", SOURCE, "push")
    };
    assert_routes(&closed, None);
}

#[test]
fn manual_use_depot_cannot_override_a_ci_policy_hosted_signal() {
    let manual = Case {
        use_depot: true,
        ..case("manual-full", CI_POLICY, "workflow_dispatch")
    };
    assert_routes(&manual, None);
}

#[test]
fn untrusted_or_non_main_contexts_stay_hosted() {
    let pr = json!({"INPUT_REF": "refs/pull/12/merge", "FIXTURE_DATE": "2027-09-13"});
    let fork = json!({
        "INPUT_REF": "refs/pull/12/merge",
        "INPUT_HEAD_REPOSITORY": "fork-owner/mesh-llm",
        "FIXTURE_DATE": "2027-09-13",
    });
    for (profile, event, trust) in [
        ("pr-ready", "pull_request", fork),
        ("pr-ready", "pull_request_target", pr),
        ("main", "push", json!({"INPUT_REF": "refs/heads/feature"})),
    ] {
        let untrusted = Case {
            trust,
            ..case(profile, SOURCE, event)
        };
        assert_routes(&untrusted, None);
    }
}

#[test]
fn pull_request_eligibility_ends_at_the_checked_in_expiry() {
    for (date, column) in [("2027-09-13", Some("pull_request")), ("2027-09-15", None)] {
        let pr = Case {
            trust: json!({"INPUT_REF": "refs/pull/12/merge", "FIXTURE_DATE": date}),
            ..case("pr-ready", SOURCE, "pull_request")
        };
        assert_routes(&pr, column);
    }
}

fn record_step() -> Node {
    let action = support::action("select-ci-runners");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps")
    };
    let last = steps.last().unwrap().clone();
    assert_eq!(text(&last, "id"), Some("record"));
    last
}

#[test]
fn selector_records_its_decision_without_secrets() {
    let inputs = [
        ("INPUT_EVENT_NAME", "push"),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_DEPOT_MAIN_ENABLED", "true"),
        ("INPUT_DEPOT_PR_ENABLED", "false"),
        ("INPUT_FORCE_HOSTED", "false"),
        ("INPUT_MANUAL_USE_DEPOT", "false"),
    ];
    let fixture = support::Fixture::new();
    assert!(fixture.selector(&inputs).status.success());
    let selected = fixture.outputs();
    let step = record_step();
    let mut command = std::process::Command::new("bash");
    command
        .env("GITHUB_OUTPUT", fixture.path().join("outputs"))
        .env("GITHUB_STEP_SUMMARY", fixture.path().join("summary"))
        .env("RUNNER_ENVIRONMENT", "github-hosted")
        .args(["-c", text(&step, "run").unwrap()]);
    for (key, raw) in step.get("env").unwrap().entries() {
        let expression = raw.text().unwrap().trim_start_matches("${{ ");
        let expression = expression.trim_end_matches(" }}");
        let value = match expression.split_once('.').unwrap() {
            ("inputs", name) => {
                let wanted = format!("INPUT_{}", name.to_uppercase());
                inputs
                    .iter()
                    .find(|(k, _)| *k == wanted)
                    .unwrap()
                    .1
                    .to_owned()
            }
            ("steps", path) => {
                let name = path.strip_prefix("select.outputs.").unwrap();
                selected[name].as_str().unwrap().to_owned()
            }
            other => panic!("{key}: unexpected binding {other:?}"),
        };
        command.env(key, value);
    }
    assert!(fixture.run(command).status.success());
    let outputs = fixture.outputs();
    let record = outputs["decision_record"].as_str().unwrap();
    let fields = record
        .split(' ')
        .map(|pair| pair.split_once('=').unwrap())
        .collect::<BTreeMap<_, _>>();
    for (key, value) in [
        ("event", "push"),
        ("ref", "refs/heads/main"),
        ("depot_main_enabled", "true"),
        ("depot_pr_enabled", "false"),
        ("force_hosted", "false"),
        ("manual_use_depot", "false"),
        ("depot_enabled", "true"),
        ("allow_native_github_cache", "false"),
        ("allow_depot_remote_cache", "false"),
        ("runner", "depot-ubuntu-24.04"),
        ("selector_runner_environment", "github-hosted"),
    ] {
        assert_eq!(fields.get(key), Some(&value), "{record}");
    }
    assert_eq!(fields.len(), 11, "{record}");
    let summary = fs::read_to_string(fixture.path().join("summary")).unwrap();
    assert_eq!(summary.lines().count(), 1);
    assert!(summary.contains(record));
}

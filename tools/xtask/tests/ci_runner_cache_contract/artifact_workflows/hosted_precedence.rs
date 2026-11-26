//! Hosted override precedence through every runner-selecting reusable-workflow
//! boundary: per-boundary forwarding and the real selector, not planner output.
use super::{Node, document, input, job, source, steps, support, text};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, process::Command};

const SELECTOR: &str = "./.github/actions/select-ci-runners";
const HOSTED: [&str; 4] = [
    "true",
    "${{ inputs.force_hosted }}",
    "${{ toJson(fromJson(inputs.lane_plan_json).signals.hosted_runner_required) != 'false' }}",
    "${{ needs.metadata.outputs.force_hosted_runners == 'true' }}",
];
const PROVIDER: [&str; 3] = [
    "false",
    "${{ inputs.use_depot }}",
    "${{ inputs.use_depot == true }}",
];
const POLICY_INPUTS: [&str; 3] = ["force_hosted", "use_depot", "original_event_name"];
const SELECTOR_INPUTS: [&str; 4] = [
    "original_event_name",
    "depot_main_enabled",
    "force_hosted",
    "manual_use_depot",
];
// GitHub expression subset used by the forwarding bindings; `==` keeps
// GitHub's no-coercion result for the string/boolean pairs used here.
const EVALUATOR: &str = "const {inputs,vars,needs,github}=JSON.parse(process.env.CONTEXT);\
const fromJson=JSON.parse,toJson=v=>JSON.stringify(v??null);\
const value=raw=>{const m=/^\\$\\{\\{ (.*) \\}\\}$/.exec(raw);\
if(!m)return raw==='true'?true:raw==='false'?false:raw;return eval(m[1]);};\
console.log(JSON.stringify(JSON.parse(process.env.VALUES).map(value)));";

fn declared(doc: &Node) -> Option<&Node> {
    doc.get("on")?.get("workflow_call")?.get("inputs")
}
fn local(caller: &Node) -> Option<&str> {
    text(caller, "uses")?.strip_prefix("./.github/workflows/")
}
fn runner_selecting(target: &str) -> Option<Node> {
    let declared = declared(&document(target))?.clone();
    declared.get("force_hosted").is_some().then_some(declared)
}
fn callers() -> Vec<String> {
    let mut files = fs::read_dir(support::root().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| {
            name.ends_with(".yml") && source(name).contains("uses: ./.github/workflows/")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn every_runner_selecting_call_forwards_override_provider_and_event() {
    let mut seen = BTreeSet::new();
    for file in callers() {
        let doc = document(&file);
        for (name, caller) in doc.get("jobs").unwrap().entries() {
            let Some(target) = local(caller) else {
                continue;
            };
            let Some(inputs) = runner_selecting(target) else {
                continue;
            };
            let at = format!("{file}/{name} -> {target}");
            let hosted = input(caller, "force_hosted").unwrap_or("<omitted>");
            assert!(HOSTED.contains(&hosted), "{at}: force_hosted {hosted}");
            if inputs.get("use_depot").is_some() {
                let provider = input(caller, "use_depot").unwrap_or("<omitted>");
                assert!(PROVIDER.contains(&provider), "{at}: use_depot {provider}");
            }
            assert!(
                input(caller, "original_event_name").is_some(),
                "{at}: original_event_name omitted"
            );
            seen.insert(format!("{file}/{name}"));
        }
    }
    for boundary in [
        "ci-linux-lane.yml/static_abi",
        "ci-linux-lane.yml/kotlin_sdk_input",
        "ci-linux-lane.yml/hosts",
        "ci-pr-canary-lane.yml/hosts",
        "native-sdk-artifact.yml/produce_linux_static_abi",
        "release.yml/release_ui",
        "release.yml/build_native_sdk_runtime",
        "release.yml/build_swift_sdk_artifact",
    ] {
        assert!(seen.contains(boundary), "{boundary} left the census");
    }
}

fn evaluate(context: &Value, raw: &[&str]) -> Vec<Value> {
    let fixture = support::Fixture::new();
    let mut command = Command::new("node");
    command
        .env("CONTEXT", context.to_string())
        .env("VALUES", serde_json::to_string(raw).unwrap())
        .args(["-e", EVALUATOR]);
    let output = fixture.run(command);
    assert!(output.status.success(), "{output:?}");
    serde_json::from_slice(&output.stdout).unwrap()
}
fn workflow_string(value: Value) -> String {
    match value {
        Value::String(text) => text,
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

struct Decision {
    at: String,
    depot: bool,
    // Binds the forwarded manual opt-in, so it must match the parent decision.
    mirrors_parent: bool,
}

fn decide(at: String, selector: &Node, context: &Value) -> Decision {
    let with = selector.get("with").unwrap();
    let bound = SELECTOR_INPUTS
        .into_iter()
        .filter_map(|key| text(with, key).map(|raw| (key, raw)))
        .collect::<Vec<_>>();
    let values = evaluate(
        context,
        &bound.iter().map(|(_, raw)| *raw).collect::<Vec<_>>(),
    );
    let mut env = bound
        .iter()
        .zip(values)
        .map(|((key, _), value)| {
            (
                format!("INPUT_{}", key.to_uppercase()),
                workflow_string(value),
            )
        })
        .collect::<Vec<_>>();
    for (key, value) in [
        ("INPUT_EVENT_NAME", "workflow_dispatch"),
        ("INPUT_REF", "refs/heads/main"),
        ("INPUT_DEPOT_PR_ENABLED", "true"),
        ("FIXTURE_DATE", "2026-11-21"),
    ] {
        env.push((key.to_owned(), value.to_owned()));
    }
    let fixture = support::Fixture::new();
    let env = env
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect::<Vec<_>>();
    let output = fixture.selector(&env);
    assert!(output.status.success(), "{at}: {output:?}");
    Decision {
        depot: fixture.outputs()["depot_enabled"] == "true",
        mirrors_parent: text(with, "manual_use_depot").is_some_and(|raw| raw != "false"),
        at,
    }
}

fn forwarded(caller: &Node, declared: &Node, context: &Value) -> Value {
    let mut nested = serde_json::Map::new();
    for key in POLICY_INPUTS {
        let Some(spec) = declared.get(key) else {
            continue;
        };
        let value = match input(caller, key) {
            Some(raw) => evaluate(context, &[raw]).remove(0),
            None if text(spec, "type") == Some("boolean") => {
                json!(text(spec, "default") == Some("true"))
            }
            None => json!(text(spec, "default").unwrap_or("")),
        };
        nested.insert(key.to_owned(), value);
    }
    Value::Object(nested)
}

fn walk(file: &str, context: &Value, decisions: &mut Vec<Decision>) {
    let doc = document(file);
    for (name, j) in doc.get("jobs").unwrap().entries() {
        if j.get("steps").is_some() {
            for selector in steps(j)
                .iter()
                .filter(|s| text(s, "uses") == Some(SELECTOR))
            {
                decisions.push(decide(format!("{file}/{name}"), selector, context));
            }
        }
        let Some(target) = local(j) else {
            continue;
        };
        let Some(declared) = runner_selecting(target) else {
            continue;
        };
        let nested = json!({
            "inputs": forwarded(j, &declared, context),
            "vars": context["vars"],
            "github": context["github"],
            "needs": {},
        });
        walk(target, &nested, decisions);
    }
}

fn assert_precedence(decisions: &[Decision], hosted: bool, parent_depot: bool, required: &[&str]) {
    for decision in decisions {
        if hosted {
            assert!(!decision.depot, "{}: hosted override lost", decision.at);
        } else if decision.mirrors_parent {
            assert_eq!(
                decision.depot, parent_depot,
                "{}: parent disagreement",
                decision.at
            );
        }
    }
    for at in required {
        assert!(
            decisions.iter().any(|d| d.at == *at && d.mirrors_parent),
            "{at} not reached"
        );
    }
}

fn vars(gate: bool) -> Value {
    json!({"DEPOT_RUNNERS_ENABLED": gate.to_string(), "DEPOT_PR_RUNNERS_ENABLED": "true"})
}

// (hosted override, manual use_depot, DEPOT_RUNNERS_ENABLED)
const CASES: [(bool, bool, bool); 4] = [
    (true, true, true),
    (false, true, false),
    (false, false, true),
    (false, false, false),
];

#[test]
fn release_hosted_rollback_wins_over_use_depot_and_gate_at_every_nested_selector() {
    let release = document("release.yml");
    let meta = steps(job(&release, "metadata"))
        .iter()
        .find(|s| text(s, "id") == Some("meta"))
        .unwrap();
    let projected = text(meta.get("env").unwrap(), "INPUT_FORCE_HOSTED_RUNNERS").unwrap();
    for (force, manual, gate) in CASES {
        let mut context = json!({
            "inputs": {"force_hosted_runners": force, "use_depot": manual},
            "vars": vars(gate),
            "github": {"event_name": "workflow_dispatch"},
        });
        let output = workflow_string(evaluate(&context, &[projected]).remove(0));
        context["needs"] = json!({"metadata": {"outputs": {"force_hosted_runners": output}}});
        let mut decisions = Vec::new();
        walk("release.yml", &context, &mut decisions);
        assert_precedence(
            &decisions,
            force,
            gate || manual,
            &[
                "release.yml/metadata",
                "native-sdk-artifact.yml/runner_policy",
                "static-abi-artifact.yml/runner_policy",
            ],
        );
    }
    let preview = document("rolling-preview.yml");
    let dispatch = steps(job(&preview, "dispatch"))
        .iter()
        .filter_map(|s| text(s, "run"))
        .find(|run| run.contains("gh workflow run release.yml"))
        .unwrap()
        .split_whitespace()
        .collect::<Vec<_>>();
    for argument in ["force_hosted_runners=true", "use_depot=false"] {
        assert!(dispatch.contains(&argument), "rolling preview {argument}");
    }
}

#[test]
fn linux_lane_hosted_signal_and_use_depot_reach_every_nested_selector() {
    for (hosted, manual, gate) in CASES {
        let plan = json!({"signals": {"hosted_runner_required": hosted}});
        let context = json!({
            "inputs": {
                "lane_plan_json": plan.to_string(),
                "use_depot": manual,
                "original_event_name": "workflow_dispatch",
            },
            "vars": vars(gate),
            "github": {"event_name": "workflow_dispatch"},
            "needs": {},
        });
        let mut decisions = Vec::new();
        walk("ci-linux-lane.yml", &context, &mut decisions);
        assert_precedence(
            &decisions,
            hosted,
            gate || manual,
            &[
                "ci-linux-host-slice.yml/runner_policy",
                "static-abi-artifact.yml/runner_policy",
                "native-sdk-artifact.yml/runner_policy",
            ],
        );
    }
}

#[test]
fn no_release_leg_preparing_hosted_bare_can_take_a_selected_provider_label() {
    let doc = document("release.yml");
    let mut legs = BTreeSet::new();
    for (name, j) in doc.get("jobs").unwrap().entries() {
        if j.get("steps").is_none()
            || !steps(j).iter().any(|s| {
                text(s, "uses") == Some("./.github/actions/prepare-automation")
                    && input(s, "runner-profile").is_some_and(|p| p.contains("hosted-bare"))
            })
        {
            continue;
        }
        let labels = match text(j, "runs-on").unwrap() {
            "${{ matrix.os }}" => {
                let Node::Seq(rows) = j
                    .get("strategy")
                    .unwrap()
                    .get("matrix")
                    .unwrap()
                    .get("include")
                    .unwrap()
                else {
                    panic!("{name}: matrix rows")
                };
                rows.iter().map(|row| text(row, "os").unwrap()).collect()
            }
            label => vec![label],
        };
        for label in labels {
            assert!(
                [
                    "ubuntu-24.04",
                    "ubuntu-24.04-arm",
                    "macos-15",
                    "windows-2022"
                ]
                .contains(&label),
                "{name}: {label} is not a GitHub-hosted label"
            );
        }
        legs.insert(name.as_str());
    }
    assert!(legs.contains("build_native_runtime"), "{legs:?}");
}

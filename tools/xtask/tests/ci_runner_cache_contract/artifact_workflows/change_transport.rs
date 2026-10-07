//! File-backed planning input and the caller action-input size boundary.
use super::{Node, fixture, support};
use std::{fs, process::Command};

fn plan_source() -> (Node, String) {
    let action = support::action("plan-ci");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!("steps")
    };
    let plan = steps
        .iter()
        .find(|step| step.get("id").and_then(Node::text) == Some("plan"))
        .unwrap()
        .clone();
    let source = plan.get("run").unwrap().text().unwrap().to_owned();
    (plan, source)
}

fn command(f: &support::Fixture, source: &str) -> Command {
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(f.path()).env(
        "PATH",
        format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
    );
    for (key, value) in [
        ("EVENT_NAME", "push"),
        ("SOURCE_SHA", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("BASE_SHA", ""),
        ("DRAFT", "false"),
        ("AFFECTED_CRATES", "[]"),
        ("UPLOAD_ARTIFACT", "false"),
        ("ARTIFACT_NAME", ""),
    ] {
        command.env(key, value);
    }
    command
        .env("GITHUB_WORKSPACE", support::root())
        .env("RUNNER_TEMP", f.path())
        .env("HOME", f.path())
        .env("TMPDIR", f.path())
        .env("GITHUB_OUTPUT", f.path().join("outputs"))
        .env("GITHUB_STEP_SUMMARY", f.path().join("summary"))
        .env("MESH_LLM_AUTOMATION_BIN", f.path().join("bin/automation"))
        .args(["-c", source]);
    command
}

#[test]
fn artifact_plan_large_file_input_survives_without_bulk_argv_or_environment() {
    let (plan, source) = plan_source();
    assert!(plan.get("env").unwrap().get("CHANGED_FILES").is_none());
    assert!(source.contains("--slurpfile changed_files \"$changed_files_json_file\""));
    let f = fixture();
    let paths = (0..40000)
        .map(|index| {
            format!(
                "mesh/crates/mesh-llm-host-runtime/src/fixture_{index:05}_{}.rs",
                "x".repeat(80)
            )
        })
        .collect::<Vec<_>>();
    let list = format!("{}\n", paths.join("\n"));
    assert!(list.len() > 4 * 1024 * 1024);
    fs::write(f.path().join("changed.txt"), &list).unwrap();
    fs::write(
        f.path().join("expected.json"),
        serde_json::to_vec(&paths).unwrap(),
    )
    .unwrap();
    f.executable("automation", r#"[[ "$*" == "ci plan --manifest-root $GITHUB_WORKSPACE" ]] || exit 90
[[ -z "${CHANGED_FILES+x}" ]] || exit 91
/bin/cat > planner-input.json
jq -e --slurpfile expected expected.json '.profile == "main" and .event_name == "push" and .changed_files == $expected[0]' planner-input.json >/dev/null
/bin/cat golden.json"#);
    // Only isolate the action's fixed transport path; no global /tmp file is touched.
    let source = source.replace(
        "/tmp/changed_files.txt",
        &format!("\"{}\"", f.path().join("changed.txt").display()),
    );
    let output = f.run(command(&f, &source));
    assert!(output.status.success(), "{output:?}");
    let input: serde_json::Value =
        serde_json::from_slice(&fs::read(f.path().join("planner-input.json")).unwrap()).unwrap();
    assert_eq!(input["changed_files"], serde_json::json!(paths));
    for mutated in [
        source.replace(
            "--slurpfile changed_files \"$changed_files_json_file\"",
            "--argjson changed_files \"$(cat \"$changed_files_json_file\")\"",
        ),
        format!(
            "export CHANGED_FILES=\"$(cat '{}')\"\n{source}",
            f.path().join("changed.txt").display()
        ),
    ] {
        fs::remove_file(f.path().join("planner-input.json")).unwrap();
        let output = f.run(command(&f, &mutated));
        assert!(
            !output.status.success(),
            "bulk transport unexpectedly admitted"
        );
        if f.path().join("planner-input.json").exists() {
            assert!(
                fs::read(f.path().join("planner-input.json"))
                    .unwrap()
                    .is_empty()
            );
            fs::remove_file(f.path().join("planner-input.json")).unwrap();
        }
        // Restore an inert marker so the next mutation cleanup is uniform.
        fs::write(f.path().join("planner-input.json"), []).unwrap();
    }
}

fn caller_limits(document: &Node) -> Result<Vec<String>, String> {
    let mut guards = Vec::new();
    for (_, job) in document.get("jobs").ok_or("jobs missing")?.entries() {
        let Some(Node::Seq(steps)) = job.get("steps") else {
            continue;
        };
        for (index, plan) in steps.iter().enumerate().filter(|(_, step)| {
            step.get("uses").and_then(Node::text) == Some("./.github/actions/plan-ci")
        }) {
            let (limit_index, limit) = steps
                .iter()
                .enumerate()
                .find(|(_, step)| step.get("id").and_then(Node::text) == Some("change_limit"))
                .ok_or("caller change_limit missing")?;
            let run = limit
                .get("run")
                .and_then(Node::text)
                .ok_or("limit run missing")?;
            let input = plan
                .get("with")
                .and_then(|with| with.get("changed_files"))
                .and_then(Node::text);
            if limit_index >= index
                || !run.contains("wc -c < /tmp/changed_files.txt")
                || !run.contains("> 65536")
                || !run.contains("echo \"force_all=true\" >> \"$GITHUB_OUTPUT\"")
                || input
                    != Some(
                        "${{ steps.change_limit.outputs.force_all == 'true' && '__force_all__' || steps.changes.outputs.changed_files }}",
                    )
            {
                return Err("caller size-limit/fallback changed".into());
            }
            guards.push(run.to_owned());
        }
    }
    Ok(guards)
}

#[test]
fn every_current_plan_caller_bounds_action_inputs_and_limit_mutations_are_refused() {
    let directory = support::root().join(".github/workflows");
    let mut callers = 0;
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "yml") {
            continue;
        }
        let source = fs::read_to_string(path).unwrap();
        let document = super::super::workflow_yaml::parse(&source).unwrap();
        let guards = caller_limits(&document).unwrap();
        for guard in guards {
            callers += 1;
            for size in [65536, 65537] {
                let f = fixture();
                fs::write(f.path().join("changed.txt"), vec![b'x'; size]).unwrap();
                let guard = guard.replace(
                    "/tmp/changed_files.txt",
                    &format!("\"{}\"", f.path().join("changed.txt").display()),
                );
                let mut command = Command::new("/bin/bash");
                command
                    .env_clear()
                    .env("PATH", "/usr/bin:/bin")
                    .env("GITHUB_OUTPUT", f.path().join("limit-output"))
                    .args(["-c", &guard]);
                assert!(f.run(command).status.success());
                let observed =
                    fs::read_to_string(f.path().join("limit-output")).unwrap_or_default();
                assert_eq!(observed, if size > 65536 { "force_all=true\n" } else { "" });
            }
        }
        if source.contains("uses: ./.github/actions/plan-ci") {
            for mutation in [
                source.replace("> 65536", "> 9999999"),
                source.replace(
                    "steps.change_limit.outputs.force_all",
                    "steps.changes.outputs.force_all",
                ),
            ] {
                let changed = super::super::workflow_yaml::parse(&mutation).unwrap();
                assert!(caller_limits(&changed).is_err());
            }
        }
    }
    assert!(callers > 0);
}

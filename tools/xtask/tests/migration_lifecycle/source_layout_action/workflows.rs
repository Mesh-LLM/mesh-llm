//! Actual workflow layout bindings and bounded component-step execution.
use crate::{
    process,
    workflow_yaml::{self, Node},
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

const UI: &str = "${{ steps.layout.outputs.ui_dir }}";
const WEBSITE: &str = "${{ steps.layout.outputs.website_dir }}";

pub(super) fn workflow(name: &str) -> Node {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    workflow_yaml::parse(&fs::read_to_string(root.join(".github/workflows").join(name)).unwrap())
        .unwrap()
}

pub(super) fn steps<'a>(tree: &'a Node, job: &str) -> &'a [Node] {
    let Node::Seq(steps) = tree
        .get("jobs")
        .unwrap()
        .get(job)
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("workflow steps")
    };
    steps
}

pub(super) fn field<'a>(node: &'a Node, name: &str) -> &'a str {
    node.get(name)
        .and_then(Node::text)
        .unwrap_or_else(|| panic!("missing scalar {name}"))
}

pub(super) fn named<'a>(steps: &'a [Node], name: &str) -> &'a Node {
    let selected = steps
        .iter()
        .filter(|s| s.get("name").and_then(Node::text) == Some(name))
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1, "unique step {name}");
    selected[0]
}

pub(super) fn tool(name: &str) -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
        .unwrap_or_else(|| panic!("layout fixture requires {name}"))
        .canonicalize()
        .unwrap()
}

pub(super) fn execute(
    root: &Path,
    executable: PathBuf,
    args: &[&str],
    values: &[(&str, &str)],
) -> (bool, String, String) {
    let mut environment = BTreeMap::from([
        (
            "PATH".into(),
            process::Value::Public(std::env::var_os("PATH").unwrap()),
        ),
        ("HOME".into(), process::Value::Public(root.into())),
    ]);
    for (key, value) in values {
        environment.insert((*key).into(), process::Value::Public((*value).into()));
    }
    let result = process::supervise_raw(
        &process::ProcessSpec {
            executable,
            cwd: root.into(),
            environment,
            arguments: args
                .iter()
                .map(|arg| process::Value::Public((*arg).into()))
                .collect(),
        },
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16384,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(16384),
            stderr: NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    assert_eq!(result.process.outcome, process::Outcome::Exited);
    assert!(
        result.process.failure.is_none() && result.process.cleanup.complete,
        "{result:?}"
    );
    (
        result.process.status.unwrap().success(),
        String::from_utf8(result.stdout.unwrap().as_bytes().to_vec()).unwrap(),
        String::from_utf8(result.stderr.unwrap().as_bytes().to_vec()).unwrap(),
    )
}

pub(super) fn bash(root: &Path, script: &str, values: &[(&str, &str)]) -> (bool, String, String) {
    execute(
        root,
        tool("bash"),
        &["-euo", "pipefail", "-c", script],
        values,
    )
}

fn after_layout(steps: &[Node], consumer: &Node) {
    let layout = steps
        .iter()
        .position(|s| s.get("id").and_then(Node::text) == Some("layout"))
        .unwrap();
    let checkout = steps
        .iter()
        .position(|s| {
            s.get("uses")
                .and_then(Node::text)
                .is_some_and(|s| s.starts_with("actions/checkout@"))
        })
        .unwrap();
    let consumer_index = steps
        .iter()
        .position(|s| std::ptr::eq(s, consumer))
        .unwrap();
    assert!(checkout < layout && layout < consumer_index);
}

#[test]
fn actual_ui_producer_and_all_host_consumers_bind_and_verify_the_resolved_tree() {
    let producer = workflow("ci-ui-artifact-slice.yml");
    let producer_steps = steps(&producer, "ui_artifact");
    let verify = named(producer_steps, "Verify console distribution");
    let upload = named(producer_steps, "Upload immutable console distribution");
    assert_eq!(field(verify, "working-directory"), UI);
    let artifact_path = field(upload.get("with").unwrap(), "path");
    assert_eq!(artifact_path, format!("{UI}/dist"));
    for prefix in ["", "mesh"] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("source with spaces");
        super::populate(&root, prefix);
        let (ok, paths, error) = super::invoke(&root);
        assert!(ok, "{error}");
        let ui = root.join(&paths["ui_dir"]);
        fs::create_dir(ui.join("dist")).unwrap();
        let index = ui.join("dist/index.html");
        fs::write(&index, "<html>console</html>").unwrap();
        assert!(bash(&ui, field(verify, "run"), &[]).0);
        for (name, job) in [
            ("ci-linux-host-slice.yml", "linux_host"),
            ("ci-macos-host-slice.yml", "macos_host"),
            ("sdk-smoke.yml", "sdk_smoke"),
        ] {
            let consumer = workflow(name);
            let consumer_steps = steps(&consumer, job);
            let download = named(consumer_steps, "Download immutable UI distribution");
            assert_eq!(field(download.get("with").unwrap(), "path"), artifact_path);
            after_layout(consumer_steps, download);
            let check = named(consumer_steps, "Verify UI distribution input");
            assert_eq!(field(check.get("env").unwrap(), "UI_DIR"), UI);
            let values = [("UI_DIR", paths["ui_dir"].as_str())];
            let (ok, _, error) = bash(&root, field(check, "run"), &values);
            assert!(ok, "{name}: {error}");
            fs::write(&index, "").unwrap();
            assert!(!bash(&root, field(check, "run"), &values).0);
            fs::remove_file(&index).unwrap();
            assert!(!bash(&root, field(check, "run"), &values).0);
            fs::write(&index, "<html>console</html>").unwrap();
        }
    }
}

#[test]
fn ui_jobs_resolve_after_checkout_and_bind_each_pnpm_step_without_job_directory_defaults() {
    for (name, jobs) in [
        ("ci-web-slice.yml", vec!["ui_quality", "ui_e2e"]),
        ("ci-ui-artifact-slice.yml", vec!["ui_artifact"]),
    ] {
        let tree = workflow(name);
        for job in jobs {
            let node = tree.get("jobs").unwrap().get(job).unwrap();
            assert!(
                node.get("defaults")
                    .and_then(|n| n.get("run"))
                    .and_then(|n| n.get("working-directory"))
                    .is_none()
            );
            let steps = steps(&tree, job);
            let mut checked = 0;
            for step in steps {
                if step
                    .get("run")
                    .and_then(Node::text)
                    .is_some_and(|text| text.starts_with("pnpm "))
                {
                    after_layout(steps, step);
                    assert_eq!(field(step, "working-directory"), UI);
                    checked += 1;
                }
            }
            assert!(checked > 0, "{name}/{job} pnpm coverage");
        }
    }
}

#[test]
fn website_just_evaluation_and_workflow_environments_use_the_resolved_directory() {
    for (name, job, step) in [
        ("ci-web-slice.yml", "website", "Build public website"),
        (
            "ci-quality-slice.yml",
            "cli_docs_sync",
            "Verify generated CLI inventory is deterministic and current",
        ),
    ] {
        let tree = workflow(name);
        let steps = steps(&tree, job);
        let consumer = named(steps, step);
        after_layout(steps, consumer);
        assert_eq!(
            field(consumer.get("env").unwrap(), "MESH_LLM_WEBSITE_DIR"),
            WEBSITE
        );
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    for override_dir in [
        None,
        Some("mesh/website"),
        Some("source tree with spaces/website"),
    ] {
        let values = override_dir
            .map(|path| vec![("MESH_LLM_WEBSITE_DIR", path)])
            .unwrap_or_default();
        let (ok, stdout, stderr) =
            execute(&root, tool("just"), &["--evaluate", "website_dir"], &values);
        assert!(ok, "{stderr}");
        assert_eq!(stdout.trim(), override_dir.unwrap_or("mesh/website"));
    }
}

#[test]
fn current_nightly_pin_fragment_rejects_missing_and_ambiguous_layouts() {
    let tree = workflow("llama-upstream-canary.yml");
    let body = steps(&tree, "resolve")
        .iter()
        .find(|s| s.get("id").and_then(Node::text) == Some("resolve"))
        .unwrap();
    let run = field(body, "run");
    let fragment = run
        .split_once("pins=()")
        .unwrap()
        .1
        .split_once("upstream=\"$UPSTREAM\"")
        .unwrap()
        .0;
    let script = format!("pins=(){fragment}\nprintf '%s' \"$old\"\n");
    let pins = [
        "third_party/llama.cpp/upstream.txt",
        "skippy/llama_cpp/upstream.txt",
    ];
    for selected in [vec![], vec![pins[0]], vec![pins[1]], pins.to_vec()] {
        let root = tempfile::tempdir().unwrap();
        for pin in &selected {
            let file = root.path().join(pin);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, format!("{}\n", "a".repeat(40))).unwrap();
        }
        let (ok, stdout, stderr) = bash(root.path(), &script, &[]);
        assert_eq!(ok, selected.len() == 1, "{stderr}");
        if ok {
            assert_eq!(stdout, "a".repeat(40));
        }
    }
}

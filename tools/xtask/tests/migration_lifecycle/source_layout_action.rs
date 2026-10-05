//! Execute the maintained sole layout action step with isolated directory trees.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

#[path = "source_layout_action/classification.rs"]
mod classification;
#[path = "source_layout_action/node_addon.rs"]
mod node_addon;
#[path = "source_layout_action/workflows.rs"]
mod workflows;

const COMPONENTS: [(&str, &str); 3] = [
    ("ui_dir", "crates/mesh-llm-ui"),
    ("website_dir", "website"),
    ("sdk_dir", "sdk"),
];

fn invoke(root: &Path) -> (bool, BTreeMap<String, String>, String) {
    let output = root.join("outputs");
    fs::write(&output, b"").unwrap();
    let action = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/resolve-source-layout/action.yml"
    ));
    let tree = crate::workflow_yaml::parse(action).unwrap();
    let crate::workflow_yaml::Node::Seq(steps) = tree.get("runs").unwrap().get("steps").unwrap()
    else {
        panic!("action steps")
    };
    assert_eq!(steps.len(), 1, "sole maintained layout resolver step");
    let script = steps[0].get("run").unwrap().text().unwrap();
    let bash = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|directory| directory.join("bash"))
        .find(|path| path.is_file())
        .expect("layout fixture requires Bash");
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: bash,
            cwd: root.to_owned(),
            environment: BTreeMap::from([(
                "GITHUB_OUTPUT".into(),
                Value::Public(output.clone().into()),
            )]),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(16384),
            stderr: NonZeroUsize::new(16384),
        },
    )
    .unwrap();
    assert_eq!(report.process.outcome, Outcome::Exited);
    assert!(report.process.failure.is_none() && report.process.cleanup.complete);
    let mut fields = BTreeMap::new();
    for line in fs::read_to_string(output).unwrap().lines() {
        let (name, value) = line.split_once('=').expect("action output field");
        assert!(fields.insert(name.to_owned(), value.to_owned()).is_none());
    }
    (
        report.process.status.unwrap().success(),
        fields,
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).into_owned(),
    )
}

fn populate(root: &Path, prefix: &str) {
    for (_, relative) in COMPONENTS {
        fs::create_dir_all(root.join(prefix).join(relative)).unwrap();
    }
}

#[test]
fn legacy_and_relocated_action_outputs_preserve_component_paths_without_aliases() {
    for prefix in ["", "mesh"] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("source tree with spaces");
        populate(&root, prefix);
        let (success, fields, stderr) = invoke(&root);
        assert!(success, "{stderr}");
        let expected: BTreeMap<_, _> = COMPONENTS
            .into_iter()
            .map(|(name, relative)| {
                (
                    name.to_owned(),
                    if prefix.is_empty() {
                        relative.to_owned()
                    } else {
                        format!("{prefix}/{relative}")
                    },
                )
            })
            .collect();
        assert_eq!(fields, expected);
        if !prefix.is_empty() {
            assert!(!root.join("crates").exists());
        }
        assert!(!root.join("sdk").is_symlink());
        assert!(!root.join("website").is_symlink());
    }
}

#[test]
fn each_ambiguous_component_is_rejected_before_its_output_is_admitted() {
    for (name, relative) in COMPONENTS {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        populate(root, "");
        fs::create_dir_all(root.join("mesh").join(relative)).unwrap();
        let (success, fields, stderr) = invoke(root);
        assert!(!success);
        assert!(stderr.contains("ambiguous source layout"), "{stderr}");
        assert!(!fields.contains_key(name));
    }
}

#[test]
fn each_missing_component_is_rejected_without_creating_a_replacement_directory() {
    for (name, relative) in COMPONENTS {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        populate(root, "");
        fs::remove_dir_all(root.join(relative)).unwrap();
        let (success, fields, stderr) = invoke(root);
        assert!(!success);
        assert!(stderr.contains("missing source directory"), "{stderr}");
        assert!(!fields.contains_key(name));
        assert!(!root.join(relative).exists());
        assert!(!root.join("mesh").join(relative).exists());
    }
}

#[path = "source_layout_action/artifact_routing.rs"]
mod artifact_routing;

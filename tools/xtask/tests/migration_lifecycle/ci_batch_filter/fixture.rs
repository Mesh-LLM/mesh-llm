use super::workflow_yaml::{self, Node};
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) const WORKFLOWS: [&str; 2] = ["ci-quality-slice.yml", "ci-rust-tests-slice.yml"];
pub(super) fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
pub(super) fn document(file: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(repository().join(file)).unwrap()).unwrap()
}
pub(super) fn steps(workflow: &str) -> Vec<Node> {
    let doc = document(&format!(".github/workflows/{workflow}"));
    doc.get("jobs")
        .unwrap()
        .entries()
        .iter()
        .flat_map(|(_, job)| match job.get("steps") {
            Some(Node::Seq(steps)) => steps.clone(),
            _ => vec![],
        })
        .collect()
}
pub(super) fn batch_name(workflow: &str) -> &'static str {
    if workflow == WORKFLOWS[0] {
        "Run one Clippy invocation for the batch"
    } else {
        "Run isolated Cargo tests for the batch"
    }
}
fn script(workflow: &str, name: &str) -> String {
    steps(workflow)
        .iter()
        .find(|step| step.get("name").and_then(Node::text) == Some(name))
        .unwrap()
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .into()
}
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn bash() -> PathBuf {
    let fixed = PathBuf::from("/opt/homebrew/bin/bash");
    if fixed.is_file() {
        return fixed;
    }
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join("bash"))
        .find(|path| path.is_file())
        .unwrap()
        .canonicalize()
        .unwrap()
}
pub(super) struct Fixture {
    _temporary: tempfile::TempDir,
    pub root: PathBuf,
}
impl Fixture {
    pub fn new(
        workflow: &str,
        requested: &[&str],
        members: &[&str],
        fails: bool,
        translation: bool,
    ) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        for path in ["bin", "calls", "tmp"] {
            fs::create_dir(root.join(path)).unwrap();
        }
        let metadata = json!({"packages":members.iter().map(|name|json!({"name":name,"id":name,"version":"0.0.0","manifest_path":root.join(name).join("Cargo.toml")})).collect::<Vec<_>>(),"workspace_members":members});
        fs::write(
            root.join("metadata.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        if fails {
            fs::write(root.join("fail-metadata"), b"").unwrap();
        }
        executable(
            &root.join("bin/cargo"),
            r#"#!/bin/bash
set -euo pipefail
index=$(find calls -type f | wc -l | tr -d ' ')
for arg in "$@"; do printf '%s\0' "$arg"; done > "calls/$index"
if [[ "$1" == metadata ]]; then
  [[ ! -f fail-metadata ]] || { echo 'fixture metadata unavailable' >&2; exit 101; }
  cat metadata.json
fi
"#,
        );
        fs::write(
            root.join("resolve.sh"),
            script(
                workflow,
                "Resolve planned batch crates against the checked-out workspace",
            ),
        )
        .unwrap();
        fs::write(
            root.join("batch.sh"),
            script(workflow, batch_name(workflow)),
        )
        .unwrap();
        let translate = if translation {
            let action = document(".github/actions/resolve-cargo-packages/action.yml");
            let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
                panic!("action steps")
            };
            steps[0].get("run").and_then(Node::text).unwrap().to_owned()
        } else {
            String::new()
        };
        fs::write(root.join("translate.sh"), translate).unwrap();
        fs::write(
            root.join("requested.json"),
            serde_json::to_vec(&requested).unwrap(),
        )
        .unwrap();
        let mode = if translation {
            "source ./translate.sh\nPLANNED_BATCH_CRATES=$(sed -n 's/^crates=//p' outputs)\n: > outputs"
        } else {
            "PLANNED_BATCH_CRATES=$REQUESTED_CRATES"
        };
        fs::write(root.join("driver.sh"),format!("set -euo pipefail\nexport REQUESTED_CRATES=$(cat requested.json)\nexport PLANNED_BATCHES='' PACKAGE_GENERATION=legacy\n{mode}\nexport PLANNED_BATCH_CRATES\nsource ./resolve.sh\nresolved=$(sed -n 's/^crates=//p' outputs)\nexport CLIPPY_CRATES=$resolved TEST_CRATES=$resolved\nsource ./batch.sh\n")).unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }
    pub fn run(&self) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!(
                    "{}:{}",
                    self.root.join("bin").display(),
                    std::env::var("PATH").unwrap()
                ),
            ),
            ("HOME", self.root.display().to_string()),
            ("RUNNER_TEMP", self.root.join("tmp").display().to_string()),
            (
                "GITHUB_OUTPUT",
                self.root.join("outputs").display().to_string(),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN",
                env!("CARGO_BIN_EXE_xtask").into(),
            ),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: bash(),
                cwd: self.root.clone(),
                environment,
                arguments: vec![Value::Public(self.root.join("driver.sh").into_os_string())],
            },
            &Limits {
                execution: Duration::from_secs(8),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            report.process.success(),
            "{:?}: {}",
            report.process,
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
        );
        for call in self.calls().iter().filter(|call| call[0] != "metadata") {
            match call[0].as_str() {
                "clippy" => {
                    assert!(call.iter().any(|arg| arg == "--all-targets"));
                    assert!(call.windows(2).any(|pair| pair == ["-D", "warnings"]));
                }
                "test" => assert!(call.iter().any(|arg| arg == "--locked")),
                other => panic!("unexpected build operation {other}"),
            }
        }
        report
    }
    pub fn calls(&self) -> Vec<Vec<String>> {
        (0..fs::read_dir(self.root.join("calls")).unwrap().count())
            .map(|index| {
                fs::read(self.root.join(format!("calls/{index}")))
                    .unwrap()
                    .split(|byte| *byte == 0)
                    .filter(|arg| !arg.is_empty())
                    .map(|arg| String::from_utf8(arg.to_vec()).unwrap())
                    .collect()
            })
            .collect()
    }
    pub fn packages(&self) -> Vec<String> {
        self.calls()
            .iter()
            .filter(|args| args[0] != "metadata")
            .flat_map(|args| {
                args.windows(2)
                    .filter(|pair| pair[0] == "-p")
                    .map(|pair| pair[1].clone())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

//! Execute the maintained smoke adapter with finite compiler/test observers.
use super::super::{
    support::{self, Fixture},
    workflow_yaml::{self, Node},
};
use serde_json::json;
use std::{
    fs,
    os::unix::fs::symlink,
    process::{Command, Output},
};

pub(super) const HOST: &str =
    "inference::skippy::resolver::tests::safetensors_checkpoint_reaches_mesh_host_runtime";
pub(super) const ADAPTER: &str =
    "config::hardware_translation_tests::safetensors_checkpoint_reaches_mesh_host_runtime";

pub(super) fn script(name: &str) -> String {
    let document = workflow_yaml::parse(
        &fs::read_to_string(support::root().join(".github/workflows/ci-rust-tests-slice.yml"))
            .unwrap(),
    )
    .unwrap();
    let Node::Seq(steps) = document
        .get("jobs")
        .unwrap()
        .get("safetensors_runtime_smoke")
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("smoke steps")
    };
    let matches = steps
        .iter()
        .filter(|step| step.get("name").and_then(Node::text) == Some(name))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    matches[0].get("run").unwrap().text().unwrap().to_owned()
}

pub(super) fn fixture(adapter: bool, failure: &str) -> Fixture {
    let fixture = Fixture::new();
    for tool in ["jq", "tail", "grep"] {
        let source = ["/usr/bin", "/bin", "/opt/homebrew/bin", "/usr/local/bin"]
            .into_iter()
            .map(|dir| std::path::Path::new(dir).join(tool))
            .find(|path| path.is_file())
            .unwrap();
        symlink(source, fixture.path().join("bin").join(tool)).unwrap();
    }
    let name = if adapter {
        "mesh-llm-skippy-adapter"
    } else {
        "mesh-llm-host-runtime"
    };
    let target = name.replace('-', "_");
    let metadata = json!({"workspace_members":["member"], "packages":[{"id":"member", "name":name}, {"id":"not-member", "name":"mesh-llm-skippy-adapter"}]});
    fs::write(fixture.path().join("metadata.json"), metadata.to_string()).unwrap();
    let executable = fixture.path().join("bin/smoke-observer");
    let artifact = |target: &str, test: bool, path: Option<&std::path::Path>| json!({"reason":"compiler-artifact", "target":{"name":target}, "profile":{"test":test}, "executable":path});
    let selected_target = if failure == "wrong-target" {
        "unrelated_target"
    } else {
        &target
    };
    let selected_path = if failure == "missing-binary" {
        None
    } else {
        Some(executable.as_path())
    };
    let rows = [
        json!({"reason":"compiler-message", "message":{"rendered":"finite compiler diagnostic"}}),
        artifact("unrelated_target", true, Some(executable.as_path())),
        artifact(selected_target, failure != "not-test", selected_path),
    ];
    fs::write(
        fixture.path().join("artifacts.jsonl"),
        rows.iter()
            .map(|row| row.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    fixture.executable("cargo", r#"printf '%s\n' "$*" >> cargo-calls
if [[ "$1" == metadata ]]; then
  [[ "$*" == 'metadata --locked --no-deps --format-version=1' ]] || exit 91
  /bin/cat metadata.json
elif [[ "$1" == test ]]; then
  [[ "$*" == "test --locked -p $EXPECTED_CRATE --no-default-features --lib --no-run --message-format=json" ]] || exit 92
  /bin/cat artifacts.jsonl
  [[ "$FAILURE" != compile ]] || exit 43
else exit 93
fi"#);
    fixture.executable("smoke-observer", r#"if [[ "$1" == --list ]]; then
  [[ "$*" == '--list --ignored --format terse' ]] || exit 94
  [[ "$FAILURE" != listing ]] || exit 44
  [[ "$FAILURE" != absent-test ]] || { printf 'unrelated: test\n'; exit 0; }
  printf '%s: test\n' "$EXPECTED_TEST"
else
  [[ "$*" == "$EXPECTED_TEST --exact --ignored --nocapture" ]] || exit 95
  [[ "$SKIPPY_SAFETENSORS_SMOKE_DIR" == "$RUNNER_TEMP/safetensors-smollm2" && "$SKIPPY_SAFETENSORS_SMOKE_IMATRIX" == synthetic ]] || exit 96
  printf '%s\n' "$SKIPPY_SAFETENSORS_SMOKE_QUANTIZATION" >> measured-calls
  [[ "$SKIPPY_SAFETENSORS_SMOKE_QUANTIZATION" != "$FAIL_QUANTIZATION" ]] || exit 45
fi"#);
    fixture
}

pub(super) fn run(
    fixture: &Fixture,
    name: &str,
    adapter: bool,
    failure: &str,
    quantizations: &str,
    fail_quantization: &str,
) -> Output {
    let mut command = Command::new("/bin/bash");
    command.env_clear().current_dir(fixture.path());
    command.env("PATH", fixture.path().join("bin"));
    command
        .env("HOME", fixture.path())
        .env("TMPDIR", fixture.path())
        .env("RUNNER_TEMP", fixture.path());
    command.env("GITHUB_OUTPUT", fixture.path().join("outputs"));
    command.env(
        "EXPECTED_CRATE",
        if adapter {
            "mesh-llm-skippy-adapter"
        } else {
            "mesh-llm-host-runtime"
        },
    );
    command.env("EXPECTED_TEST", if adapter { ADAPTER } else { HOST });
    command
        .env("FAILURE", failure)
        .env("FAIL_QUANTIZATION", fail_quantization);
    command.env(
        "SAFETENSORS_SMOKE_TEST_BINARY",
        fixture.path().join("bin/smoke-observer"),
    );
    command.env(
        "SAFETENSORS_SMOKE_TEST_NAME",
        if adapter { ADAPTER } else { HOST },
    );
    command.env("SAFETENSORS_QUANTIZATIONS_JSON", quantizations);
    command.args(["-c", &script(name)]);
    fixture.run(command)
}

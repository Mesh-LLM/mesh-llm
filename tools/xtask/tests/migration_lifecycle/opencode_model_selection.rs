//! Actual OpenCode model-selection prefix using the existing Rust controller.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt, path::PathBuf,
    time::Duration,
};

const SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../scripts/ci-opencode-smoke.sh"
));

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new(models: serde_json::Value) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("OpenCode model selection");
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::create_dir(root.join("scripts")).unwrap();
        let prefix = SOURCE
            .split_once("CONFIG_BASE_URL=\"$MESH_BASE_URL\"")
            .unwrap()
            .0;
        assert!(prefix.contains("automation agent-pick-model"));
        fs::write(
            root.join("scripts/selection.sh"),
            format!("{prefix}printf '%s\\n' \"$MODEL\" \"$MESH_MODEL\"\n"),
        )
        .unwrap();
        fs::write(
            root.join("models.json"),
            serde_json::to_vec(&models).unwrap(),
        )
        .unwrap();
        for (name, body) in [
            (
                "curl",
                "printf 'curl\\n' >> \"$TRACE\"\n/bin/cat \"$MODELS_FIXTURE\"\n",
            ),
            (
                "opencode",
                "printf 'unexpected opencode\\n' >> \"$TRACE\"\nexit 93\n",
            ),
            (
                "python3",
                "printf 'unexpected Python\\n' >> \"$TRACE\"\nexit 92\n",
            ),
        ] {
            let path = root.join("bin").join(name);
            fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        Self {
            _temporary: temporary,
            root,
        }
    }

    fn run(&self, values: &[(&str, &str)]) -> (bool, Vec<String>, String) {
        self.run_named("selection.sh", values)
    }

    fn run_named(&self, script: &str, values: &[(&str, &str)]) -> (bool, Vec<String>, String) {
        let path = format!("{}:/usr/bin:/bin", self.root.join("bin").display());
        let mut environment = BTreeMap::from([
            ("PATH".into(), Value::Public(path.into())),
            (
                "HOME".into(),
                Value::Public(self.root.clone().into_os_string()),
            ),
            (
                "TRACE".into(),
                Value::Public(self.root.join("trace").into_os_string()),
            ),
            (
                "MODELS_FIXTURE".into(),
                Value::Public(self.root.join("models.json").into_os_string()),
            ),
            (
                "OPENCODE_SMOKE_WORK_DIR".into(),
                Value::Public(self.root.clone().into_os_string()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
        ]);
        for (key, value) in values {
            environment.insert((*key).into(), Value::Public((*value).into()));
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: vec![Value::Public(
                    self.root.join("scripts").join(script).into_os_string(),
                )],
                environment,
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
        assert!(
            report.process.failure.is_none() && report.process.cleanup.complete,
            "{report:?}"
        );
        let output = String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap();
        let trace = fs::read_to_string(self.root.join("trace")).unwrap_or_default();
        assert!(!trace.contains("unexpected"), "{trace}");
        (
            report.process.status.unwrap().success(),
            output.lines().map(str::to_owned).collect(),
            trace,
        )
    }
}

#[test]
fn preferred_family_order_stable_response_order_and_first_nonempty_fallback() {
    for (models, expected) in [
        (
            serde_json::json!({"data":[{"id":"Qwen-coder"},{"id":"GLM-4"},{"id":"MiniMax-2"}]}),
            "MiniMax-2",
        ),
        (
            serde_json::json!({"data":[{"id":"qwen-first"},{"id":"QWEN-second"}]}),
            "qwen-first",
        ),
        (
            serde_json::json!({"data":[{}, {"id":""},{"id":"plain-first"},{"id":"plain-second"}]}),
            "plain-first",
        ),
    ] {
        let fixture = Fixture::new(models);
        let (ok, output, trace) = fixture.run(&[]);
        assert!(ok);
        assert_eq!(output, [format!("mesh/{expected}"), expected.to_owned()]);
        assert_eq!(trace, "curl\n");
    }
}

#[test]
fn empty_and_malformed_rosters_cannot_select_a_model() {
    for models in [
        serde_json::json!({"data":[]}),
        serde_json::json!({"data":[{"id":true}]}),
    ] {
        let fixture = Fixture::new(models);
        let (ok, output, trace) = fixture.run(&[]);
        assert!(!ok && output.is_empty());
        assert_eq!(trace, "curl\n");
    }
}

#[test]
fn explicit_client_and_mesh_model_overrides_remain_authoritative() {
    let fixture = Fixture::new(serde_json::json!({"data":[{"id":"MiniMax-2"}]}));
    let (ok, output, trace) = fixture.run(&[("OPENCODE_SMOKE_MODEL", "mesh/selected")]);
    assert!(ok);
    assert_eq!(output, ["mesh/selected", "selected"]);
    assert!(trace.is_empty());
    let (ok, output, trace) = fixture.run(&[("MESH_OPENCODE_MODEL", "selected")]);
    assert!(ok);
    assert_eq!(output, ["mesh/selected", "selected"]);
    assert_eq!(trace, "curl\n");
}

#[test]
fn captured_surface_call_uses_the_frozen_owner_and_rejects_incomplete_evidence() {
    let fixture = Fixture::new(serde_json::json!({"data":[]}));
    let selector = SOURCE
        .split_once("# Frozen automation selection ends.")
        .unwrap()
        .0;
    let call = SOURCE
        .lines()
        .find(|line| line.contains("automation agent-fixture-evidence surface"))
        .unwrap();
    fs::write(
        fixture.root.join("scripts/surface.sh"),
        format!("{selector}\n{call}\n"),
    )
    .unwrap();
    let body = serde_json::json!({"stream":true,"tools":[{"type":"function"}],"tool_choice":"auto","parallel_tool_calls":true,"messages":[{"role":"system"},{"role":"user"},{"role":"assistant","tool_calls":[{"id":"fixture"}]},{"role":"tool"}]});
    let mut second = body.clone();
    second["stream"] = serde_json::json!(false);
    let rows = [
        serde_json::json!({"method":"GET","path":"/v1/models"}),
        serde_json::json!({"method":"POST","path":"/v1/chat/completions","body":body}),
        serde_json::json!({"method":"POST","path":"/v1/chat/completions","body":second}),
    ];
    let capture = fixture.root.join("capture.jsonl");
    for start in [0, 1] {
        fs::write(
            &capture,
            rows[start..]
                .iter()
                .map(serde_json::Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let (ok, output, trace) = fixture.run_named(
            "surface.sh",
            &[
                ("SURFACE_LOG", capture.to_str().unwrap()),
                ("LONG_PROMPT_CHARS", "0"),
            ],
        );
        assert!(trace.is_empty());
        if start == 0 {
            assert!(ok);
            assert_eq!(
                output,
                [
                    "OpenAI agent surface validation passed",
                    "  captured requests: models=1 chat=2"
                ]
            );
        } else {
            assert!(!ok && output.is_empty());
        }
    }
}

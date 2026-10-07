//! The actual complete WAN entrypoint, with finite tools and no acquisition.
use super::{ENTRYPOINT, executable, invoke, stdout};
use std::{fs, path::PathBuf};

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    original: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        fs::create_dir(root.join("bin")).unwrap();
        let original = br#"{
  "model_id": "org/admitted-model", "lane_count": 2,
  "layer_start": 12, "layer_end": 24, "activation_codec": "bf16-rne-v1",
  "source_model_sha256": "admitted-source",
  "resident_tensor_names": ["blk.12.attn_q.weight"],
  "execution_contract": "admitted-contract",
  "activation_import_identities": ["frontier/layer-12"],
  "activation_export_identities": ["frontier/layer-24"]
}

"#
        .to_vec();
        fs::write(root.join("admitted stage.json"), &original).unwrap();
        fs::write(root.join("entrypoint.sh"), ENTRYPOINT).unwrap();
        executable(
            &root.join("bin/skippy"),
            "#!/bin/bash\nset -euo pipefail\n[[ $1 == serve ]] || exit 91\nprintf '%s\\0' \"$@\" > \"$FIXTURE_ROOT/launch.argv\"\nprintf 'owned serve stdout\\n'\nexit \"${OWNED_SERVE_STATUS:-0}\"\n",
        );
        executable(
            &root.join("bin/jq"),
            "#!/bin/bash\nset -euo pipefail\n[[ $# == 3 && $3 == \"$FIXTURE_ROOT/admitted stage.json\" && -f $3 ]] || exit 92\ncase \"$1:$2\" in\n '-er:.model_id') printf 'org/admitted-model\\n' ;;\n '-er:.lane_count') printf '2\\n' ;;\n '-er:.layer_start') printf '12\\n' ;;\n '-er:.layer_end') printf '24\\n' ;;\n '-r:.activation_codec // \"raw-f32-v1\"') printf 'bf16-rne-v1\\n' ;;\n *) exit 93 ;;\nesac\n",
        );
        for tool in [
            "skippy-package-builder",
            "tc",
            "curl",
            "nc",
            "metrics-server",
        ] {
            executable(
                &root.join("bin").join(tool),
                "#!/bin/bash\nprintf '%s\\n' \"$0\" >> \"$FIXTURE_ROOT/forbidden-helper\"\nexit 94\n",
            );
        }
        Self {
            _directory: directory,
            root,
            original,
        }
    }
    fn run(&self, index: u8, controls: &str) -> super::process::RawProcessReport {
        invoke(
            &self.root,
            &format!(
                "set -euo pipefail\nexport CONFIG_PATH=\"$FIXTURE_ROOT/admitted stage.json\"\nexport STAGE_INDEX={index} STAGE_COUNT=4 WAN_ENABLE=0\n{controls}\nexec /bin/bash \"$FIXTURE_ROOT/entrypoint.sh\" stage\n"
            ),
        )
    }
    fn args(&self) -> Vec<String> {
        let bytes = fs::read(self.root.join("launch.argv")).unwrap();
        assert_eq!(bytes.last(), Some(&0));
        bytes[..bytes.len() - 1]
            .split(|byte| *byte == 0)
            .map(|arg| String::from_utf8(arg.to_vec()).unwrap())
            .collect()
    }
    fn unchanged(&self) {
        assert_eq!(
            fs::read(self.root.join("admitted stage.json")).unwrap(),
            self.original
        );
        assert!(!self.root.join("forbidden-helper").exists());
    }
    fn base_args(&self, codec: &str) -> Vec<String> {
        ["serve", "--stage-transport", "binary", "--config"]
            .into_iter()
            .map(str::to_owned)
            .chain([self
                .root
                .join("admitted stage.json")
                .to_string_lossy()
                .into_owned()])
            .chain(
                [
                    "--activation-codec",
                    codec,
                    "--metrics-otlp-grpc",
                    "http://metrics:14317",
                    "--telemetry-queue-capacity",
                    "4096",
                    "--telemetry-level",
                    "debug",
                    "--max-inflight",
                    "2",
                ]
                .into_iter()
                .map(str::to_owned),
            )
            .collect()
    }
}
#[test]
fn actual_entrypoint_admitted_config_preserves_bytes_and_override_without_acquisition() {
    let fixture = Fixture::new();
    let report = fixture.run(1, "export MODEL_PATH=\"$FIXTURE_ROOT/missing-model.gguf\" MODEL_PACKAGE_REF='hf://unused/package@main' CONFIG_DIR=\"$FIXTURE_ROOT/should-not-be-created\" N_BATCH=999 ACTIVATION_WIRE_DTYPE=f16");
    assert!(report.process.status.unwrap().success());
    assert_eq!(stdout(&report), b"owned serve stdout\n");
    let mut expected = fixture.base_args("f16-rne-v1");
    expected.push("--worker-only".into());
    assert_eq!(fixture.args(), expected);
    fixture.unchanged();
    assert!(!fixture.root.join("should-not-be-created").exists());
}
#[test]
fn actual_entrypoint_head_preserves_public_api_and_prefill_controls() {
    let fixture = Fixture::new();
    let report = fixture.run(0, "export OPENAI_BIND_ADDR=127.0.0.1:19437 OPENAI_DEFAULT_MAX_TOKENS=41 OPENAI_GENERATION_CONCURRENCY=3 OPENAI_PREFILL_CHUNK_SIZE=192 OPENAI_PREFILL_CHUNK_POLICY=fixed");
    assert!(report.process.status.unwrap().success());
    assert_eq!(stdout(&report), b"owned serve stdout\n");
    let mut expected = fixture.base_args("bf16-rne-v1");
    expected.extend(
        [
            "--bind-addr",
            "127.0.0.1:19437",
            "--model-id",
            "org/admitted-model",
            "--default-max-tokens",
            "41",
            "--generation-concurrency",
            "3",
            "--prefill-chunk-size",
            "192",
            "--prefill-chunk-policy",
            "fixed",
            "--prefill-adaptive-start",
            "128",
            "--prefill-adaptive-step",
            "128",
            "--prefill-adaptive-max",
            "512",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    assert_eq!(fixture.args(), expected);
    fixture.unchanged();
}
#[test]
fn actual_entrypoint_worker_excludes_public_api_and_forwards_child_exit() {
    let fixture = Fixture::new();
    let report = fixture.run(
        1,
        "export OPENAI_BIND_ADDR=127.0.0.1:19437 OWNED_SERVE_STATUS=43",
    );
    assert_eq!(report.process.status.unwrap().code(), Some(43));
    assert_eq!(stdout(&report), b"owned serve stdout\n");
    let mut expected = fixture.base_args("bf16-rne-v1");
    expected.push("--worker-only".into());
    assert_eq!(fixture.args(), expected);
    fixture.unchanged();
}
#[test]
fn actual_entrypoint_unsupported_dtype_refuses_before_launch_or_acquisition() {
    let fixture = Fixture::new();
    let report = fixture.run(0, "export ACTIVATION_WIRE_DTYPE=q8");
    assert_eq!(report.process.status.unwrap().code(), Some(64));
    assert!(stdout(&report).is_empty());
    let stderr = report.stderr.as_ref().unwrap().as_bytes();
    assert!(String::from_utf8_lossy(stderr).contains("unsupported ACTIVATION_WIRE_DTYPE: q8"));
    assert!(!fixture.root.join("launch.argv").exists());
    fixture.unchanged();
}

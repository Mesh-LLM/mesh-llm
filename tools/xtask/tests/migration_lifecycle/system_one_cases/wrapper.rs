use super::server;
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
#[test]
fn actual_smoke_case_wrapper_consumes_typed_status_and_per_mode_report_without_model_claim() {
    for mutation in ["", "wrong-code"] {
        let server = server::Server::new("contract", mutation);
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path().canonicalize().unwrap();
        let source = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/skippy-system-one-smoke.sh"),
        )
        .unwrap();
        let function = source
            .split("run_cases_against_stage() {\n")
            .nth(1)
            .unwrap()
            .split("\nprewarm_plan()")
            .next()
            .unwrap();
        let port = server.url.rsplit(':').next().unwrap();
        let body = format!(
            r#"set -euo pipefail
WORK_DIR="$PWD"
REPORT_DIR="$PWD"
ALIAS=openjev-latest
CTX_SIZE=8192
READ_N_GPU_LAYERS=''
REPORT_REASONS=''
automation=("$MESH_LLM_AUTOMATION_BIN")
require_smoke_binaries() {{ :; }}
artifact_summary() {{ printf '%s\n' '{{"repo":"fixture","revision":"pinned","file":"model.gguf","model_ref":"fixture-model","sha256":"fixture"}}'; }}
cached_artifact_path() {{ printf '%s/model.gguf\n' "$PWD"; }}
verify_artifact_digest() {{ :; }}
stage_layer_end() {{ printf '2\n'; }}
pick_port() {{ printf '{port}\n'; }}
write_stage_config() {{ : > "$1"; }}
start_stage_server() {{ test -f "$2"; }}
cleanup() {{ printf 'cleanup\n' >> cleanup.log; }}
: > model.gguf
run_cases_against_stage() {{
{function}
if run_cases_against_stage contract fixture '' contract 128 2; then
 printf 'wrapper passed\n'
else
 exit "$?"
fi
"#
        );
        fs::write(root.join("caller.sh"), body).unwrap();
        let environment = BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(std::env::var_os("PATH").unwrap()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
        ]);
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                arguments: vec![Value::Public("caller.sh".into())],
                cwd: root.clone(),
                environment,
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
            report.process.cleanup.complete && report.process.failure.is_none(),
            "{:?}",
            report.process
        );
        server.finish();
        let data: Json =
            serde_json::from_slice(&fs::read(root.join("system-one-contract.json")).unwrap())
                .unwrap();
        if mutation.is_empty() {
            assert!(report.process.success());
            assert_eq!(data["status"], "pass");
            assert_eq!(data["mode"], "contract");
            assert!(
                String::from_utf8_lossy(report.stdout.unwrap().as_bytes())
                    .contains("wrapper passed")
            );
        } else {
            assert_eq!(report.process.status.unwrap().code(), Some(1));
            assert_eq!(data["status"], "fail");
        }
        assert_eq!(
            fs::read_to_string(root.join("cleanup.log")).unwrap(),
            "cleanup\n"
        );
    }
}

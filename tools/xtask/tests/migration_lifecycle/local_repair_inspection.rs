//! Actual extracted local caller protocol; domain admission is owned by local_inspection units.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, os::unix::fs::PermissionsExt, path::PathBuf, time::Duration};
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source = fs::read_to_string(repo.join("scripts/llama-canary-agent-repair.sh")).unwrap();
        let helper = source
            .split("repair_source_inspection() {")
            .nth(1)
            .unwrap()
            .split("\nvalidate_agent_manifest_changes()")
            .next()
            .unwrap();
        let guard = source
            .split("  repair_workload_controller_unchanged() {")
            .nth(1)
            .unwrap()
            .split("\n# Legacy workload automation selection ends.")
            .next()
            .unwrap();
        fs::write(root.join("adapter.sh"),format!("set -euo pipefail\nrepair_workload_controller_unchanged() {{{guard}\nrepair_source_inspection() {{{helper}\nrun_verification_logged() {{ local label=\"$1\" log=\"$2\"; shift 2; printf '%s\\n' \"$label\" >> \"$log\"; \"$@\"; }}\n")).unwrap();
        fs::write(
            root.join("controller"),
            r#"#!/bin/bash
set -euo pipefail
[[ "$1" == automation && "$2" == canary-receipts && "$4" == --input ]] || exit 97
jq -c --arg verb "$3" '. + {verb:$verb}' "$5" >> "$REQUESTS" || exit 97
exit "${CONTROLLER_STATUS:-0}"
"#,
        )
        .unwrap();
        fs::set_permissions(root.join("controller"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::create_dir(root.join("scratch")).unwrap();
        Self { _temp: temp, root }
    }
    fn run(&self, body: &str, status: u8, digest: &str) -> process::RawProcessReport {
        let env = [
            ("PATH", std::env::var("PATH").unwrap()),
            ("TRUSTED_ROOT", self.root.display().to_string()),
            ("ROOT", self.root.display().to_string()),
            ("BASE_HEAD", "a".repeat(40)),
            ("CANDIDATE_BASE_HEAD", "a".repeat(40)),
            (
                "RUNNER_TEMP",
                self.root.join("scratch").display().to_string(),
            ),
            ("REQUESTS", self.root.join("requests").display().to_string()),
            ("CONTROLLER_STATUS", status.to_string()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect::<BTreeMap<_, _>>();
        process::supervise_raw(&ProcessSpec{executable:"/bin/bash".into(),cwd:self.root.clone(),environment:env,arguments:vec![Value::Public("-c".into()),Value::Public(format!("source ./adapter.sh\nrepair_workload_controller=\"$ROOT/controller\"\nrepair_workload_controller_sha='{digest}'\nrepair_workload_automation=(\"$repair_workload_controller\")\n{body}").into())]},&Limits{execution:Duration::from_secs(10),graceful_shutdown:Duration::from_secs(1),forced_shutdown:Duration::from_secs(1),retained_bytes_per_stream:65536,readiness:Readiness::None,completion:Completion::Exit},&Cancellation::default(),RawCaptureOptions{stdout:std::num::NonZeroUsize::new(65536),stderr:std::num::NonZeroUsize::new(65536)}).unwrap()
    }
    fn digest(&self) -> String {
        hex::encode(Sha256::digest(
            fs::read(self.root.join("controller")).unwrap(),
        ))
    }
    fn requests(&self) -> Vec<Json> {
        fs::read_to_string(self.root.join("requests"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn clean(&self) {
        assert_eq!(fs::read_dir(self.root.join("scratch")).unwrap().count(), 0);
    }
}
#[test]
fn actual_local_helper_preserves_identity_check_modes_and_logged_failure_cleanup() {
    let f = Fixture::new();
    let digest = f.digest();
    let result=f.run("repair_source_inspection local-manifest-policy\nrepair_source_inspection local-parity-inventory '' ./parity.log\nrepair_source_inspection local-split-roster false\nrepair_source_inspection local-split-roster true",0,&digest);
    assert!(result.process.success(), "{:?}", result.process);
    let requests = f.requests();
    assert_eq!(requests.len(), 4);
    for row in &requests {
        assert_eq!(row["authority"]["controller"]["revision"], "a".repeat(40));
        assert_eq!(row["authority"]["controller"]["executable_sha256"], digest);
        assert_eq!(
            row["authority"]["controller"]["root"],
            f.root.display().to_string()
        );
        assert_eq!(row["authority"]["base"], "a".repeat(40));
        assert!(row.get("context").is_none());
    }
    assert!(requests[0].get("check").is_none());
    assert!(requests[1].get("check").is_none());
    assert_eq!(requests[2]["check"], false);
    assert_eq!(requests[3]["check"], true);
    f.clean();
    let failed = f.run(
        "repair_source_inspection local-parity-inventory '' ./failed.log",
        7,
        &digest,
    );
    assert_eq!(
        failed.process.status.and_then(|status| status.code()),
        Some(7)
    );
    assert_eq!(f.requests().len(), 5);
    f.clean();
    let rejected = f.run(
        "repair_source_inspection local-manifest-policy",
        0,
        &"0".repeat(64),
    );
    assert!(!rejected.process.success());
    assert_eq!(f.requests().len(), 5);
    f.clean();
}

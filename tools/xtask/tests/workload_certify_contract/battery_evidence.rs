//! The battery must verify the same product executable its workload producer ran.
use super::{Fixture, fs, root};

fn function(source: &str, name: &str) -> String {
    let marker = format!("{name}() {{\n");
    let body = source
        .split_once(&marker)
        .unwrap()
        .1
        .split_once("\n}\n")
        .unwrap()
        .0;
    format!("{marker}{body}\n}}\n")
}

#[test]
fn battery_verifies_produced_skippy_and_propagates_rejected_evidence() {
    let source =
        fs::read_to_string(root().join("skippy/scripts/skippy-family-battery.sh")).unwrap();
    for reject in [false, true] {
        let fixture = Fixture::new();
        fs::create_dir(fixture.path().join("scripts")).unwrap();
        fixture.executable("scripts/skippy-workload-certify.sh", "exit 0");
        fixture.executable("candidate/skippy", "exit 0");
        let script = format!(
            "{}\n{}\n{}",
            function(&source, "verify_workload_oracle"),
            function(&source, "run_workload_certify"),
            FIXTURE,
        );
        let mut command = fixture.command();
        command
            .args(["-c", &script])
            .env("REJECT_EVIDENCE", if reject { "1" } else { "0" });
        let receipt = fixture.run(command);
        assert!(
            receipt.status.success(),
            "{}\n{}",
            receipt.stdout,
            receipt.stderr
        );
        assert_eq!(
            fs::read_to_string(fixture.path().join("verified"))
                .unwrap()
                .trim(),
            fixture.path().join("candidate/skippy").to_str().unwrap()
        );
    }
}

const FIXTURE: &str = r#"
set -euo pipefail
ROOT="$FIXTURE_ROOT" CERT_DIR="$FIXTURE_ROOT/work" RESULTS_JSONL="$FIXTURE_ROOT/results"
SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR="$FIXTURE_ROOT/candidate"
SKIPPY_WORKLOAD_ORACLE_SERVER=oracle SKIPPY_WORKLOAD_ORACLE_COMPLETION=oracle SKIPPY_WORKLOAD_ORACLE_TTS=oracle
TOTAL=0 CERT_FAILURE_COUNT=0 DRY_RUN=0 FAILURES=()
cert_timeout_for_startup() { echo 10; }
slugify() { echo "$1"; }
run_battery_timeout() { shift 2; "$@"; }
jq() { echo '{}'; }
owner() {
  if [[ "$1 $2 $3" == 'automation canary-receipts prepared-source' ]]; then
    printf '%040d\n' 0
    return
  fi
  [[ "$1 $2 $3" == 'automation workload-oracle-evidence verify' ]] || exit 91
  while (( $# > 0 )); do
    if [[ "$1" == --candidate-executable ]]; then
      [[ "$2" == "$FIXTURE_ROOT/candidate/skippy" && -x "$2" ]] || exit 92
      printf '%s\n' "$2" > "$FIXTURE_ROOT/verified"
      break
    fi
    shift
  done
  test -f "$FIXTURE_ROOT/verified"
  [[ "$REJECT_EVIDENCE" == 0 ]]
}
automation=(owner)
for class in embedding rerank encoder_decoder ocr speech_recognition speech_synthesis; do
  TOTAL=0 CERT_FAILURE_COUNT=0 FAILURES=()
  rm -f "$FIXTURE_ROOT/verified"
  run_workload_certify fixture "$class" "$FIXTURE_ROOT/model.gguf" fixture revision 5 1 "$class-smoke,$class-oracle" ''
  [[ "$CERT_FAILURE_COUNT" == "$REJECT_EVIDENCE" ]]
  [[ "${#FAILURES[@]}" == "$REJECT_EVIDENCE" ]]
  test -f "$FIXTURE_ROOT/verified"
done
"#;

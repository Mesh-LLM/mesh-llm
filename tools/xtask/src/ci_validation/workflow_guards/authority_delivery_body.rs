//! Exact existing delivery adapter bodies, separate from hosted authenticity.
use super::{Node, authority_sources::field};
const FREEZE: &str = "set -euo pipefail\nsha=\"$(git rev-parse HEAD)\"\n[[ \"$sha\" =~ ^[0-9a-f]{40}$ ]] || exit 1\ntest \"$sha\" = \"$GITHUB_SHA\" || exit 1\nprintf 'sha=%s\\n' \"$sha\" >> \"$GITHUB_OUTPUT\"\n";
const ADMISSION: &str = "set -euo pipefail\ntest \"$AUTOMATION_SOURCE_SHA\" = \"$EXPECTED_SOURCE_SHA\" || exit 1\ntest \"$RUNNER_OS\" = Linux || exit 1\ntest \"$RUNNER_ARCH\" = X64 || exit 1\n[[ \"$AUTOMATION_SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || exit 1\n[[ \"$AUTOMATION_ARTIFACT_ID\" =~ ^[1-9][0-9]{0,19}$ ]] || exit 1\n[[ \"$AUTOMATION_BINARY_SHA256\" =~ ^[0-9a-f]{64}$ ]] || exit 1\ntest \"$AUTOMATION_RESULTS_ORIGIN\" = 'https://results-receiver.actions.githubusercontent.com' || exit 1\nmkdir \"$RUNNER_TEMP/immutable-automation-restored\"\n";
const VERIFICATION: &str = "set -euo pipefail\n[[ \"$AUTOMATION_SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || exit 1\nif [[ -n \"${AUTOMATION_ARTIFACT_ID:-}\" || -n \"${AUTOMATION_BINARY_SHA256:-}\" ]]; then\n  [[ \"$AUTOMATION_ARTIFACT_ID\" =~ ^[1-9][0-9]{0,19}$ ]] || exit 1\n  [[ \"$AUTOMATION_BINARY_SHA256\" =~ ^[0-9a-f]{64}$ ]] || exit 1\nfi\ncd \"$RUNNER_TEMP/immutable-automation-restored\"\nbinary=xtask\nif [[ \"$RUNNER_OS\" == Windows ]]; then binary=xtask.exe; fi\nshopt -s nullglob dotglob\nmembers=(*)\n[[ ${#members[@]} == 3 ]] || exit 1\nfor member in \"$binary\" source.txt SHA256SUMS; do\n  [[ -f \"$member\" && ! -L \"$member\" ]] || exit 1\ndone\n[[ \"$(wc -c < source.txt | tr -d '[:space:]')\" == 41 ]] || exit 1\nIFS= read -r source_line < source.txt || exit 1\n[[ \"$source_line\" == \"$AUTOMATION_SOURCE_SHA\" ]] || exit 1\nchecksum_pattern='^([0-9a-f]{64}) ([ *])([a-zA-Z0-9.]+)$'\n[[ \"$(wc -c < SHA256SUMS | tr -d '[:space:]')\" -le 256 ]] || exit 1\nseen_binary=0\nseen_source=0\nwhile IFS= read -r line || [[ -n \"$line\" ]]; do\n  [[ \"$line\" =~ $checksum_pattern ]] || exit 1\n  digest=\"${BASH_REMATCH[1]}\"\n  name=\"${BASH_REMATCH[3]}\"\n  case \"$name\" in\n    \"$binary\")\n      [[ \"$seen_binary\" == 0 ]] || exit 1\n      seen_binary=1\n      if [[ -n \"${AUTOMATION_BINARY_SHA256:-}\" ]]; then\n        [[ \"$digest\" == \"$AUTOMATION_BINARY_SHA256\" ]] || exit 1\n      fi\n      ;;\n    source.txt) [[ \"$seen_source\" == 0 ]] || exit 1; seen_source=1 ;;\n    *) exit 1 ;;\n  esac\ndone < SHA256SUMS\n[[ \"$seen_binary\" == 1 && \"$seen_source\" == 1 ]] || exit 1\nif command -v sha256sum >/dev/null 2>&1; then\n  sha256sum --strict -c SHA256SUMS\nelse\n  shasum -a 256 -c SHA256SUMS\nfi\ntest -s \"$binary\" || exit 1\nchmod +x \"$binary\"\nbinary_path=\"$PWD/$binary\"\nif command -v cygpath >/dev/null 2>&1; then\n  binary_path=\"$(cygpath -m \"$binary_path\")\"\nfi\nprintf 'MESH_LLM_AUTOMATION_BIN=%s\\n' \"$binary_path\" >> \"$GITHUB_ENV\"\n";
const ORIGIN: &str = "set -euo pipefail\n[[ \"$AUTOMATION_SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || exit 1\ntest \"$AUTOMATION_SOURCE_SHA\" = \"$EXPECTED_SOURCE_SHA\" || exit 1\ncase \"${ACTIONS_RESULTS_URL:-}\" in\n  https://results-receiver.actions.githubusercontent.com|https://results-receiver.actions.githubusercontent.com/) ;;\n  *) echo \"Unexpected trusted artifact service origin\" >&2; exit 1 ;;\nesac\nprintf 'origin=https://results-receiver.actions.githubusercontent.com\\n' >> \"$GITHUB_OUTPUT\"\n";
fn normalized(value: &str) -> String {
    value
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
pub(super) fn check(freeze: &Node, admission: &Node, verification: &Node) -> Result<(), String> {
    for (step, expected) in [
        (freeze, FREEZE),
        (admission, ADMISSION),
        (verification, VERIFICATION),
    ] {
        if field(step, "run").map(normalized) != Some(normalized(expected)) {
            return Err("authority delivery adapter body changed or bypassed".into());
        }
    }
    Ok(())
}
pub(super) fn origin(
    step: &Node,
    producer: &Node,
    marker: &Node,
    admission: &Node,
    transport: &Node,
) -> Result<(), String> {
    const OUTPUT: &str =
        "${{ needs.authority_automation.outputs.authority_automation_results_origin }}";
    let fail = || "trusted artifact origin admission/step scope changed".to_owned();
    if field(step, "shell") != Some("bash")
        || field(step, "run").map(normalized) != Some(normalized(ORIGIN))
        || field(
            producer.get("outputs").ok_or_else(fail)?,
            "authority_automation_results_origin",
        ) != Some("${{ steps.authority_origin.outputs.origin }}")
    {
        return Err(fail());
    }
    let env = step.get("env").ok_or_else(fail)?;
    if env.entries().len() != 2
        || field(env, "EXPECTED_SOURCE_SHA") != Some("${{ github.sha }}")
        || field(env, "AUTOMATION_SOURCE_SHA") != Some("${{ steps.authority_source.outputs.sha }}")
    {
        return Err(fail());
    }
    if field(
        admission.get("env").ok_or_else(fail)?,
        "AUTOMATION_RESULTS_ORIGIN",
    ) != Some(OUTPUT)
    {
        return Err(fail());
    }
    let env = transport.get("env").ok_or_else(fail)?;
    if env.entries().len() != 1 || field(env, "ACTIONS_RESULTS_URL") != Some(OUTPUT) {
        return Err(fail());
    }
    if marker.get("env").is_some() || producer.get("env").is_some() {
        return Err(fail());
    }
    for owned in [step, transport] {
        if ["if", "continue-on-error", "working-directory"]
            .iter()
            .any(|k| owned.get(k).is_some())
        {
            return Err(fail());
        }
    }
    Ok(())
}

const QUALITY_ORIGIN: &str = "set -euo pipefail\n[[ \"$AUTOMATION_SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || exit 1\ntest \"$AUTOMATION_SOURCE_SHA\" = \"$(git rev-parse HEAD)\" || exit 1\ncase \"${ACTIONS_RESULTS_URL:-}\" in\n  https://results-receiver.actions.githubusercontent.com|https://results-receiver.actions.githubusercontent.com/) ;;\n  *) echo \"Unexpected trusted artifact service origin\" >&2; exit 1 ;;\nesac\nprintf 'origin=https://results-receiver.actions.githubusercontent.com\\n' >> \"$GITHUB_OUTPUT\"\n";
const QUALITY_ADMISSION: &str = "set -euo pipefail\ntest \"$RUNNER_OS\" = Linux || exit 1\ntest \"$RUNNER_ARCH\" = X64 || exit 1\n[[ \"$AUTOMATION_SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || exit 1\n[[ \"$AUTOMATION_ARTIFACT_ID\" =~ ^[1-9][0-9]{0,19}$ ]] || exit 1\n[[ \"$AUTOMATION_BINARY_SHA256\" =~ ^[0-9a-f]{64}$ ]] || exit 1\ntest \"$AUTOMATION_RESULTS_ORIGIN\" = 'https://results-receiver.actions.githubusercontent.com' || exit 1\nmkdir \"$RUNNER_TEMP/immutable-automation-restored\"\n";
pub(super) fn quality_producer(
    job: &Node,
    steps: &[Node],
    freeze: usize,
    upload: usize,
) -> Result<(), String> {
    let found: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| field(s, "id") == Some("authority_origin"))
        .collect();
    let [(index, step)] = found.as_slice() else {
        return Err("unique protected origin step required".into());
    };
    if !(freeze < *index && *index < upload)
        || field(step, "shell") != Some("bash")
        || field(step, "if") != Some("steps.sentinel_policy.outputs.depot_enabled == 'true'")
        || field(step, "run").map(normalized) != Some(normalized(QUALITY_ORIGIN))
    {
        return Err("protected hosted origin custody changed".into());
    }
    let outputs = job.get("outputs").ok_or("protected origin output absent")?;
    if field(outputs, "authority_automation_results_origin")
        != Some("${{ steps.authority_origin.outputs.origin }}")
    {
        return Err("protected origin output projection changed".into());
    }
    let env = step.get("env").ok_or("protected origin identity absent")?;
    if env.entries().len() != 3
        || field(env, "AUTOMATION_SOURCE_SHA") != Some("${{ steps.authority_source.outputs.sha }}")
        || field(env, "GIT_MASTER") != Some("1")
        || field(env, "GIT_OPTIONAL_LOCKS") != Some("0")
    {
        return Err("protected origin source identity changed".into());
    }
    if job.get("env").is_some() || step.get("continue-on-error").is_some() {
        return Err("protected origin cannot be globally overridden or bypassed".into());
    }
    Ok(())
}
pub(super) fn quality_consumer(job: &Node, steps: &[Node], attest: usize) -> Result<(), String> {
    let find = |name: &str| -> Result<(usize, &Node), String> {
        let found: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| field(s, "name") == Some(name))
            .collect();
        let [found] = found.as_slice() else {
            return Err("unique protected delivery step required".into());
        };
        Ok(*found)
    };
    let (admit, admission) = find("Admit protected automation producer identity")?;
    let (download, transport) = find("Download protected authority automation")?;
    if !(admit < download && download < attest)
        || field(admission, "shell") != Some("bash")
        || field(admission, "run").map(normalized) != Some(normalized(QUALITY_ADMISSION))
    {
        return Err("protected origin must be admitted before artifact download".into());
    }
    const OUTPUT: &str = "${{ needs.runner_policy.outputs.authority_automation_results_origin }}";
    if field(
        admission
            .get("env")
            .ok_or("protected origin binding absent")?,
        "AUTOMATION_RESULTS_ORIGIN",
    ) != Some(OUTPUT)
    {
        return Err("protected consumer origin projection changed".into());
    }
    let env = transport
        .get("env")
        .ok_or("protected download origin absent")?;
    if env.entries().len() != 1 || field(env, "ACTIONS_RESULTS_URL") != Some(OUTPUT) {
        return Err("protected download must use only admitted hosted origin".into());
    }
    if job.get("env").is_some()
        || steps[attest].get("env").is_some()
        || [admission, transport]
            .iter()
            .any(|s| s.get("if").is_some() || s.get("continue-on-error").is_some())
    {
        return Err("protected delivery origin scope or bypass changed".into());
    }
    Ok(())
}

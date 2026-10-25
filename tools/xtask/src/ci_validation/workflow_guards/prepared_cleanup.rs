//! Persistent cleanup custody for the four source-built native workflow phases.
use super::{Node, field};
use crate::command::DynResult;

fn lines(run: &str) -> Vec<&str> {
    run.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

pub(super) fn invocation(run: &str) -> bool {
    let actual = lines(run);
    let prefix = lines(CLEANUP_PREFIX);
    actual.len() == prefix.len() + 1
        && actual[..prefix.len()] == prefix
        && matches!(
            actual.last().copied(),
            Some(
                "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops runner-cleanup --job runner-contract --evidence-uploaded false"
                    | "\"$MESH_LLM_AUTOMATION_BIN\" ci-ops runner-cleanup --job smoke --evidence-uploaded false"
            )
        )
}

pub(super) fn check(job: &Node) -> DynResult<()> {
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("prepared cleanup requires owning steps".into());
    };
    let named = |name| {
        let found: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, step)| field(step, "name") == Some(name))
            .collect();
        match found.as_slice() {
            [(index, step)] => Ok((*index, *step)),
            _ => Err("prepared cleanup phase must exist exactly once"),
        }
    };
    let (prepare_index, prepare) =
        named("Prepare source-bound native cleanup before managed work")?;
    let (admit_index, admit) = named("Admit cleanup controller before managed work")?;
    let cleanup_index = steps.len().checked_sub(1).ok_or("cleanup missing")?;
    let work_index = admit_index + 1;
    if prepare_index + 1 != admit_index || work_index + 1 != cleanup_index {
        return Err(
            "prepare and custody admission must precede exactly one managed phase and cleanup"
                .into(),
        );
    }
    for step in [prepare, admit, &steps[work_index], &steps[cleanup_index]] {
        if field(step, "continue-on-error").is_some_and(|value| value != "false")
            || field(step, "if").is_some_and(|_| !std::ptr::eq(step, &steps[cleanup_index]))
        {
            return Err("prepared managed phases must not mask errors or skip admission".into());
        }
    }
    if field(prepare, "timeout-minutes") != Some("20")
        || field(prepare, "run").map(lines) != Some(lines(PREPARE))
        || prepare.get("env").and_then(|env| field(env, "SOURCE_SHA"))
            != Some("${{ inputs.source_sha || github.sha }}")
        || field(admit, "run").map(lines) != Some(lines(ADMIT))
    {
        return Err(
            "prepared controller requires exact selected-source, private-copy and digest custody"
                .into(),
        );
    }
    let work = &steps[work_index];
    let cleanup = field(&steps[cleanup_index], "run").ok_or("cleanup run missing")?;
    let valid = if field(work, "name") == Some("Check trusted runner package") {
        field(job, "timeout-minutes") == Some("45")
            && field(work, "timeout-minutes") == Some("15")
            && field(work, "run") == Some("cargo check --locked -p mesh-llm-config")
            && cleanup
                .trim_end()
                .ends_with("ci-ops runner-cleanup --job runner-contract --evidence-uploaded false")
    } else {
        field(work, "uses") == Some("./.github/actions/run-laya-product-smoke")
            && field(work, "timeout-minutes") == Some("70")
            && field(job, "timeout-minutes")
                == Some("${{ inputs.timeout_minutes > 100 && inputs.timeout_minutes || 100 }}")
            && cleanup
                .trim_end()
                .ends_with("ci-ops runner-cleanup --job smoke --evidence-uploaded false")
    };
    if !valid || !invocation(cleanup) {
        return Err(
            "prepared cleanup must follow its finite managed owner and phase budgets".into(),
        );
    }
    Ok(())
}

const PREPARE: &str = r#"set -euo pipefail
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 1
test "$(GIT_MASTER=1 git rev-parse HEAD)" = "$SOURCE_SHA"
report="$(just automation-bootstrap)"
binary="$(printf '%s\n' "$report" | awk -F= '$1 == "binary_path" { count++; print substr($0, index($0, "=") + 1) } END { if (count != 1) exit 1 }')"
[[ "$binary" == /* && -f "$binary" && ! -L "$binary" && -x "$binary" ]] || exit 1
sha="$(sha256sum "$binary" | awk '{print $1}')"
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || exit 1
controller_dir="$(mktemp -d "$RUNNER_TEMP/native-cleanup.XXXXXX")"
cp "$binary" "$controller_dir/xtask"
chmod 500 "$controller_dir/xtask"
test "$(sha256sum "$controller_dir/xtask" | awk '{print $1}')" = "$sha"
binary="$controller_dir/xtask"
{
  printf 'MESH_LLM_AUTOMATION_BIN=%s\n' "$binary"
  printf 'MESH_NATIVE_CLEANUP_SHA=%s\n' "$sha"
  printf 'MESH_NATIVE_CLEANUP_SOURCE=%s\n' "$SOURCE_SHA"
} >> "$GITHUB_ENV"
"#;
const ADMIT: &str = r#"set -euo pipefail
test "$(GIT_MASTER=1 git rev-parse HEAD)" = "$MESH_NATIVE_CLEANUP_SOURCE"
test "$(sha256sum "$MESH_LLM_AUTOMATION_BIN" | awk '{print $1}')" = "$MESH_NATIVE_CLEANUP_SHA"
printf 'MESH_NATIVE_MANAGED_WORK_STARTED=true\n' >> "$GITHUB_ENV"
"#;
const CLEANUP_PREFIX: &str = r#"set -euo pipefail
if [[ "${MESH_NATIVE_MANAGED_WORK_STARTED:-}" != true ]]; then
  echo 'native cleanup refused: managed work never started; no owned workload cleanup attempted' >&2
  exit 1
fi
test "$(GIT_MASTER=1 git rev-parse HEAD)" = "$MESH_NATIVE_CLEANUP_SOURCE"
test "$(sha256sum "$MESH_LLM_AUTOMATION_BIN" | awk '{print $1}')" = "$MESH_NATIVE_CLEANUP_SHA"
"#;

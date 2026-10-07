//! Closed SDK adapter expansion shared by required graph and closure traversal.
use crate::command::DynResult;
use std::{fs, path::Path};
const OWNER: &str = "tools/xtask/src/automation/smoke_observation/sdk_supervision/child.rs";
pub(super) fn target(
    root: &Path,
    path: &str,
    text: &str,
    line: usize,
    block: &str,
) -> DynResult<Option<&'static str>> {
    let direct = block.contains("automation smoke-observation sdk-client");
    let continuation = block.starts_with("--client ")
        && line > 1
        && text.lines().nth(line - 2).is_some_and(|previous| {
            previous.trim()
                == "\"${workload_automation[@]}\" automation smoke-observation sdk-client \\"
        });
    if !direct && !continuation {
        return Ok(None);
    }
    // The multiline opener is handled at its client-bearing continuation.
    if direct && block.ends_with('\\') && !block.contains("--client ") {
        return Ok(None);
    }
    let prefix = match path {
        "scripts/ci-compat-smoke.sh" => {
            "\"${automation[@]}\" automation smoke-observation sdk-client --client "
        }
        "scripts/skippy-workload-certify.sh" | "skippy/scripts/skippy-workload-certify.sh"
            if continuation =>
        {
            "--client "
        }
        _ => return Err("required SDK graph refuses unknown adapter caller".into()),
    };
    let rest = block
        .strip_prefix(prefix)
        .ok_or("required SDK graph adapter syntax changed")?;
    let (client, tail) = rest
        .split_once(' ')
        .ok_or("required SDK graph missing client options")?;
    if !tail.starts_with("--python \"$SDK_PYTHON\"") || tail.contains("--client ") {
        return Err("required SDK graph interpreter/client selection changed".into());
    }
    let (child, lock) = match (path, client) {
        ("scripts/ci-compat-smoke.sh", "openai") => (
            "scripts/ci-openai-python-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        ("scripts/ci-compat-smoke.sh", "litellm") => (
            "scripts/ci-litellm-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        ("scripts/ci-compat-smoke.sh", "langchain") => (
            "scripts/ci-langchain-openai-smoke.py",
            "ci/required-sdk-python/requirements.lock",
        ),
        (
            "scripts/skippy-workload-certify.sh" | "skippy/scripts/skippy-workload-certify.sh",
            "embeddings",
        ) => (
            "scripts/ci-openai-embeddings-smoke.py",
            "ci/canary-python/uv.lock",
        ),
        _ => return Err("required SDK graph unknown client or caller class".into()),
    };
    let source = fs::read_to_string(root.join(OWNER))?;
    let selected = source
        .split_once(&format!("\"{client}\" => ("))
        .and_then(|(_, tail)| tail.split_once("),").map(|(branch, _)| branch))
        .ok_or("required SDK graph native roster missing")?;
    let file = child.strip_prefix("scripts/").ok_or("SDK child path")?;
    if !selected.contains(&format!("\"{file}\""))
        || !selected.contains(&format!("\"{lock}\""))
        || !source.contains("OsString::from(\"-I\")")
        || !source.contains("executable: python.to_path_buf()")
    {
        return Err("required SDK graph native launch/roster changed".into());
    }
    Ok(Some(child))
}
#[cfg(test)]
#[path = "sdk_calls_tests.rs"]
mod tests;

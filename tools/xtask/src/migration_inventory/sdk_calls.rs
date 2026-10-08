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
    external(root, child)?;
    Ok(Some(child))
}
// Static source admission only; runtime admission separately checks the actual
// external checkout. The old leaf name is a closed selector, not a local file.
pub(super) fn external(root: &Path, child: &str) -> DynResult<String> {
    if !matches!(
        child,
        "scripts/ci-openai-python-smoke.py"
            | "scripts/ci-litellm-smoke.py"
            | "scripts/ci-langchain-openai-smoke.py"
            | "scripts/ci-openai-embeddings-smoke.py"
    ) {
        return Err("required SDK graph external leaf refused".into());
    }
    const PIN: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../ci/required-sdk-python/sdk-source.json"
    ));
    for (path, compiled) in [
        ("ci/required-sdk-python/sdk-source.json", PIN),
        (
            "tools/xtask/src/automation/python_sdk_source.rs",
            include_bytes!("../automation/python_sdk_source.rs").as_slice(),
        ),
        (
            "tools/xtask/src/automation/smoke_observation/sdk_supervision.rs",
            include_bytes!("../automation/smoke_observation/sdk_supervision.rs").as_slice(),
        ),
        (
            OWNER,
            include_bytes!("../automation/smoke_observation/sdk_supervision/child.rs").as_slice(),
        ),
    ] {
        if fs::read(root.join(path))? != compiled {
            return Err("required SDK graph source admission differs from compiled owner".into());
        }
    }
    let pin: serde_json::Value = serde_json::from_slice(PIN)?;
    let repository = pin["repository"].as_str().ok_or("SDK repository pin")?;
    let revision = pin["revision"].as_str().ok_or("SDK revision pin")?;
    let digest = pin["manifest_sha256"].as_str().ok_or("SDK manifest pin")?;
    if repository != "Mesh-LLM/mesh-llm-python-sdk"
        || revision.len() != 40
        || digest.len() != 64
        || !revision
            .bytes()
            .chain(digest.bytes())
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("required SDK graph external identity refused".into());
    }
    Ok(format!(
        "{repository}@{revision}:{child}#manifest-sha256={digest}"
    ))
}
#[cfg(test)]
#[path = "sdk_calls_tests.rs"]
mod tests;

//! Proven roots only: the BASH_SOURCE self/parent directory idiom, a later
//! `NAME="$ROOT/rel.py"` in the same file, or an action's `$GITHUB_ACTION_PATH`.

use super::{Edge, GraphBuilder, contexts, inline_context, inline_launch};
use crate::command::DynResult;
use std::collections::BTreeMap;
use std::path::Path;

const SELF_DIR: &str = "\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"";
const PARENT_DIR: &str = "\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"";

fn normalize(path: &Path) -> String {
    let mut parts = Vec::new();
    for part in path.to_string_lossy().split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_owned()),
        }
    }
    parts.join("/")
}

fn bindings(path: &str, text: &str) -> BTreeMap<String, String> {
    let directory = Path::new(path).parent().unwrap_or_else(|| Path::new(""));
    let mut result = BTreeMap::new();
    if path.ends_with("/action.yml") {
        result.insert("GITHUB_ACTION_PATH".to_owned(), normalize(directory));
    }
    for line in text.lines() {
        let Some((name, value)) = line.trim().split_once('=') else {
            continue;
        };
        if name.is_empty()
            || !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            continue;
        }
        let resolved = match value {
            SELF_DIR => Some(normalize(directory)),
            PARENT_DIR => Some(normalize(&directory.join(".."))),
            _ => expand(value.trim_matches('"'), &result),
        };
        if let Some(resolved) = resolved {
            result.insert(name.to_owned(), resolved);
        }
    }
    result
}

fn expand(value: &str, bindings: &BTreeMap<String, String>) -> Option<String> {
    let rest = value.strip_prefix('$')?;
    let (name, tail) = rest.split_once('/').unwrap_or((rest, ""));
    let name = name.trim_start_matches('{').trim_end_matches('}');
    let root = bindings.get(name)?;
    let joined = normalize(&Path::new(root).join(tail));
    (!tail.contains(['$', '`', '*', ' ']) && Path::new(&joined).extension().is_some())
        .then_some(joined)
}

fn target(block: &str, bindings: &BTreeMap<String, String>) -> Option<String> {
    let words = block
        .split_whitespace()
        .map(|word| word.trim_matches(['"', '\'', ';', '(', ')']));
    words
        .clone()
        .filter(|word| word.starts_with('$'))
        .find_map(|word| expand(word, bindings).filter(|path| path.ends_with(".py")))
        .or_else(|| {
            words
                .filter(|word| word.starts_with("scripts\\") && word.ends_with(".py"))
                .find(|word| !word.contains(['$', '*', '/']))
                .map(|word| word.replace('\\', "/"))
        })
}

pub(super) fn record(
    builder: &mut GraphBuilder<'_>,
    path: &str,
    text: &str,
    (line, block): (usize, &str),
    trust: &'static str,
) -> DynResult<bool> {
    let candidate = builder
        .observed
        .iter()
        .find(|row| row.path == path && row.source_block == block);
    if candidate.is_none() || super::sources::tokens(block).next().is_some() {
        return Ok(false);
    }
    let variable = inline_launch::variable_script_target(block);
    if !variable && !(block.contains(".py") && candidate.is_some_and(|row| row.executable)) {
        return Ok(false);
    }
    let Some(child) = target(block, &bindings(path, text)) else {
        inline_context::record_unbound_script(builder, path, line, block, trust);
        return Ok(!variable);
    };
    if !builder.known.contains(child.as_str()) {
        return Err(format!("required graph: missing child {child} from {path}:{line}").into());
    }
    let optional = block.starts_with("if ") || block.contains("|| true");
    let contract = (trust == "same_commit")
        .then(|| {
            builder.contracts.for_call(
                path,
                line,
                block,
                &child,
                builder.observed,
                builder.validated,
            )
        })
        .flatten();
    let (status, reason) = match (optional, &contract) {
        (true, _) => (
            "optional_branch",
            Some("conditional command; runtime selection not proven".to_owned()),
        ),
        (false, Some(_)) => ("reached_execution", None),
        (false, None) if contexts::selected_interpreter_call(block) => (
            "unknown_selection",
            Some(
                "selected interpreter binding and invocation contract are not source-proven"
                    .to_owned(),
            ),
        ),
        (false, None) => (
            "reached_execution",
            Some(format!(
                "missing source-backed interpreter contract for {path}:{line} -> {child}"
            )),
        ),
    };
    builder.edges.push(Edge {
        parent: path.to_owned(),
        line,
        source_block: block.to_owned(),
        trust_revision: trust,
        status,
        child: Some(child.clone()),
        unresolved_reason: reason,
        argv: contract.as_ref().map(|row| row.argv.clone()),
        status_streams_effects: contract.as_ref().map(|row| row.effects.clone()),
        contract_source: contract.map(|row| row.source),
    });
    if status == "reached_execution" && trust == "same_commit" {
        builder.queue.push_back((child, trust));
    }
    Ok(true)
}

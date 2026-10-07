//! Closed root shell forwarding into an explicitly named current product owner.
use super::{Edge, GraphBuilder};
use crate::command::DynResult;
const START: &str = "exec \"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")";
const PREFIX: &str = "exec \"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)/";
const BASH_START: &str = "exec bash \"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")";
const BASH_PREFIX: &str = "exec bash \"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)/";
fn target(path: &str, block: &str) -> DynResult<Option<String>> {
    if !path.starts_with("scripts/")
        || path["scripts/".len()..].contains('/')
        || !(block.starts_with(START) || block.starts_with(BASH_START))
    {
        return Ok(None);
    }
    let child = block
        .strip_prefix(PREFIX)
        .or_else(|| block.strip_prefix(BASH_PREFIX))
        .and_then(|tail| tail.strip_suffix("\" \"$@\""))
        .ok_or("required graph: changed product forwarder root or argument custody")?;
    if !(child.starts_with("mesh/scripts/") || child.starts_with("skippy/scripts/"))
        || !child.ends_with(".sh")
        || child.split('/').any(|part| {
            part.is_empty()
                || matches!(part, "." | "..")
                || !part
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        })
    {
        return Err(
            "required graph: product forwarder requires a literal safe product script path".into(),
        );
    }
    Ok(Some(child.into()))
}
pub(super) fn record(
    builder: &mut GraphBuilder<'_>,
    path: &str,
    line: usize,
    block: &str,
    trust: &'static str,
) -> DynResult<bool> {
    let Some(child) = target(path, block)? else {
        return Ok(false);
    };
    if !builder.known.contains(child.as_str()) {
        return Err(
            format!("required graph: missing product owner {child} from {path}:{line}").into(),
        );
    }
    let same = trust == "same_commit";
    builder.edges.push(Edge {
        parent: path.into(),
        line,
        source_block: block.into(),
        trust_revision: trust,
        status: if same {
            "reached_execution"
        } else {
            "unknown_selection"
        },
        child: Some(child.clone()),
        unresolved_reason: (!same)
            .then(|| "protected product owner bytes are external to this checkout".into()),
        argv: Some(if block.starts_with(BASH_PREFIX) {
            format!("bash {child} $@")
        } else {
            format!("{child} $@")
        }),
        status_streams_effects: None,
        contract_source: None,
    });
    if same {
        builder.queue.push_back((child, trust));
    }
    Ok(true)
}
#[cfg(test)]
#[path = "product_forwarders_tests.rs"]
mod tests;

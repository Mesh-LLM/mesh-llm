use super::{Edge, GraphBuilder};
use crate::command::DynResult;
use std::path::Path;
pub(super) fn record(
    root: &Path,
    builder: &mut GraphBuilder<'_>,
    path: &str,
    text: &str,
    line: usize,
    block: &str,
    trust: &'static str,
) -> DynResult<bool> {
    let Some(child) = super::super::sdk_calls::target(root, path, text, line, block)? else {
        return Ok(false);
    };
    if !builder.known.contains(child) {
        return Err(format!("required graph: missing SDK child {child}").into());
    }
    let contract = (trust == "same_commit")
        .then(|| {
            builder.contracts.for_call(
                path,
                line,
                block,
                child,
                builder.observed,
                builder.validated,
            )
        })
        .flatten();
    let proven = contract.is_some();
    builder.edges.push(Edge {
        parent: path.into(),
        line,
        source_block: block.into(),
        trust_revision: trust,
        status: if proven {
            "reached_execution"
        } else {
            "unknown_selection"
        },
        child: Some(child.into()),
        unresolved_reason: (!proven).then(|| {
            "SDK adapter launch requires its exact source-backed interpreter contract".into()
        }),
        argv: contract.as_ref().map(|c| c.argv.clone()),
        status_streams_effects: contract.as_ref().map(|c| c.effects.clone()),
        contract_source: contract.map(|c| c.source),
    });
    if proven {
        builder.queue.push_back((child.into(), trust));
    }
    Ok(true)
}

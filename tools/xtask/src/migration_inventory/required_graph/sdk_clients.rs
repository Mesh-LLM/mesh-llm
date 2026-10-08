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
    let external = super::super::sdk_calls::external(root, child)?;
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
            "external_sdk_boundary"
        } else {
            "unknown_selection"
        },
        child: Some(external),
        unresolved_reason: (!proven).then(|| {
            "SDK adapter launch requires its exact source-backed interpreter contract".into()
        }),
        argv: contract.as_ref().map(|c| c.argv.clone()),
        status_streams_effects: contract.as_ref().map(|c| c.effects.clone()),
        contract_source: contract.map(|c| c.source),
    });

    Ok(true)
}

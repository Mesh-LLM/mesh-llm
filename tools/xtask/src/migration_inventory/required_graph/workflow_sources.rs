use super::{
    Edge, GraphBuilder,
    sources::{local_target, protected_target, selection},
};
use crate::command::DynResult;

pub(super) fn record(
    builder: &mut GraphBuilder<'_>,
    path: &str,
    line: usize,
    block: &str,
    trust: &'static str,
) -> DynResult<bool> {
    if protected_target(block) {
        builder.edges.push(Edge {
            parent: path.to_owned(),
            line,
            source_block: block.to_owned(),
            trust_revision: "protected_main_external",
            status: if block.ends_with("@main") {
                "selected"
            } else {
                "unknown_selection"
            },
            child: None,
            unresolved_reason: Some(
                if block.ends_with("@main") {
                    "protected @main workflow bytes are external to this checkout"
                } else {
                    "remote workflow revision is not protected @main"
                }
                .to_owned(),
            ),
            argv: None,
            status_streams_effects: None,
            contract_source: None,
        });
        return Ok(true);
    }
    if let Some(child) = local_target(block) {
        if !builder.known.contains(child.as_str()) {
            return Err(format!("required graph: missing child {child} from {path}:{line}").into());
        }
        builder.edges.push(Edge {
            parent: path.to_owned(),
            line,
            source_block: block.to_owned(),
            trust_revision: trust,
            status: if trust == "same_commit" {
                "reached_execution"
            } else {
                "unknown_selection"
            },
            child: Some(child.clone()),
            unresolved_reason: if trust == "same_commit" {
                None
            } else {
                Some("protected @main workflow bytes are external to this checkout".to_owned())
            },
            argv: None,
            status_streams_effects: None,
            contract_source: None,
        });
        if trust == "same_commit" {
            builder.queue.push_back((child, trust));
        }
        return Ok(true);
    }
    if let Some(value) = selection(block) {
        if value.starts_with("actions/setup-python@") {
            builder.edges.push(Edge {
                parent: path.to_owned(),
                line,
                source_block: block.to_owned(),
                trust_revision: trust,
                status: "provisioning_selection",
                child: None,
                unresolved_reason: None,
                argv: None,
                status_streams_effects: None,
                contract_source: None,
            });
        } else if value.starts_with("./") || value.contains("${{") {
            builder.edges.push(Edge {
                parent: path.to_owned(),
                line,
                source_block: block.to_owned(),
                trust_revision: trust,
                status: "unknown_selection",
                child: None,
                unresolved_reason: Some(format!("unresolved workflow/action target {value}")),
                argv: None,
                status_streams_effects: None,
                contract_source: None,
            });
        }
        return Ok(true);
    }
    Ok(false)
}

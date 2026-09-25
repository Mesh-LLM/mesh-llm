use super::contexts::{self, Context};
use super::{Edge, GraphBuilder};
use crate::command::DynResult;

impl GraphBuilder<'_> {
    pub(super) fn push_edge(
        &mut self,
        parent: &str,
        line: usize,
        block: &str,
        child: &str,
        trust: &'static str,
    ) -> DynResult<()> {
        let context = contexts::classify(block, child);
        let contract = if trust == "same_commit" {
            self.contracts
                .for_call(parent, line, block, child, self.observed, self.validated)
        } else {
            None
        };
        let selected_interpreter = contexts::selected_interpreter_call(block);
        let reached = matches!(context, Context::Launch | Context::Optional);
        let python = child.ends_with(".py") && reached;
        let (status, reason) = if trust == "protected_main_external" {
            (
                "unknown_selection",
                Some("protected @main child bytes are external to this checkout".to_owned()),
            )
        } else if selected_interpreter && contract.is_none() {
            (
                "unknown_selection",
                Some(
                    "selected interpreter binding and invocation contract are not source-proven"
                        .to_owned(),
                ),
            )
        } else {
            match context {
                Context::Launch => (
                    "reached_execution",
                    if python && contract.is_none() {
                        Some(format!(
                            "missing source-backed interpreter contract for {parent}:{line} -> {child}"
                        ))
                    } else {
                        None
                    },
                ),
                Context::Optional => (
                    "optional_branch",
                    Some("conditional command; runtime selection not proven".to_owned()),
                ),
                Context::Reference => ("reference_only", None),
                Context::Provision => ("provisioning_selection", None),
                Context::Unknown => (
                    "unknown_selection",
                    Some("source syntax or selection does not prove a launch".to_owned()),
                ),
            }
        };
        if !self.known.contains(child) && reached {
            return Err(
                format!("required graph: missing child {child} from {parent}:{line}").into(),
            );
        }
        self.edges.push(Edge {
            parent: parent.to_owned(),
            line,
            source_block: block.to_owned(),
            trust_revision: trust,
            status,
            child: Some(child.to_owned()),
            unresolved_reason: reason,
            argv: contract.as_ref().map(|row| row.argv.clone()),
            status_streams_effects: contract.as_ref().map(|row| row.effects.clone()),
            contract_source: contract.map(|row| row.source),
        });
        if reached && trust == "same_commit" && self.known.contains(child) {
            self.queue.push_back((child.to_owned(), trust));
        }
        Ok(())
    }
}

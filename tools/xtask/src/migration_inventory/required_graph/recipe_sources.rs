use super::{
    Edge, GraphBuilder,
    sources::{source_lines, tokens},
};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub(super) fn expand(
    root: &Path,
    builder: &mut GraphBuilder<'_>,
    visited: &mut BTreeSet<(String, &'static str)>,
) -> DynResult<()> {
    while let Some((path, trust)) = builder.queue.pop_front() {
        if !builder.known.contains(path.as_str()) || !root.join(&path).is_file() {
            return Err(format!("required graph: missing source {path}").into());
        }
        if !visited.insert((path.clone(), trust)) {
            continue;
        }
        let text = fs::read_to_string(root.join(&path))?;
        if path.ends_with(".py") && trust == "same_commit" {
            super::python_children::scan(builder, &path, &text)?;
            continue;
        }
        for (line, block) in source_lines(&path, &text) {
            if path.ends_with(".py") {
                if block.contains("spec_from_file_location(") || block.contains("subprocess.") {
                    builder.edges.push(Edge {
                        parent: path.clone(),
                        line,
                        source_block: block,
                        trust_revision: trust,
                        status: "unknown_selection",
                        child: None,
                        unresolved_reason: Some(
                            "Python child selection needs callable and argument proof".to_owned(),
                        ),
                        argv: None,
                        status_streams_effects: None,
                        contract_source: None,
                    });
                }
                continue;
            }
            for child in tokens(&block) {
                if child != path {
                    builder.push_edge(&path, line, &block, child, trust)?;
                }
            }
        }
    }
    Ok(())
}

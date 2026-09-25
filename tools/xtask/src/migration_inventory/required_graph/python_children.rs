use super::python_targets::{path_target, root_bindings, subprocess_target};
use super::{Edge, GraphBuilder};
use crate::command::DynResult;
use std::collections::BTreeMap;

struct ImportSpec {
    target: Option<String>,
    line: usize,
    source: String,
    local: bool,
}

fn call_lines(text: &str) -> Vec<(usize, String)> {
    let mut result = Vec::new();
    let mut pending: Option<(usize, String, usize)> = None;
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if let Some((start, mut expression, depth)) = pending.take() {
            expression.push(' ');
            expression.push_str(line);
            let depth = depth
                .saturating_add(line.matches('(').count())
                .saturating_sub(line.matches(')').count());
            if depth == 0 {
                result.push((start, expression));
            } else {
                pending = Some((start, expression, depth));
            }
        } else if (line.contains("subprocess.") || line.contains("spec_from_file_location("))
            && !line.starts_with('#')
            && !line.starts_with(['\'', '"'])
            && !line.contains(" = '")
            && !line.contains(" = \"")
        {
            let depth = line
                .matches('(')
                .count()
                .saturating_sub(line.matches(')').count());
            if depth == 0 {
                result.push((index + 1, line.to_owned()));
            } else {
                pending = Some((index + 1, line.to_owned(), depth));
            }
        } else if line.contains(".loader.exec_module(") && !line.starts_with('#') {
            result.push((index + 1, line.to_owned()));
        }
    }
    result
}

fn record(
    builder: &mut GraphBuilder<'_>,
    parent: &str,
    line: usize,
    source: &str,
    target: Option<String>,
    local: bool,
    kind: &str,
) -> DynResult<()> {
    if local
        && let Some(child) = target.as_deref()
        && !builder.known.contains(child)
    {
        return Err(format!("required graph: missing child {child} from {parent}:{line}").into());
    }
    let reached = local && target.is_some();
    builder.edges.push(Edge {
        parent: parent.to_owned(),
        line,
        source_block: source.to_owned(),
        trust_revision: "same_commit",
        status: if reached { "reached_execution" } else { "unknown_selection" },
        child: target.clone(),
        unresolved_reason: Some(if reached {
            format!("Python {kind} descendant has no verified per-call argv/status/stream/effects contract")
        } else {
            format!("Python {kind} target bytes or execution are runtime-selected")
        }),
        argv: None,
        status_streams_effects: None,
        contract_source: None,
    });
    if reached && let Some(child) = target {
        builder.queue.push_back((child, "same_commit"));
    }
    Ok(())
}

pub(super) fn scan(builder: &mut GraphBuilder<'_>, parent: &str, text: &str) -> DynResult<()> {
    let mut specs = BTreeMap::<String, ImportSpec>::new();
    let bindings = root_bindings(text);
    for (index, line) in call_lines(text) {
        let source = text.lines().nth(index - 1).unwrap_or("").trim();
        if let Some((name, expression)) =
            line.split_once(" = importlib.util.spec_from_file_location(")
        {
            let (target, local) = path_target(parent, expression);
            specs.insert(
                name.trim().to_owned(),
                ImportSpec {
                    target,
                    line: index,
                    source: source.to_owned(),
                    local,
                },
            );
            continue;
        }
        if let Some((name, _)) = line.split_once(".loader.exec_module(") {
            if let Some(spec) = specs.remove(name.trim()) {
                record(
                    builder,
                    parent,
                    spec.line,
                    &spec.source,
                    spec.target,
                    spec.local,
                    "import",
                )?;
            } else {
                record(builder, parent, index, source, None, false, "import")?;
            }
            continue;
        }
        if line.contains("subprocess.run(")
            || line.contains("subprocess.Popen(")
            || line.contains("subprocess.check_call(")
            || line.contains("subprocess.check_output(")
        {
            let (target, local) = subprocess_target(parent, &line, &bindings);
            record(builder, parent, index, source, target, local, "subprocess")?;
        }
    }
    for spec in specs.into_values() {
        record(
            builder,
            parent,
            spec.line,
            &spec.source,
            spec.target,
            false,
            "import specification",
        )?;
    }
    Ok(())
}

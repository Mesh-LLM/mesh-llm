use super::{Edge, GraphBuilder};

pub(super) fn record(
    builder: &mut GraphBuilder<'_>,
    path: &str,
    line: usize,
    block: &str,
    trust: &'static str,
) -> bool {
    if shell_nonexecution(path, block) {
        return true;
    }
    if let Some(reason) = interpreter_selection(path, block) {
        builder.edges.push(Edge {
            parent: path.to_owned(),
            line,
            source_block: block.to_owned(),
            trust_revision: trust,
            status: "unknown_selection",
            child: None,
            unresolved_reason: Some(reason.to_owned()),
            argv: None,
            status_streams_effects: None,
            contract_source: None,
        });
        return true;
    }
    let candidate = builder.observed.iter().find(|row| {
        row.path == path
            && row.source_block == block
            && row.executable
            && !block.contains(".py")
            && !super::inline_launch::variable_script_target(block)
    });
    if block.contains("spec_from_file_location(")
        || block.contains("subprocess.run(")
        || block.contains("subprocess.Popen(")
        || block.contains("subprocess.check_call(")
        || candidate.is_some()
    {
        let contract = if trust == "same_commit" {
            builder
                .contracts
                .inline_call(path, line, block, builder.observed, builder.validated)
        } else {
            None
        };
        builder.edges.push(Edge {
            parent: path.to_owned(),
            line,
            source_block: block.to_owned(),
            trust_revision: trust,
            status: if contract.is_some() {
                "reached_execution"
            } else {
                "unknown_selection"
            },
            child: None,
            unresolved_reason: contract
                .is_none()
                .then(|| super::inline_launch::unresolved_reason(block).to_owned()),
            argv: contract.as_ref().map(|row| row.argv.clone()),
            status_streams_effects: contract.as_ref().map(|row| row.effects.clone()),
            contract_source: contract.map(|row| row.source),
        });
    }
    false
}

pub(super) fn record_unbound_script(
    builder: &mut GraphBuilder<'_>,
    path: &str,
    line: usize,
    block: &str,
    trust: &'static str,
) {
    builder.edges.push(Edge {
        parent: path.to_owned(),
        line,
        source_block: block.to_owned(),
        trust_revision: trust,
        status: "unknown_selection",
        child: None,
        unresolved_reason: Some(super::inline_launch::unresolved_reason(block).to_owned()),
        argv: None,
        status_streams_effects: None,
        contract_source: None,
    });
}

fn shell_source(path: &str) -> bool {
    path.ends_with(".sh")
}

fn assigned_name(block: &str) -> Option<&str> {
    let (name, _) = block.split_once('=')?;
    (!name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_'))
    .then_some(name)
}

pub(super) fn shell_nonexecution(path: &str, block: &str) -> bool {
    if !shell_source(path) {
        return false;
    }
    let block = block.trim_start();
    let function = block.strip_suffix("() {").is_some_and(|name| {
        !name.is_empty()
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    });
    function
        || block.starts_with("local ")
            && block["local ".len()..].split_whitespace().all(|word| {
                word.chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            })
        || assigned_name(block).is_some_and(|_| !block.contains("$("))
}

pub(super) fn interpreter_selection(path: &str, block: &str) -> Option<&'static str> {
    let (_, value) = block.split_once('=')?;
    let value = value.trim().strip_suffix(" || {").unwrap_or(value.trim());
    if shell_source(path)
        && assigned_name(block).is_some()
        && matches!(
            value,
            "\"$(python_bin)\""
                | "$(python_bin)"
                | "\"$(command -v python3)\""
                | "\"$(command -v python)\""
        )
    {
        Some(
            "shell assignment selects an interpreter; its binding and later launch remain unproven",
        )
    } else {
        None
    }
}

//! Authored YAML handoffs and bounded command argument bindings.
use super::{Node, field};
use crate::command::DynResult;

pub(super) fn member<'a>(node: &'a Node, key: &str) -> DynResult<&'a Node> {
    node.get(key)
        .ok_or_else(|| format!("missing workflow field {key}").into())
}
pub(super) fn job<'a>(document: &'a Node, name: &str) -> DynResult<&'a Node> {
    member(member(document, "jobs")?, name)
}
pub(super) fn steps(node: &Node) -> DynResult<&[Node]> {
    match member(node, "steps")? {
        Node::Seq(steps) => Ok(steps),
        _ => Err("workflow steps must be a sequence".into()),
    }
}
pub(super) fn step<'a>(steps: &'a [Node], key: &str, value: &str) -> DynResult<(usize, &'a Node)> {
    steps
        .iter()
        .enumerate()
        .find(|(_, step)| field(step, key) == Some(value))
        .ok_or_else(|| format!("missing workflow step {key}={value}").into())
}
pub(super) fn binding(node: &Node, key: &str, expected: &str) -> DynResult<()> {
    if field(node, key) == Some(expected) {
        Ok(())
    } else {
        Err(format!("workflow {key} must bind {expected}").into())
    }
}
pub(super) fn needs(node: &Node, expected: &[&str]) -> DynResult<()> {
    let actual = member(node, "needs")?.list();
    if expected.iter().all(|name| actual.contains(name)) {
        Ok(())
    } else {
        Err(format!("workflow dependencies must include {expected:?}").into())
    }
}
pub(super) fn condition(node: &Node, expected: &str) -> DynResult<()> {
    let compact = |text: &str| {
        text.chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
    };
    let actual = compact(field(node, "if").ok_or("missing workflow admission condition")?);
    let expected = compact(expected);
    if actual.trim_start_matches("${{").trim_end_matches("}}")
        == expected.trim_start_matches("${{").trim_end_matches("}}")
    {
        Ok(())
    } else {
        Err(format!("workflow admission must be {expected}").into())
    }
}
pub(super) fn command(node: &Node, prefix: &[&str], pairs: &[(&str, &str)]) -> DynResult<()> {
    let source = field(node, "run").ok_or("missing workflow command")?;
    let lines = source
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\\\n", " ");
    for line in lines.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.starts_with(prefix)
            && pairs
                .iter()
                .all(|(flag, value)| words.windows(2).any(|pair| pair == [*flag, *value]))
        {
            return Ok(());
        }
    }
    Err(format!("workflow command must invoke {prefix:?} with bound arguments {pairs:?}").into())
}
pub(super) fn checkout(steps: &[Node], source: &str, path: Option<&str>) -> DynResult<usize> {
    for (index, node) in steps.iter().enumerate() {
        if !field(node, "uses").is_some_and(|value| value.starts_with("actions/checkout@")) {
            continue;
        }
        let inputs = member(node, "with")?;
        if field(inputs, "ref") == Some(source) && field(inputs, "path") == path {
            binding(inputs, "persist-credentials", "false")?;
            return Ok(index);
        }
    }
    Err(format!("missing credential-free checkout of {source} at {path:?}").into())
}
pub(super) fn before(first: usize, second: usize) -> DynResult<()> {
    if first < second {
        Ok(())
    } else {
        Err("workflow handoff precedes its producer".into())
    }
}

#[cfg(test)]
pub(super) fn mutable<'a>(node: &'a mut Node, key: &str) -> &'a mut Node {
    let Node::Map(entries) = node else {
        panic!("mapping mutation required");
    };
    &mut entries
        .iter_mut()
        .find(|(name, _)| name == key)
        .expect("mutation field exists")
        .1
}
#[cfg(test)]
pub(super) fn replace(node: &mut Node, path: &[&str], value: Node) {
    if path.is_empty() {
        *node = value;
        return;
    }
    let Node::Map(entries) = node else {
        panic!("mapping mutation required");
    };
    let member = entries
        .iter_mut()
        .find(|(key, _)| key == path[0])
        .expect("mutation field exists");
    replace(&mut member.1, &path[1..], value);
}

use super::super::just_recipes::{Assignment, Dependency, Justfile, Parameter, Recipe};
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) struct Invocation {
    pub(super) recipe: String,
    pub(super) arguments: Vec<String>,
}

pub(super) struct RenderedLine {
    pub(super) source: String,
    pub(super) result: Result<String, String>,
}

pub(super) fn call(line: &str) -> Result<Option<Invocation>, String> {
    let all = line
        .trim_start_matches(['@', '-'])
        .split_whitespace()
        .collect::<Vec<_>>();
    let words = &all[all.iter().take_while(|word| env_assignment(word)).count()..];
    let Some(index) = words.iter().position(|word| *word == "just") else {
        return Ok(None);
    };
    if index > 1 || (index == 1 && words[0] != "run:") {
        return Ok(None);
    }
    let recipe = words.get(index + 1).ok_or("missing Just recipe target")?;
    if !literal(recipe) {
        return Err("dynamic Just recipe target".to_owned());
    }
    let arguments = match words[index + 2..]
        .iter()
        .position(|word| matches!(*word, ">" | ">>"))
    {
        Some(redirect) => match &words[index + 2 + redirect + 1..] {
            [target] if literal(target) => &words[index + 2..index + 2 + redirect],
            _ => return Err(format!("dynamic Just arguments for {recipe}")),
        },
        None => &words[index + 2..],
    };
    if arguments.iter().any(|word| !literal(word)) {
        return Err(format!("dynamic Just arguments for {recipe}"));
    }
    let arguments = arguments.iter().map(|word| (*word).to_owned()).collect();
    Ok(Some(Invocation {
        recipe: (*recipe).to_owned(),
        arguments,
    }))
}

fn env_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, value)| {
        !name.is_empty()
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            && !value.contains("$(")
            && !value.contains('`')
    })
}

fn literal(value: &str) -> bool {
    !value.is_empty()
        && value.chars().all(|ch| {
            ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.' | ':' | '=' | '+')
        })
}

fn bounded_command(value: &str) -> bool {
    !value.is_empty()
        && !value.contains([
            '$', '`', '{', '}', '\'', '"', ';', '|', '&', '<', '>', '\\', '\n',
        ])
}

pub(super) fn dependency(dependency: &Dependency) -> Option<Invocation> {
    Some(Invocation {
        recipe: dependency.recipe.clone(),
        arguments: dependency
            .arguments
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .filter(|text| literal(text))
                    .map(str::to_owned)
            })
            .collect::<Option<Vec<_>>>()?,
    })
}

fn bindings(parameters: &[Parameter], args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut result = BTreeMap::new();
    let mut remaining = args.iter();
    for parameter in parameters {
        let value = match parameter.kind.as_str() {
            "star" | "plus" => remaining.by_ref().cloned().collect::<Vec<_>>().join(" "),
            "singular" => remaining
                .next()
                .cloned()
                .or_else(|| {
                    parameter
                        .default
                        .as_ref()
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .ok_or_else(|| format!("required recipe argument {}", parameter.name))?,
            kind => return Err(format!("unsupported recipe parameter kind {kind}")),
        };
        if !value.is_empty()
            && !(match parameter.kind.as_str() {
                "star" | "plus" => bounded_command(&value),
                "singular" => literal(&value),
                _ => false,
            })
        {
            return Err(format!("dynamic recipe argument {}", parameter.name));
        }
        result.insert(parameter.name.clone(), value);
    }
    if remaining.next().is_some() {
        return Err("extra recipe arguments".to_owned());
    }
    Ok(result)
}

pub(super) fn render(recipe: &Recipe, args: &[String], dump: &Justfile) -> Vec<RenderedLine> {
    let params = bindings(&recipe.parameters, args);
    recipe
        .body
        .iter()
        .map(|line| {
            let source = line
                .iter()
                .map(|part| match part {
                    Value::String(text) => text.clone(),
                    _ => "{{ ... }}".to_owned(),
                })
                .collect::<String>();
            let result = params
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|params| render_line(line, params, dump));
            RenderedLine { source, result }
        })
        .collect()
}

fn render_line(
    line: &[Value],
    params: &BTreeMap<String, String>,
    dump: &Justfile,
) -> Result<String, String> {
    let mut rendered = String::new();
    for part in line {
        match part {
            Value::String(text) => rendered.push_str(text),
            Value::Array(parts) => {
                let Some(Value::Array(expression)) = parts.first() else {
                    return Err("unsupported Just interpolation".to_owned());
                };
                let [Value::String(kind), Value::String(name)] = expression.as_slice() else {
                    return Err("dynamic Just interpolation".to_owned());
                };
                if kind != "variable" {
                    return Err("dynamic Just interpolation".to_owned());
                }
                let value = params
                    .get(name)
                    .map(String::as_str)
                    .or_else(|| dump.assignments.get(name).and_then(assignment_literal))
                    .ok_or_else(|| format!("unresolved Just variable {name}"))?;
                rendered.push_str(value);
            }
            _ => return Err("unsupported Just body".to_owned()),
        }
    }
    Ok(rendered)
}

fn assignment_literal(assignment: &Assignment) -> Option<&str> {
    assignment
        .value
        .as_str()
        .filter(|value| !value.contains(['$', '`', '\n']))
}

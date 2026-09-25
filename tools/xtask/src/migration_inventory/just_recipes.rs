use crate::command::{DynResult, run_command, trimmed_stderr_or_stdout};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::process::Command;

#[derive(Deserialize)]
pub(super) struct Justfile {
    pub(super) assignments: BTreeMap<String, Assignment>,
    pub(super) recipes: BTreeMap<String, Recipe>,
}

#[derive(Deserialize)]
pub(super) struct Assignment {
    pub(super) value: Value,
}

#[derive(Deserialize)]
pub(super) struct Recipe {
    pub(super) body: Vec<Vec<Value>>,
    pub(super) dependencies: Vec<Dependency>,
    #[serde(default)]
    pub(super) attributes: Vec<String>,
    #[serde(default)]
    pub(super) parameters: Vec<Parameter>,
}

#[derive(Deserialize)]
pub(super) struct Parameter {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) default: Option<Value>,
}

#[derive(Deserialize)]
pub(super) struct Dependency {
    pub(super) recipe: String,
    #[serde(default)]
    pub(super) arguments: Vec<Value>,
}

pub(super) fn dump(root: &Path) -> DynResult<Justfile> {
    let mut command = Command::new("just");
    command
        .current_dir(root)
        .args(["--justfile", "Justfile", "--dump", "--dump-format", "json"]);
    let output = run_command(&mut command)?;
    if !output.status.success() {
        return Err(format!(
            "Just recipe: parse failed: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn command_words(line: &str) -> impl Iterator<Item = &str> {
    line.trim_start()
        .trim_start_matches(['@', '-'])
        .split_whitespace()
        .skip_while(|part| part.contains('=') && !part.starts_with('=') && !part.contains('/'))
}

fn recipe_call(line: &str) -> Option<&str> {
    let mut words = command_words(line);
    if words.next()? == "just" {
        words.next()
    } else {
        None
    }
}

fn python_executable(line: &str) -> bool {
    let executable = command_words(line).next().unwrap_or("");
    let name = Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(executable);
    name.starts_with("python") || name.ends_with(".py")
}

fn rendered_line(parts: &[Value], assignments: &BTreeMap<String, Assignment>) -> DynResult<String> {
    let mut output = String::new();
    for part in parts {
        match part {
            Value::String(text) => output.push_str(text),
            Value::Array(interpolation) => {
                let Some(Value::Array(expression)) = interpolation.first() else {
                    return Err("Just recipe: unsupported interpolation shape".into());
                };
                let [Value::String(kind), Value::String(name)] = expression.as_slice() else {
                    return Err("Just recipe: unresolved interpolation expression".into());
                };
                if kind != "variable" {
                    return Err("Just recipe: unresolved interpolation operator".into());
                }
                let Some(assignment) = assignments.get(name) else {
                    return Err(format!("Just recipe: unresolved variable target {name}").into());
                };
                let Some(value) = assignment.value.as_str() else {
                    return Err(format!("Just recipe: unresolved variable target {name}").into());
                };
                output.push_str(value);
            }
            _ => return Err("Just recipe: unsupported body component".into()),
        }
    }
    Ok(output)
}

pub(super) fn check_recipe_children(root: &Path, required: &BTreeSet<String>) -> DynResult<()> {
    if !root.join("Justfile").is_file() {
        return Ok(());
    }
    let mut command = Command::new("just");
    command
        .current_dir(root)
        .args(["--justfile", "Justfile", "--dump", "--dump-format", "json"]);
    let result = run_command(&mut command)?;
    if !result.status.success() {
        return Err(format!(
            "Just recipe: parse failed: {}",
            trimmed_stderr_or_stdout(&result)
        )
        .into());
    }
    let parsed: Justfile = serde_json::from_slice(&result.stdout)?;
    let mut queue = required.iter().cloned().collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let recipe = parsed
            .recipes
            .get(&name)
            .ok_or_else(|| format!("Just recipe: missing required recipe {name}"))?;
        queue.extend(
            recipe
                .dependencies
                .iter()
                .map(|dependency| dependency.recipe.clone()),
        );
        for line in &recipe.body {
            let first = line.first().and_then(Value::as_str).unwrap_or("");
            if !first
                .trim_start()
                .trim_start_matches(['@', '-'])
                .trim_start()
                .starts_with("{{")
                && !first.trim().is_empty()
            {
                if first.trim_start().starts_with('#') {
                    continue;
                }
                let interpolated_target = first.contains("{{")
                    || (first.trim_start().trim_start_matches(['@', '-']).trim() == "just"
                        && line.len() > 1);
                let expanded = if interpolated_target {
                    let expanded = rendered_line(line, &parsed.assignments)?;
                    if python_executable(&expanded) && !python_executable(first) {
                        return Err(format!("Just recipe: unowned variable-expanded interpreter in {name}: {expanded}").into());
                    }
                    expanded
                } else {
                    first.to_owned()
                };
                if let Some(child) = recipe_call(&expanded) {
                    if interpolated_target && !parsed.recipes.contains_key(child) {
                        return Err(format!(
                            "Just recipe: unresolved recipe target in {name}: {expanded}"
                        )
                        .into());
                    }
                    if parsed.recipes.contains_key(child) {
                        queue.push_back(child.to_owned());
                    }
                }
                continue;
            }
            let expanded = rendered_line(line, &parsed.assignments)?;
            if let Some(child) = recipe_call(&expanded)
                && parsed.recipes.contains_key(child)
            {
                queue.push_back(child.to_owned());
            }
            if python_executable(&expanded) {
                return Err(format!(
                    "Just recipe: unowned variable-expanded interpreter in {name}: {expanded}"
                )
                .into());
            }
        }
    }
    Ok(())
}

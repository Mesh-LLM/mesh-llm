use super::{super::just_recipes, Edge, GraphBuilder, just_bindings, recipe_sources};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::Path;

pub(super) fn expand(
    root: &Path,
    builder: &mut GraphBuilder<'_>,
    visited: &mut BTreeSet<(String, &'static str)>,
    recipes: &BTreeSet<String>,
) -> DynResult<()> {
    if !root.join("Justfile").is_file() || recipes.is_empty() {
        return Ok(());
    }
    let dump = just_recipes::dump(root)?;
    let mut queue = recipes
        .iter()
        .map(|name| just_bindings::Invocation {
            recipe: name.clone(),
            arguments: Vec::new(),
        })
        .collect::<std::collections::VecDeque<_>>();
    let mut seen = BTreeSet::new();
    while let Some(invocation) = queue.pop_front() {
        let name = invocation.recipe;
        if !seen.insert((name.clone(), invocation.arguments.clone())) {
            continue;
        }
        let recipe = dump
            .recipes
            .get(&name)
            .ok_or_else(|| format!("Just recipe: missing required recipe {name}"))?;
        for dependency in &recipe.dependencies {
            match just_bindings::dependency(dependency) {
                Some(child) => queue.push_back(child),
                None => unknown(
                    builder,
                    &name,
                    &dependency.recipe,
                    "unresolved Just dependency argument",
                ),
            }
        }
        let mut continued = String::new();
        for line in just_bindings::render(recipe, &invocation.arguments, &dump) {
            let command = match line.result {
                Ok(command) => command,
                Err(reason) => {
                    if line.source.trim_start().starts_with("#")
                        || line.source.trim_start().starts_with("echo ")
                    {
                        continue;
                    }
                    continued.clear();
                    unknown(builder, &name, &line.source, &reason);
                    continue;
                }
            };
            if let Some(head) = command.trim_end().strip_suffix('\\') {
                continued.push_str(head.trim());
                continued.push(' ');
                continue;
            }
            continued.push_str(command.trim());
            let command = std::mem::take(&mut continued);
            let block = command.trim();
            if block.starts_with('#') || block.starts_with("echo ") {
                continue;
            }
            match just_bindings::call(block) {
                Ok(Some(child)) => {
                    if dump.recipes.contains_key(&child.recipe) {
                        builder.edges.push(Edge {
                            parent: format!("just:{name}"),
                            line: 0,
                            source_block: block.to_owned(),
                            trust_revision: "same_commit",
                            status: if platform_reason(recipe).is_some() {
                                "optional_branch"
                            } else {
                                "selected"
                            },
                            child: Some(format!("just:{}", child.recipe)),
                            unresolved_reason: platform_reason(recipe),
                            argv: Some(block.to_owned()),
                            status_streams_effects: None,
                            contract_source: None,
                        });
                        queue.push_back(child);
                    } else {
                        unknown(builder, &name, block, "unresolved Just recipe target");
                    }
                    continue;
                }
                Err(reason) => {
                    unknown(builder, &name, block, &reason);
                    continue;
                }
                Ok(None) => {}
            }
            if block == "exec" {
                unknown(
                    builder,
                    &name,
                    block,
                    "required executable argument is unbound",
                );
                continue;
            }
            if let Some(argv) = block.strip_prefix("exec ") {
                if argv.is_empty() {
                    unknown(
                        builder,
                        &name,
                        block,
                        "required executable argument is unbound",
                    );
                    continue;
                }
                builder.edges.push(Edge {
                    parent: format!("just:{name}"),
                    line: 0,
                    source_block: block.to_owned(),
                    trust_revision: "same_commit",
                    status: if platform_reason(recipe).is_some() {
                        "optional_branch"
                    } else {
                        "selected"
                    },
                    child: None,
                    unresolved_reason: platform_reason(recipe),
                    argv: Some(argv.to_owned()),
                    status_streams_effects: None,
                    contract_source: None,
                });
            }
            for child in super::sources::tokens(block) {
                if let Some(reason) = platform_reason(recipe) {
                    let context = super::contexts::classify(block, child);
                    if matches!(
                        context,
                        super::contexts::Context::Launch | super::contexts::Context::Optional
                    ) {
                        if !builder.known.contains(child) {
                            return Err(format!(
                                "required graph: missing conditional child {child} from just:{name}"
                            )
                            .into());
                        }
                        builder.edges.push(Edge {
                            parent: format!("just:{name}"),
                            line: 0,
                            source_block: block.to_owned(),
                            trust_revision: "same_commit",
                            status: "optional_branch",
                            child: Some(child.to_owned()),
                            unresolved_reason: Some(reason),
                            argv: None,
                            status_streams_effects: None,
                            contract_source: None,
                        });
                    }
                } else {
                    builder.push_edge(&format!("just:{name}"), 0, block, child, "same_commit")?;
                }
            }
        }
    }
    recipe_sources::expand(root, builder, visited)?;
    Ok(())
}

fn platform_reason(recipe: &just_recipes::Recipe) -> Option<String> {
    recipe
        .attributes
        .iter()
        .any(|attribute| matches!(attribute.as_str(), "macos" | "linux" | "windows" | "unix"))
        .then(|| {
            format!(
                "Just recipe platform selection: {}",
                recipe.attributes.join(", ")
            )
        })
}

fn unknown(builder: &mut GraphBuilder<'_>, recipe: &str, block: &str, reason: &str) {
    builder.edges.push(Edge {
        parent: format!("just:{recipe}"),
        line: 0,
        source_block: block.to_owned(),
        trust_revision: "same_commit",
        status: "unknown_selection",
        child: None,
        unresolved_reason: Some(reason.to_owned()),
        argv: None,
        status_streams_effects: None,
        contract_source: None,
    });
}

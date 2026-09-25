mod checks;
mod exception_policy;
mod just_recipes;
mod ledger;
mod loader_closure;
mod manual_calls;
mod other_shard;
mod python_shard;
mod required_closure;
mod required_graph;
#[cfg(test)]
mod required_graph_tests;
mod scan;
mod script_source_calls;
mod selected_process;
mod shard_rows;
mod shards;
mod test_shard;

use crate::command::DynResult;
use checks::{check_inventory, check_policy};
use ledger::{MigrationLedgers, tracked_paths};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let root = crate::repo_consistency::repo_root()?;
    let ledgers = MigrationLedgers::load(&root)?;
    let paths = tracked_paths(&root)?;
    let observed = scan::scan_paths(&root, &paths)?;
    let validated = shards::check_shards(&root, &observed)?;
    match args {
        [command, flag] if command == "graph" && flag == "--json" => {
            let roots = required_closure::required_roots(&root, &paths)?;
            let refs = roots.iter().map(String::as_str).collect::<Vec<_>>();
            let graph = required_graph::report(&root, &paths, &observed, &validated, &refs)?;
            println!("{}", serde_json::to_string_pretty(&graph)?);
            return required_graph::require_census(&graph);
        }
        [command, flag] if command == "inventory" && flag == "--check" => {
            check_inventory(&root, &paths, &ledgers, &observed, &validated)?
        }
        [command, flag] if command == "policy" && flag == "--check" => {
            check_policy(&root, &paths, &ledgers, &observed, &validated)?
        }
        _ => {
            return Err(
                "usage: cargo xtool automation {inventory|policy} --check | graph --json".into(),
            );
        }
    }
    println!("automation migration check passed: {}", args[0]);
    Ok(())
}

#[cfg(test)]
mod just_recipe_tests;
#[cfg(test)]
mod loader_closure_tests;
#[cfg(test)]
mod other_shard_tests;
#[cfg(test)]
mod required_closure_tests;
#[cfg(test)]
mod required_workflow_tests;
#[cfg(test)]
mod script_source_tests;
#[cfg(test)]
mod selected_process_tests;
#[cfg(test)]
mod shard_tests;
#[cfg(test)]
mod test_shard_tests;
#[cfg(test)]
mod tests;

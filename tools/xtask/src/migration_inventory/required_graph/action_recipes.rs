//! Finite source-owned Skippy CLI profile selection; no shell-variable interpreter.
use super::just_bindings::Invocation;
const ACTION: &str = ".github/actions/prepare-skippy-cli-input/action.yml";
const SELECTOR: [&str; 5] = [
    "case \"$INPUT_PROFILE\" in",
    "debug|dev) recipe=skippy-cli-build; profile_dir=debug ;;",
    "release) recipe=skippy-cli-release-build; profile_dir=release ;;",
    "*) echo \"unsupported Skippy profile: $INPUT_PROFILE\" >&2; exit 1 ;;",
    "esac",
];
pub(super) fn selected(
    path: &str,
    text: &str,
    line: usize,
    block: &str,
) -> Result<Option<Vec<Invocation>>, String> {
    if path != ACTION || block != "just \"$recipe\"" {
        return Ok(None);
    }
    let lines = text.lines().map(str::trim).collect::<Vec<_>>();
    let preceding = line
        .checked_sub(6)
        .and_then(|start| lines.get(start..start + 5));
    if preceding != Some(SELECTOR.as_slice()) {
        return Err(
            "Skippy CLI recipe selector requires its exact finite profile case and refusal branch"
                .into(),
        );
    }
    Ok(Some(
        ["skippy-cli-build", "skippy-cli-release-build"]
            .map(|recipe| Invocation {
                recipe: recipe.into(),
                arguments: Vec::new(),
            })
            .into(),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        format!("{}\njust \"$recipe\"\n", SELECTOR.join("\n"))
    }
    #[test]
    fn action_recipe_selector_retains_only_both_closed_native_recipes() {
        let selected = selected(ACTION, &fixture(), 6, "just \"$recipe\"")
            .unwrap()
            .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|row| row.recipe.as_str())
                .collect::<Vec<_>>(),
            ["skippy-cli-build", "skippy-cli-release-build"]
        );
        assert!(selected.iter().all(|row| row.arguments.is_empty()));
    }
    #[test]
    fn action_recipe_selector_refuses_retarget_missing_exit_and_intervening_assignment() {
        for mutation in [
            fixture().replace("recipe=skippy-cli-build", "recipe=foreign"),
            fixture().replace("exit 1", "true"),
            fixture().replace("esac\n", "esac\nrecipe=foreign\n"),
        ] {
            let line = mutation.lines().count();
            assert!(selected(ACTION, &mutation, line, "just \"$recipe\"").is_err());
        }
        assert!(
            selected(
                ".github/actions/foreign/action.yml",
                &fixture(),
                6,
                "just \"$recipe\""
            )
            .unwrap()
            .is_none()
        );
        assert!(
            selected(ACTION, &fixture(), 6, "just \"$recipe\" extra")
                .unwrap()
                .is_none()
        );
    }
}
#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::{
        command::DynResult,
        migration_inventory::{required_graph::report, required_graph_tests::source},
    };
    use std::collections::BTreeSet;
    #[test]
    fn finite_action_profile_walks_both_actual_recipe_bodies() -> DynResult<()> {
        let root = crate::command::unique_temp_dir("action-profile-recipes");
        source(
            &root,
            ACTION,
            &format!(
                "run: |\n    {}\n    just \"$recipe\"\n",
                SELECTOR.join("\n    ")
            ),
        )?;
        source(
            &root,
            "Justfile",
            "default:\n    echo unused\nskippy-cli-build:\n    exec cargo check -p first-native\nskippy-cli-release-build:\n    exec cargo check -p second-native\n",
        )?;
        let paths = [ACTION, "Justfile"].map(str::to_owned);
        let graph = report(&root, &paths, &[], &BTreeSet::new(), &[ACTION])?;
        for (recipe, command) in [
            ("just:skippy-cli-build", "exec cargo check -p first-native"),
            (
                "just:skippy-cli-release-build",
                "exec cargo check -p second-native",
            ),
        ] {
            assert!(
                graph
                    .selected_recipe_commands
                    .iter()
                    .any(|(owner, argv)| owner == recipe && argv == command)
            );
            assert!(graph.edges.iter().any(|edge| edge.parent == ACTION
                && edge.child.as_deref() == Some(recipe)
                && edge.status == "unknown_selection"));
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}

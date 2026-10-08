//! Native Just owns import syntax; inventory owns selected recipe closure.
//! Finite source fixtures only: no recipe body is executed.
use super::just_recipes::{check_recipe_children, dump};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

struct Fixture(tempfile::TempDir);

impl Fixture {
    fn new() -> DynResult<Self> {
        Ok(Self(
            tempfile::Builder::new()
                .prefix("native just import ")
                .tempdir()?,
        ))
    }

    fn root(&self) -> &Path {
        self.0.path()
    }

    fn source(&self, path: &str, text: &str) -> DynResult<()> {
        let target = self.root().join(path);
        fs::create_dir_all(target.parent().ok_or("fixture path parent")?)?;
        fs::write(target, text)?;
        Ok(())
    }

    fn selected(&self) -> DynResult<()> {
        check_recipe_children(self.root(), &BTreeSet::from(["ci-validate".to_owned()]))
    }

    fn rejected_parse(&self) {
        let parsed = match dump(self.root()) {
            Ok(_) => panic!("native Just unexpectedly accepted malformed import fixture"),
            Err(error) => error,
        };
        assert!(
            parsed.to_string().contains("Just recipe: parse failed:"),
            "{parsed}"
        );
        let selected = self.selected().unwrap_err();
        assert!(
            selected.to_string().contains("Just recipe: parse failed:"),
            "{selected}"
        );
        assert!(!self.root().join("recipe-executed").exists());
    }
}

#[test]
fn native_nested_relative_imports_reach_selected_recipe_and_interpreter_closure() -> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import 'just/build.just'\nci-validate: child-check\n",
    )?;
    fixture.source("just/build.just", "import 'nested/runtime.just'\n")?;
    fixture.source(
        "just/nested/runtime.just",
        "child-check:\n    touch recipe-executed\n",
    )?;
    let parsed = dump(fixture.root())?;
    assert!(parsed.recipes.contains_key("ci-validate"));
    assert!(parsed.recipes.contains_key("child-check"));
    assert_eq!(parsed.recipes["ci-validate"].dependencies.len(), 1);
    assert_eq!(
        parsed.recipes["ci-validate"].dependencies[0].recipe,
        "child-check"
    );
    fixture.selected()?;
    assert!(!fixture.root().join("recipe-executed").exists());

    // A missing nested import must propagate, never leave a partial recipe set.
    fs::remove_file(fixture.root().join("just/nested/runtime.just"))?;
    fixture.rejected_parse();

    // Restore the imported recipe with a selected interpreter to prove closure.
    fixture.source(
        "just/nested/runtime.just",
        "interpreter := 'python3'\nchild-check:\n    {{ interpreter }} -c 'pass'\n",
    )?;
    let error = fixture.selected().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unowned variable-expanded interpreter"),
        "{error}"
    );
    assert!(!fixture.root().join("recipe-executed").exists());
    Ok(())
}

#[test]
fn native_missing_import_rejects_dump_and_required_recipe_selection() -> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import 'missing.just'\nci-validate:\n    touch recipe-executed\n",
    )?;
    fixture.rejected_parse();
    Ok(())
}

#[test]
fn native_direct_import_cycle_rejects_dump_and_required_recipe_selection() -> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import 'other.just'\nci-validate:\n    touch recipe-executed\n",
    )?;
    fixture.source("other.just", "import 'Justfile'\n")?;
    fixture.rejected_parse();
    Ok(())
}

#[test]
fn native_normalized_import_cycle_rejects_dump_and_required_recipe_selection() -> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import 'just/build.just'\nci-validate:\n    touch recipe-executed\n",
    )?;
    fixture.source("just/build.just", "import '../Justfile'\n")?;
    fixture.rejected_parse();
    Ok(())
}

#[test]
fn native_optional_import_and_indented_commands_preserve_parser_and_closure_ownership()
-> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import? 'optional.just'\nci-validate:\n    mod build\n    touch recipe-executed\n",
    )?;
    let parsed = dump(fixture.root())?;
    assert_eq!(parsed.recipes.len(), 1);
    let body = &parsed.recipes["ci-validate"].body;
    assert!(
        body.iter()
            .any(|line| line.iter().any(|part| part.as_str() == Some("mod build")))
    );
    fixture.selected()?;
    assert!(!fixture.root().join("recipe-executed").exists());

    // Optional absence is valid native syntax; presence cannot hide a selected child.
    fixture.source(
        "optional.just",
        "interpreter := 'python3'\nchild-check:\n    {{ interpreter }} -c 'pass'\n",
    )?;
    fixture.source(
        "Justfile",
        "import? 'optional.just'\nci-validate: child-check\n",
    )?;
    let error = fixture.selected().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unowned variable-expanded interpreter"),
        "{error}"
    );
    // Native module syntax is also valid; only the root recipe is selected here.
    // This does not qualify execution closure through a namespaced invocation.
    fixture.source("build.just", "safe:\n    true\n")?;
    fixture.source(
        "Justfile",
        "mod build\nci-validate:\n    touch recipe-executed\n",
    )?;
    assert!(dump(fixture.root())?.recipes.contains_key("ci-validate"));
    fixture.selected()?;
    assert!(!fixture.root().join("recipe-executed").exists());
    Ok(())
}

#[test]
fn native_duplicate_imported_recipe_is_not_silently_overwritten() -> DynResult<()> {
    let fixture = Fixture::new()?;
    fixture.source(
        "Justfile",
        "import 'other.just'\nci-validate:\n    touch recipe-executed\n",
    )?;
    fixture.source("other.just", "ci-validate:\n    true\n")?;
    fixture.rejected_parse();
    Ok(())
}

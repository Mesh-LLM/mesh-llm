use super::*;
use crate::migration_inventory::scan;
use std::collections::BTreeSet;

const FORMER_UI_DEPENDENCIES: &str = "/crates/mesh-llm-ui/node_modules/";

fn git(root: &Path, args: &[&str]) -> DynResult<()> {
    let output = run_command(Command::new("git").current_dir(root).args(args))?;
    if !output.status.success() {
        return Err(trimmed_stderr_or_stdout(&output).into());
    }
    Ok(())
}

fn source(root: &Path, path: &str) -> DynResult<()> {
    let file = root.join(path);
    fs::create_dir_all(file.parent().unwrap())?;
    fs::write(file, "subprocess.run(['python3', 'child.py'])\n")?;
    Ok(())
}

#[test]
fn generated_former_ui_dependencies_do_not_hide_tracked_or_maintained_python() -> DynResult<()> {
    // Bind the fixture to the narrowly scoped checked-in dependency rule.
    assert!(
        include_str!("../../../../../.gitignore")
            .lines()
            .any(|line| line == FORMER_UI_DEPENDENCIES)
    );
    let temp = tempfile::tempdir()?;
    let root = temp.path();
    git(root, &["init", "--quiet"])?;
    fs::write(
        root.join(".gitignore"),
        format!("{FORMER_UI_DEPENDENCIES}\n"),
    )?;
    let ignored = "crates/mesh-llm-ui/node_modules/katex/vendor.py";
    let tracked = "crates/mesh-llm-ui/node_modules/katex/tracked.py";
    source(root, ignored)?;
    source(root, tracked)?;
    git(root, &["add", "--force", "--", tracked])?;
    let maintained = [
        "scripts/new.py",
        "mesh/scripts/new.py",
        "skippy/scripts/new.py",
        "crates/new-owner/new.py",
        "mesh/crates/new-owner/new.py",
        "skippy/crates/new-owner/new.py",
        "new.py",
        "crates/mesh-llm-ui/node_modules-tools/new.py",
    ];
    for path in maintained {
        source(root, path)?;
    }

    // The collector applies Git ownership; the scanner still detects every admitted call.
    let paths = tracked_paths(root)?;
    assert!(!paths.iter().any(|path| path == ignored));
    assert!(paths.iter().any(|path| path == tracked));
    let observed = scan::scan_paths(root, &paths)?;
    let actual = observed
        .iter()
        .map(|row| row.path.as_str())
        .collect::<BTreeSet<_>>();
    let expected = maintained
        .into_iter()
        .chain([tracked])
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(observed.len(), expected.len());
    assert_eq!(
        fs::read_to_string(root.join(ignored))?,
        "subprocess.run(['python3', 'child.py'])\n"
    );
    Ok(())
}

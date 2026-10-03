//! Flat Just facade ownership checked against native Just parsing and visibility.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
struct Ownership {
    imports: Vec<String>,
    recipes: BTreeMap<String, BTreeSet<String>>,
}
fn ownership() -> Ownership {
    serde_json::from_str(include_str!("just_layout_contracts/recipe-ownership.json")).unwrap()
}
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn just(root: &Path, arguments: &[&str]) -> Result<Vec<u8>, String> {
    let search = std::env::var_os("PATH").ok_or("native Just PATH required")?;
    let executable = std::env::split_paths(&search)
        .map(|p| p.join("just"))
        .find(|p| p.is_file())
        .ok_or("native Just executable required")?;
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: executable.canonicalize().map_err(|e| e.to_string())?,
            cwd: root.to_owned(),
            arguments: arguments
                .iter()
                .map(|v| Value::Public((*v).into()))
                .collect(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public(search)),
                ("HOME".into(), Value::Public(root.into())),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 4 * 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(4 * 1024 * 1024),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .map_err(|e| e.to_string())?;
    if !report.process.cleanup.complete
        || report.process.failure.is_some()
        || report.process.status.is_none_or(|s| !s.success())
    {
        return Err(format!(
            "native Just refusal: {}",
            String::from_utf8_lossy(&report.process.stderr.bytes_retained)
        ));
    }
    report
        .stdout
        .map(|bytes| bytes.as_bytes().to_vec())
        .ok_or_else(|| "native Just stdout missing".into())
}
fn dump(root: &Path) -> Result<serde_json::Value, String> {
    serde_json::from_slice(&just(root, &["--dump", "--dump-format", "json"])?)
        .map_err(|e| e.to_string())
}

// Source declarations bind file ownership. Native Just validates their syntax,
// selected definitions, first recipe, dependency graph and visibility below.
fn declaration(line: &str) -> Option<&str> {
    if line.starts_with(char::is_whitespace) || !line.contains(':') || line.contains(":=") {
        return None;
    }
    let word = line.split_whitespace().next()?.trim_end_matches(':');
    (word
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        && word.chars().next()?.is_ascii_alphabetic())
    .then_some(word)
}
fn check(root: &Path) -> Result<(), String> {
    let policy = ownership();
    let source = fs::read_to_string(root.join("Justfile")).map_err(|e| e.to_string())?;
    let imports = source
        .lines()
        .filter_map(|line| line.strip_prefix("import '")?.strip_suffix('\''))
        .collect::<Vec<_>>();
    if imports != policy.imports {
        return Err("flat import order changed".into());
    }
    if source
        .lines()
        .any(|line| line.starts_with("mod ") || line.starts_with("import?"))
    {
        return Err("unexpected module or optional import".into());
    }
    let parsed = dump(root)?;
    if parsed["first"] != "default"
        || parsed["modules"] != serde_json::json!({})
        || parsed["recipes"]["default"]["dependencies"][0]["recipe"] != "build"
    {
        return Err("default recipe or flat module contract changed".into());
    }
    let mut known = policy
        .recipes
        .values()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    known.insert("default".into());
    for (file, expected) in policy.recipes {
        let source = fs::read_to_string(root.join(&file)).map_err(|e| e.to_string())?;
        let actual = source
            .lines()
            .filter_map(declaration)
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(format!("recipe ownership changed: {file}"));
        }
    }
    for (name, recipe) in parsed["recipes"]
        .as_object()
        .ok_or("native recipe map missing")?
    {
        if !known.contains(name) {
            return Err(format!("unowned native recipe: {name}"));
        }
        let expected = matches!(name.as_str(), "with-lld" | "automation-run");
        if recipe["private"].as_bool() != Some(expected) {
            return Err(format!("private visibility changed: {name}"));
        }
    }
    let summary = String::from_utf8(just(root, &["--summary"])?).map_err(|e| e.to_string())?;
    let public = summary.split_whitespace().collect::<BTreeSet<_>>();
    if public.contains("with-lld") || public.contains("automation-run") {
        return Err("internal facade listed as public".into());
    }
    Ok(())
}
struct Fixture(tempfile::TempDir);
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        for file in std::iter::once("Justfile".to_owned()).chain(ownership().imports) {
            let target = temp.path().join(&file);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(repo().join(file), target).unwrap();
        }
        let target = temp.path().join("scripts/lib/macos-deployment-target.txt");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(
            repo().join("scripts/lib/macos-deployment-target.txt"),
            target,
        )
        .unwrap();
        Self(temp)
    }
    fn root(&self) -> &Path {
        self.0.path()
    }
    fn replace(&self, file: &str, from: &str, to: &str) {
        let path = self.root().join(file);
        let source = fs::read_to_string(&path).unwrap();
        assert!(source.contains(from), "mutation anchor absent");
        fs::write(path, source.replacen(from, to, 1)).unwrap();
    }
}
#[test]
fn just_layout_actual_facade_keeps_flat_imports_recipe_owners_and_native_visibility() {
    check(&repo()).unwrap();
    check(Fixture::new().root()).unwrap();
}
#[test]
fn just_layout_import_reorder_and_default_change_are_refused() {
    let fixture = Fixture::new();
    fixture.replace(
        "Justfile",
        "import 'just/build.just'",
        "import 'just/release-build.just'",
    );
    assert!(check(fixture.root()).unwrap_err().contains("import order"));
    let fixture = Fixture::new();
    fixture.replace("Justfile", "default: build", "default: ui-test");
    assert!(
        check(fixture.root())
            .unwrap_err()
            .contains("default recipe")
    );
}
#[test]
fn just_layout_recipe_move_preserves_native_parse_but_refuses_wrong_owner() {
    let fixture = Fixture::new();
    let source = fs::read_to_string(repo().join("just/utilities.just")).unwrap();
    let start = source.find("[unix]\nhooks-install:\n").unwrap();
    let end = start + source[start..].find("\n\n").unwrap() + 2;
    let recipe = &source[start..end];
    fixture.replace("just/utilities.just", recipe, "");
    let target = fixture.root().join("just/build.just");
    fs::write(
        &target,
        format!("{}\n{recipe}", fs::read_to_string(&target).unwrap()),
    )
    .unwrap();
    dump(fixture.root()).unwrap();
    assert!(
        check(fixture.root())
            .unwrap_err()
            .contains("recipe ownership")
    );
}
#[test]
fn just_layout_private_facade_exposure_is_refused_by_native_metadata_and_summary() {
    let fixture = Fixture::new();
    fixture.replace("just/ci.just", "[private]\n[unix]", "[unix]");
    let native = dump(fixture.root()).unwrap();
    assert_eq!(native["recipes"]["automation-run"]["private"], false);
    assert!(
        String::from_utf8(just(fixture.root(), &["--summary"]).unwrap())
            .unwrap()
            .split_whitespace()
            .any(|v| v == "automation-run")
    );
    assert!(
        check(fixture.root())
            .unwrap_err()
            .contains("private visibility")
    );
}

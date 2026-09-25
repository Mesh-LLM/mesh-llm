//! Every frozen fixture case: Rust against its golden plan or diagnostic,
//! the recorded action outputs for every plan, and (when enabled) the legacy
//! planner against Rust on identical inputs.

use super::Run;
use super::action::real_action_outputs;
use super::process::{Captured, first_difference};
use super::projection::{action_outputs, outputs_difference};
use super::stage::Tools;
use crate::command::DynResult;
use serde_json::Value;
use std::fs;
use std::path::Path;

struct Case {
    name: String,
    manifest: String,
    input: Vec<u8>,
}

fn cases(fixtures: &Path) -> DynResult<Vec<Case>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(fixtures.join("cases"))? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            names.push(path);
        }
    }
    names.sort();
    names
        .into_iter()
        .map(|path| {
            let document: Value = serde_json::from_slice(&fs::read(&path)?)?;
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or("case file name")?
                .to_owned();
            let manifest = document["manifest"]
                .as_str()
                .ok_or("case manifest")?
                .to_owned();
            let input = serde_json::to_vec(&document["input"])?;
            Ok(Case {
                name,
                manifest,
                input,
            })
        })
        .collect()
}

/// Rust against the golden: exact plan bytes plus newline, or the exact
/// diagnostic with the scratch manifest directory shown as `<manifests>/`.
fn golden_difference(run: &Run<'_>, case: &Case, rust: &Captured) -> DynResult<Option<String>> {
    let expected = run.stage.fixtures().join("expected");
    let plan = expected.join(format!("{}.plan.json", case.name));
    if plan.is_file() {
        let mut golden = fs::read(plan)?;
        golden.push(b'\n');
        let expected = Captured {
            stdout: golden,
            stderr: Vec::new(),
            code: Some(0),
        };
        return Ok(super::process::process_difference(&expected, rust));
    }
    let golden = fs::read(expected.join(format!("{}.error.txt", case.name)))?;
    let placeholder = format!("{}/", run.stage.manifests().display());
    let stderr = String::from_utf8_lossy(&rust.stderr).replace(&placeholder, "<manifests>/");
    let expected = Captured {
        stdout: Vec::new(),
        stderr: golden,
        code: Some(2),
    };
    let actual = Captured {
        stdout: rust.stdout.clone(),
        stderr: stderr.into_bytes(),
        code: rust.code,
    };
    Ok(super::process::process_difference(&expected, &actual))
}

/// The Rust projection against the recorded outputs of the real action.
fn outputs(run: &mut Run<'_>, case: &Case, rust: &Captured) -> DynResult<()> {
    let recorded = run
        .stage
        .fixtures()
        .join("expected")
        .join(format!("{}.outputs.txt", case.name));
    let projected = action_outputs(&rust.stdout);
    let difference = match (&projected, fs::read_to_string(&recorded)) {
        (Ok(projected), Ok(recorded)) => outputs_difference(&recorded, projected),
        (Err(error), _) => Some(format!("projection failed: {error}")),
        (_, Err(error)) => Some(format!("missing {}: {error}", recorded.display())),
    };
    run.ledger
        .record("action-outputs", case.name.clone(), difference);
    if let Ok(projected) = projected {
        run.keep(
            &format!("frozen/{}.outputs.txt", case.name),
            projected.as_bytes(),
        )?;
    }
    Ok(())
}

fn compare_case(run: &mut Run<'_>, case: &Case) -> DynResult<()> {
    let manifest_root = run.stage.manifest_set(&case.manifest)?;
    let manifest_arg = manifest_root
        .to_str()
        .ok_or("non-UTF8 scratch path")?
        .to_owned();
    let args = ["--manifest-root", manifest_arg.as_str()];
    let search_path = run.stage.search_path(Tools::FixtureCargo);
    let rust = run.rust_plan(&args, &search_path, &case.input)?;
    run.keep(&format!("frozen/{}.rust.stdout", case.name), &rust.stdout)?;
    let difference = golden_difference(run, case, &rust)?;
    run.ledger
        .record("frozen-cases", case.name.clone(), difference);
    let legacy = run.compare_legacy(
        "frozen-cases",
        &case.name,
        &rust,
        (&args, &search_path, &case.input),
    )?;
    if rust.code != Some(0) {
        return Ok(());
    }
    outputs(run, case, &rust)?;
    if let Some(legacy) = legacy.filter(|legacy| legacy.code == Some(0)) {
        let actual = real_action_outputs(run, &case.name, &legacy.stdout)?;
        let expected = action_outputs(&rust.stdout).map_err(|error| error.to_string())?;
        let difference = outputs_difference(&expected, &actual)
            .or_else(|| first_difference(expected.as_bytes(), actual.as_bytes()));
        run.ledger.record(
            "action-outputs",
            format!("{} legacy-action-vs-rust", case.name),
            difference,
        );
    }
    Ok(())
}

pub(super) fn compare(run: &mut Run<'_>) -> DynResult<()> {
    let cases = cases(run.stage.fixtures())?;
    if cases.is_empty() {
        return Err("no frozen cases found".into());
    }
    for case in &cases {
        compare_case(run, case)?;
    }
    Ok(())
}

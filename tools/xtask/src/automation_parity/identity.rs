//! Runner-image identity's hidden planner import and the planner's command
//! surface. `runner-image-identity.py check|diagnose` loads the protected
//! planner and reads `_select_rows(main, force_all_rows)` runtime rows; that
//! selection is exactly the Rust main plan's `runtime_products`, so the Rust
//! side proves the same row contract against `ci/runner-images.json`.

use super::Run;
use super::process::Captured;
use super::report::Outcome;
use super::stage::Tools;
use crate::command::DynResult;
use serde_json::Value;
use std::fs;

const MAIN: &[u8] = br#"{"profile":"main","event_name":"push","source_sha":"0000000000000000000000000000000000000000","base_sha":"","changed_files":[]}"#;

/// The planner-row half of the identity `check`, in Rust: returns the
/// number of container-image rows, or the first drift.
fn row_contract(plan: &Value, catalog: &Value) -> Result<usize, String> {
    let rows = plan["matrices"]["runtime_products"]
        .as_array()
        .ok_or("main plan has no runtime_products")?;
    let imaged = rows
        .iter()
        .filter(|row| row.get("container_image").is_some())
        .collect::<Vec<_>>();
    let expected = catalog["runtime_rows"]
        .as_object()
        .ok_or("catalog has no runtime_rows")?;
    let mut ids = imaged
        .iter()
        .filter_map(|row| row["id"].as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    let before = ids.len();
    ids.dedup();
    if before != ids.len() || before != imaged.len() {
        return Err("duplicate planner image row".to_owned());
    }
    if ids != expected.keys().map(String::as_str).collect::<Vec<_>>() {
        return Err(format!("planner image row census drift: {ids:?}"));
    }
    for row in imaged {
        let id = row["id"].as_str().unwrap_or_default();
        let wanted = &expected[id];
        for field in ["platform", "architecture", "backend"] {
            if row[field] != wanted[field] {
                return Err(format!("{id}: planner {field} drift"));
            }
        }
        let image = &catalog["images"][wanted["image_id"].as_str().unwrap_or_default()];
        if row["container_image"] != image["reference"] {
            return Err(format!("{id}: planner image drift"));
        }
        if row["toolchain_epoch"] != image["native_toolchain_epoch"] {
            return Err(format!("{id}: planner epoch drift"));
        }
    }
    Ok(before)
}

/// Rust's row contract, and the legacy loader's own verdict when enabled.
pub(super) fn runner_image(run: &mut Run<'_>) -> DynResult<()> {
    let search_path = run.stage.search_path(Tools::Real);
    let plan = run.rust_plan(&[], &search_path, MAIN)?;
    let catalog: Value =
        serde_json::from_slice(&fs::read(run.root.join("ci/runner-images.json"))?)?;
    let parsed: Value = serde_json::from_slice(&plan.stdout).unwrap_or(Value::Null);
    let rows = row_contract(&parsed, &catalog);
    run.ledger.record(
        "runner-image-identity",
        "rust planner rows",
        rows.clone().err(),
    );
    let Some(legacy) = &run.legacy else {
        return Ok(());
    };
    let check = legacy.identity("check", &search_path)?;
    let diagnose = legacy.identity("diagnose", &search_path)?;
    run.keep("runner-image-identity/check.stdout", &check.stdout)?;
    run.keep("runner-image-identity/diagnose.stdout", &diagnose.stdout)?;
    let reported: Value = serde_json::from_slice(&check.stdout).unwrap_or(Value::Null);
    let difference = match (check.code, diagnose.code, rows) {
        (Some(0), Some(0), Ok(count)) if reported["runtime_rows"] == count => None,
        (Some(0), Some(0), Ok(count)) => Some(format!(
            "legacy reports {} rows, Rust {count}",
            reported["runtime_rows"]
        )),
        (check_code, diagnose_code, rows) => Some(format!(
            "legacy check {check_code:?} diagnose {diagnose_code:?}, Rust {rows:?}: {}",
            String::from_utf8_lossy(&check.stderr)
        )),
    };
    run.ledger.record(
        "runner-image-identity",
        "legacy check/diagnose vs rust rows",
        difference,
    );
    Ok(())
}

/// Argument, stdin and catalog failures shared with the legacy planner.
pub(super) fn command_surface(run: &mut Run<'_>) -> DynResult<()> {
    let empty = run.stage.directory("empty-manifest-root")?;
    let empty_arg = empty.to_str().ok_or("non-UTF8 scratch path")?.to_owned();
    let search_path = run.stage.search_path(Tools::FixtureCargo);
    let calls: [(&str, Vec<&str>, &[u8]); 3] = [
        ("unknown argument", vec!["--bogus"], MAIN),
        ("missing catalog", vec!["--manifest-root", &empty_arg], MAIN),
        ("non-JSON stdin", vec![], b"not json"),
    ];
    for (label, args, stdin) in calls {
        let rust = run.rust_plan(&args, &search_path, stdin)?;
        let Some(legacy) = &run.legacy else {
            let contract = rust.code == Some(2) && rust.stdout.is_empty();
            run.ledger.record(
                "command-surface",
                label,
                (!contract).then(|| format!("{rust:?}")),
            );
            continue;
        };
        let legacy = legacy.plan(&args, &search_path, stdin)?;
        let outcome = surface_outcome(label, &legacy, &rust);
        run.ledger.push(
            "command-surface",
            format!("{label} legacy-vs-rust"),
            outcome.0,
            outcome.1,
        );
    }
    Ok(())
}

/// Exact parity, except the documented JSON-parser wording on invalid stdin.
fn surface_outcome(label: &str, legacy: &Captured, rust: &Captured) -> (Outcome, String) {
    let Some(difference) = super::process::process_difference(legacy, rust) else {
        return (Outcome::Identical, String::new());
    };
    let prefix = b"ERROR: unable to build CI plan: ";
    let same_shape = legacy.code == rust.code
        && legacy.stdout == rust.stdout
        && legacy.stderr.starts_with(prefix)
        && rust.stderr.starts_with(prefix);
    if label == "non-JSON stdin" && same_shape {
        let detail = "known: parser detail is serde_json wording, not Python JSONDecodeError; prefix, stdout and status 2 match";
        return (Outcome::Explained, detail.to_owned());
    }
    (Outcome::Different, difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migration_ci_shadow_row_contract_detects_image_drift() {
        let catalog = json!({
            "runtime_rows": {"linux-cpu": {"image_id": "cpu", "platform": "linux", "architecture": "amd64", "backend": "cpu"}},
            "images": {"cpu": {"reference": "ref", "native_toolchain_epoch": "epoch"}}
        });
        let row = json!({"id": "linux-cpu", "platform": "linux", "architecture": "amd64", "backend": "cpu", "container_image": "ref", "toolchain_epoch": "epoch"});
        let plan = json!({"matrices": {"runtime_products": [row.clone()]}});
        assert_eq!(row_contract(&plan, &catalog), Ok(1));
        let mut drifted = row;
        drifted["container_image"] = json!("other");
        let plan = json!({"matrices": {"runtime_products": [drifted]}});
        assert_eq!(
            row_contract(&plan, &catalog),
            Err("linux-cpu: planner image drift".to_owned())
        );
    }
}

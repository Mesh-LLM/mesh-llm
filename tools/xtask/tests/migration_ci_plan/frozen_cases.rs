//! Every frozen case: full plan bytes or the exact legacy diagnostic.

use crate::support::{
    Run, Stage, TestResult, assert_same_process, fixture_root, repository_root, text,
};
use serde_json::Value;
use std::fs;

/// Plans are hashed by the caller after `jq -c .`; the goldens hold those
/// bytes (no trailing newline) and the planner prints them plus `\n`.
struct Case {
    name: String,
    manifest: String,
    input: Vec<u8>,
}

fn load_case(name: &str) -> Result<Case, Box<dyn std::error::Error>> {
    let path = fixture_root().join("cases").join(format!("{name}.json"));
    let document: Value = serde_json::from_slice(&fs::read(path)?)?;
    let manifest = document["manifest"]
        .as_str()
        .ok_or("case manifest")?
        .to_owned();
    Ok(Case {
        name: name.to_owned(),
        manifest,
        input: serde_json::to_vec(&document["input"])?,
    })
}

fn all_case_names() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut names = fs::read_dir(fixture_root().join("cases"))?
        .map(|entry| {
            let path = entry?.path();
            let stem = path.file_stem().and_then(|stem| stem.to_str());
            Ok(stem.ok_or("case file name")?.to_owned())
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    names.sort();
    Ok(names)
}

/// Runs one case through the ported planner (and the legacy planner when
/// opted in) and compares it with its golden.
fn check_case(case: &Case) -> TestResult {
    let stage = Stage::new(&case.name)?;
    let manifest_root = stage.manifest_root(&case.manifest)?;
    let manifest_arg = manifest_root.to_str().ok_or("non-UTF8 scratch path")?;
    let path = stage.search_path()?;
    let run = Run {
        args: &["--manifest-root", manifest_arg],
        stdin: &case.input,
        path: &path,
    };
    let ported = run.ported()?;
    if let Some(legacy) = run.legacy()? {
        assert_same_process(&legacy, &ported, &case.name);
    }
    let expected = fixture_root().join("expected");
    let plan = expected.join(format!("{}.plan.json", case.name));
    let placeholder = format!("{}/", stage.path().join("manifests").display());
    if plan.is_file() {
        let golden = fs::read_to_string(plan)?;
        assert_eq!(text(&ported.stderr), "", "{}: stderr", case.name);
        assert_eq!(ported.status.code(), Some(0), "{}: status", case.name);
        assert_eq!(text(&ported.stdout), format!("{golden}\n"), "{}", case.name);
    } else {
        let golden = fs::read_to_string(expected.join(format!("{}.error.txt", case.name)))?;
        let stderr = text(&ported.stderr).replace(&placeholder, "<manifests>/");
        assert_eq!(stderr, golden, "{}: diagnostic", case.name);
        assert_eq!(text(&ported.stdout), "", "{}: stdout", case.name);
        assert_eq!(ported.status.code(), Some(2), "{}: status", case.name);
    }
    Ok(())
}

#[test]
fn migration_ci_plan_every_frozen_case_matches_its_golden() -> TestResult {
    // Given: every frozen legacy case and golden.
    let names = all_case_names()?;
    assert_eq!(names.len(), 63, "frozen case count");
    for name in names {
        // When/Then: each case reproduces its full plan bytes or diagnostic.
        check_case(&load_case(&name)?)?;
    }
    Ok(())
}

/// Named QA scenarios, so a regression names the behavior it broke.
macro_rules! frozen_case {
    ($test:ident, $case:literal) => {
        #[test]
        fn $test() -> TestResult {
            check_case(&load_case($case)?)
        }
    };
}

frozen_case!(migration_ci_plan_noop_draft, "noop-draft");
frozen_case!(migration_ci_plan_noop_ready, "noop-ready");
frozen_case!(migration_ci_plan_docs_only, "docs-only");
frozen_case!(migration_ci_plan_direct_runtime, "runtime");
frozen_case!(
    migration_ci_plan_reverse_dependency_computed,
    "reverse-dependency-metrics"
);
frozen_case!(
    migration_ci_plan_reverse_dependency_given,
    "reverse-dependency-log-store"
);
frozen_case!(migration_ci_plan_native_pin_escalates, "native-pin");
frozen_case!(migration_ci_plan_control_plane_fails_open, "control-ready");
frozen_case!(
    migration_ci_plan_manual_full_force_all,
    "manual-full-force-all"
);
frozen_case!(migration_ci_plan_plugin_exemplar_signal, "plugin-exemplar");
frozen_case!(
    migration_ci_plan_windows_unit_and_portable,
    "platform-windows"
);
frozen_case!(migration_ci_plan_core_smoke_fallback, "smoke-fallback");
frozen_case!(migration_ci_plan_rejects_unknown_path, "fail-unknown-path");
frozen_case!(
    migration_ci_plan_rejects_unknown_input_field,
    "fail-unknown-field"
);
frozen_case!(
    migration_ci_plan_rejects_schema_version,
    "fail-schema-version"
);
frozen_case!(migration_ci_plan_rejects_cycles, "fail-cycle");
frozen_case!(
    migration_ci_plan_rejects_duplicate_slices,
    "fail-duplicate-slice"
);
frozen_case!(
    migration_ci_plan_rejects_duplicate_rows,
    "fail-duplicate-row"
);
frozen_case!(migration_ci_plan_rejects_bad_matrix, "fail-macos-multiarch");
frozen_case!(
    migration_ci_plan_rejects_invalid_source_sha,
    "fail-source-sha"
);

#[test]
fn migration_ci_plan_real_catalogs_plan_every_profile() -> TestResult {
    // Given: the checked-in catalogs and the three routing profiles the
    // protected callers use most (main, manual-full, draft docs-only).
    let stage = Stage::new("real-catalogs")?;
    let path = stage.search_path()?;
    let root = repository_root();
    let root = root.to_str().ok_or("non-UTF8 checkout")?;
    let sha = "a".repeat(40);
    let inputs = [
        format!(
            r#"{{"profile":"main","event_name":"push","source_sha":"{sha}","base_sha":"","changed_files":["crates/mesh-llm/src/lib.rs"]}}"#
        ),
        format!(
            r#"{{"profile":"manual-full","event_name":"workflow_dispatch","source_sha":"{sha}","base_sha":"","changed_files":["__force_all__"]}}"#
        ),
        format!(
            r#"{{"profile":"pr-draft","event_name":"pull_request","source_sha":"{sha}","base_sha":"{sha}","changed_files":["CONTRIBUTING.md",".github/README.md"]}}"#
        ),
    ];
    for input in inputs {
        let run = Run {
            args: &["--manifest-root", root],
            stdin: input.as_bytes(),
            path: &path,
        };
        // When: the ported planner (and optionally the legacy one) runs.
        let ported = run.ported()?;
        if let Some(legacy) = run.legacy()? {
            assert_same_process(&legacy, &ported, &input);
        }
        // Then: a schema-version-1 plan is emitted on one line.
        assert_eq!(ported.status.code(), Some(0), "{}", text(&ported.stderr));
        let stdout = text(&ported.stdout);
        assert_eq!(stdout.matches('\n').count(), 1, "one line");
        let plan: Value = serde_json::from_str(&stdout)?;
        assert_eq!(plan["schema_version"], 1);
    }
    Ok(())
}

//! Current controller owns planning across nested selected-source checkouts.
//! The real selected shell battery consumes frozen bytes; native execution is disabled.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessReport, ProcessSpec, Readiness,
    Value,
};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap, ffi::OsString, fs, os::unix::fs::PermissionsExt, path::Path,
    time::Duration,
};

fn bounded(
    executable: &Path,
    cwd: &Path,
    arguments: Vec<OsString>,
    extra: BTreeMap<OsString, Value>,
) -> ProcessReport {
    let mut environment = ["PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"]
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
        .collect::<BTreeMap<_, _>>();
    environment.extend(extra);
    let spec = ProcessSpec {
        executable: executable.to_owned(),
        cwd: cwd.to_owned(),
        arguments: arguments.into_iter().map(Value::Public).collect(),
        environment,
    };
    // Eight execution seconds plus two cleanup seconds keep a ten-second budget.
    let limits = Limits {
        execution: Duration::from_secs(8),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise(
        &spec,
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.cleanup.complete, "{report:?}");
    assert!(
        !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    report
}

fn planner(source: &Path, plan: &Path) -> ProcessReport {
    bounded(
        Path::new(env!("CARGO_BIN_EXE_xtask")),
        source,
        vec![
            "ci".into(),
            "family-plan".into(),
            "--manifest".into(),
            source.join("ci/llama-canary/family-certified.json").into(),
            "--shard-count".into(),
            "256".into(),
            "--output".into(),
            plan.into(),
        ],
        BTreeMap::new(),
    )
}

fn selected_source(root: &Path, source: &Path) {
    fs::create_dir_all(source.join("scripts/lib")).unwrap();
    fs::create_dir_all(source.join("ci/llama-canary")).unwrap();
    fs::create_dir_all(source.join("tools/xtask")).unwrap();
    for relative in [
        "Cargo.toml",
        "tools/xtask/Cargo.toml",
        "scripts/skippy-family-battery.sh",
        "ci/llama-canary/family-certified.json",
    ] {
        fs::copy(root.join(relative), source.join(relative)).unwrap();
    }
    for entry in fs::read_dir(root.join("scripts/lib")).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            fs::copy(
                &path,
                source.join("scripts/lib").join(path.file_name().unwrap()),
            )
            .unwrap();
        }
    }
    let historical = source.join("scripts/plan-family-battery.py");
    fs::write(&historical, "#!/bin/bash\necho forbidden-historical-planner >&2\ntouch historical-planner-executed\nexit 91\n").unwrap();
    fs::set_permissions(historical, fs::Permissions::from_mode(0o755)).unwrap();
}

fn battery(source: &Path, plan: &Path, evidence: &Path, run: &str) -> ProcessReport {
    bounded(
        Path::new("/bin/bash"),
        source,
        vec![
            source.join("scripts/skippy-family-battery.sh").into(),
            "--skip-build".into(),
            "--dry-run".into(),
            "--plan".into(),
            plan.into(),
        ],
        BTreeMap::from([
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            (
                "FAMILY_BATTERY_ARTIFACT_ROOT".into(),
                Value::Public(evidence.into()),
            ),
            ("FAMILY_BATTERY_RUN_ID".into(), Value::Public(run.into())),
        ]),
    )
}

fn rejected_omission(source: &Path, plan: Json, directory: &Path, evidence: &Path) {
    let mut tampered = plan;
    tampered["selected_models"].as_array_mut().unwrap().pop();
    let invalid = directory.join("omitted family.json");
    fs::write(&invalid, serde_json::to_vec(&tampered).unwrap()).unwrap();
    let rejected = battery(source, &invalid, evidence, "invalid");
    assert!(!rejected.success(), "{rejected:?}");
    assert!(
        String::from_utf8_lossy(&rejected.stderr.bytes_retained)
            .contains("differs from the canonical manifest and selection")
    );
    assert!(!source.join("historical-planner-executed").exists());
    assert_eq!(
        fs::read_dir(evidence.join("invalid/model-scans"))
            .unwrap()
            .count(),
        0
    );
    assert!(
        fs::read(evidence.join("invalid/results.jsonl"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn nested_selected_battery_uses_frozen_controller_plan_and_ignores_historical_planner() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = directory
        .path()
        .join("outer checkout/selected historical source");
    selected_source(&root, &source);
    let plan = directory.path().join("admitted plan.json");
    let generated = planner(&source, &plan);
    assert!(generated.success(), "{generated:?}");
    let before = fs::read(&plan).unwrap();
    let parsed: Json = serde_json::from_slice(&before).unwrap();
    assert_eq!(parsed["manifest"], "ci/llama-canary/family-certified.json");
    assert!(!parsed["selected_models"].as_array().unwrap().is_empty());
    let evidence = directory.path().join("battery evidence");
    let admitted = battery(&source, &plan, &evidence, "valid");
    assert!(admitted.success(), "{admitted:?}");
    assert_eq!(fs::read(&plan).unwrap(), before);
    assert_eq!(
        fs::read(evidence.join("valid/policy-plan.json")).unwrap(),
        before
    );
    assert!(!source.join("historical-planner-executed").exists());
    assert!(
        !String::from_utf8_lossy(&admitted.stderr.bytes_retained)
            .contains("forbidden-historical-planner")
    );
    assert!(
        fs::read_to_string(evidence.join("valid/resolved-models.tsv"))
            .unwrap()
            .lines()
            .count()
            > 1
    );
    rejected_omission(&source, parsed, directory.path(), &evidence);
}

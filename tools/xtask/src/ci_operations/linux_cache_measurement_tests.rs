use super::*;

const RUNTIME_KEY: &str = "skippy-runtime-linux-v2-x86_64-unknown-linux-gnu-epoch-abc";

fn tree(files: usize) -> Tree {
    Tree {
        files,
        digest: "d".into(),
    }
}

fn observation<'a>(
    key: &'a str,
    edited: bool,
    hit: &'a str,
    restored: &'a str,
    output: &'a Tree,
) -> Observation<'a> {
    Observation {
        key_before: RUNTIME_KEY,
        key,
        edited,
        cache_hit: hit,
        restored_key: restored,
        output,
        stale_roots: &[],
    }
}

#[test]
fn cohorts_reuse_the_production_inventories_and_admitted_edits() {
    for (name, inventory, edit) in [
        (
            "cpu-runtime",
            Inventory::LinuxRuntime,
            "skippy/crates/skippy-cli/src/main.rs",
        ),
        (
            "hosted-xtask",
            Inventory::HostedXtask,
            "tools/xtask/tests/integration.rs",
        ),
        (
            "hosted-controller",
            Inventory::HostedController,
            "tools/xtask/src/main.rs",
        ),
    ] {
        let cohort = Cohort::parse(name).unwrap();
        assert_eq!(cohort.inventory(), inventory);
        assert_eq!(cohort.unrelated_edit().unwrap(), edit);
        assert!(PRESERVED_EDITS.contains(&(inventory, edit)));
        assert!(
            !inventory
                .inputs()
                .iter()
                .any(|pattern| glob::Pattern::new(pattern).unwrap().matches(edit)),
            "{edit} must stay outside {name}"
        );
    }
    assert!(Cohort::parse("windows-abi").is_err());
    assert!(Condition::parse("warm").is_err());
}

#[test]
fn key_admits_only_a_complete_expanded_cohort_identity() {
    let receipts = tempfile::tempdir().unwrap();
    let directory = receipts.path();
    for value in [
        "",
        "skippy-runtime-linux-v2-",
        "xtask-hosted-bare-v2-Linux-X64-a-b-c",
        "skippy-runtime-linux-v2-x86_64--abc",
        "skippy-runtime-linux-v2-${{ hashFiles('x') }}",
        "skippy-runtime-linux-v2-a b",
    ] {
        assert!(
            key(directory, Cohort::CpuRuntime, "before", value).is_err(),
            "{value}"
        );
    }
    assert!(key(directory, Cohort::CpuRuntime, "after", RUNTIME_KEY).is_err());
    key(directory, Cohort::CpuRuntime, "before", RUNTIME_KEY).unwrap();
    assert!(key(directory, Cohort::CpuRuntime, "before", RUNTIME_KEY).is_err());
}

#[test]
fn cold_requires_no_retrieval_and_an_absent_output() {
    let (empty, full) = (tree(0), tree(2));
    let cold = |hit, restored, output| {
        decide(
            Condition::Cold,
            &observation(RUNTIME_KEY, false, hit, restored, output),
        )
    };
    assert_eq!(cold("", "", &empty), Verdict::Admitted);
    for verdict in [
        cold("false", "", &empty),
        cold("", RUNTIME_KEY, &empty),
        cold("", "", &full),
    ] {
        assert!(matches!(verdict, Verdict::Refused(_)), "{verdict:?}");
    }
}

#[test]
fn warm_arms_are_blocked_without_an_exact_entry_and_never_relabelled_cold() {
    let (empty, full) = (tree(0), tree(2));
    for (condition, edited) in [
        (Condition::ExactWarm, false),
        (Condition::UnrelatedEdit, true),
    ] {
        let warm = |hit, restored, output| {
            decide(
                condition,
                &observation(RUNTIME_KEY, edited, hit, restored, output),
            )
        };
        assert_eq!(warm("true", RUNTIME_KEY, &full), Verdict::Admitted);
        for verdict in [warm("false", "", &empty), warm("", "", &empty)] {
            assert!(matches!(verdict, Verdict::Blocked(_)), "{verdict:?}");
        }
        for verdict in [
            warm("true", "other", &full),
            warm("true", RUNTIME_KEY, &empty),
        ] {
            assert!(matches!(verdict, Verdict::Refused(_)), "{verdict:?}");
        }
    }
}

#[test]
fn unrelated_edit_must_preserve_the_key_and_only_that_arm_edits() {
    let full = tree(2);
    let moved = decide(
        Condition::UnrelatedEdit,
        &observation("skippy-runtime-linux-v2-x-moved", true, "true", "", &full),
    );
    assert!(matches!(moved, Verdict::Refused(reason) if reason.contains("moved the cache key")));
    for (condition, edited) in [
        (Condition::UnrelatedEdit, false),
        (Condition::ExactWarm, true),
        (Condition::Cold, true),
    ] {
        let verdict = decide(
            condition,
            &observation(RUNTIME_KEY, edited, "true", RUNTIME_KEY, &full),
        );
        assert!(matches!(verdict, Verdict::Refused(_)), "{condition:?}");
    }
    let stale = [".deps/llama.cpp/build-stage-abi-dynamic-cpu".to_owned()];
    let reused = Observation {
        stale_roots: &stale,
        ..observation(RUNTIME_KEY, false, "true", RUNTIME_KEY, &full)
    };
    assert!(matches!(
        decide(Condition::ExactWarm, &reused),
        Verdict::Refused(reason) if reason.contains("fresh root")
    ));
}

#[test]
fn edit_appends_one_comment_to_the_admitted_path_once() {
    let checkout = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let source = checkout.path().join("tools/xtask/src/main.rs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    fs::write(&source, "fn main() {}\n").unwrap();
    edit(checkout.path(), receipts.path(), Cohort::HostedController).unwrap();
    assert_eq!(
        fs::read_to_string(&source).unwrap(),
        format!("fn main() {{}}\n{EDIT_MARKER}")
    );
    assert!(edit(checkout.path(), receipts.path(), Cohort::HostedController).is_err());
    assert_eq!(
        fs::read_to_string(receipts.path().join("edit.txt")).unwrap(),
        "tools/xtask/src/main.rs"
    );
}

fn sample_env(condition: &'static str, hit: &'static str) -> impl Fn(&str) -> String {
    move |name: &str| {
        match name {
            "GITHUB_EVENT_NAME" => "workflow_dispatch",
            "MEASUREMENT_CONDITION" => condition,
            "MEASUREMENT_SAMPLE" => "1",
            "LLAMA_STAGE_BUILD_DIR" => ".deps/llama.cpp/build-stage-abi-dynamic-cpu",
            "CACHE_HIT" => hit,
            "CACHE_PRIMARY_KEY" if hit == "true" => RUNTIME_KEY,
            "PREPARE_OUTCOME" => "success",
            _ => "",
        }
        .to_owned()
    }
}

fn record_keys(directory: &Path) {
    for phase in ["before", "measured"] {
        key(directory, Cohort::CpuRuntime, phase, RUNTIME_KEY).unwrap();
    }
}

fn mark_all(directory: &Path) {
    for (offset, phase) in PHASES.iter().enumerate() {
        let at = format!("2026-11-18T10:00:{:02}Z", offset * 5);
        mark(directory, phase, timestamp(&at).ok().flatten().unwrap()).unwrap();
    }
}

fn json_file(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
fn exact_warm_receipt_binds_key_phases_and_an_unchanged_output() {
    let checkout = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let output = checkout.path().join("runtime-input/cpu");
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join("manifest.json"), "{}").unwrap();
    let env = sample_env("exact-warm", "true");
    record_keys(receipts.path());
    mark_all(receipts.path());
    admit(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap();
    finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap();
    let receipt = json_file(&receipts.path().join("receipt.json"));
    assert_eq!(receipt["admitted"], true);
    assert_eq!(receipt["admission"]["key"], RUNTIME_KEY);
    assert_eq!(receipt["admission"]["restored_key"], RUNTIME_KEY);
    assert_eq!(receipt["seconds"]["lookup"], 5.0);
    assert_eq!(receipt["seconds"]["primary"], 15.0);
    assert_eq!(receipt["run"]["GITHUB_EVENT_NAME"], "workflow_dispatch");

    fs::write(output.join("manifest.json"), "{\"rebuilt\":true}").unwrap();
    let error = finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap_err();
    assert!(error.contains("rebuild"), "{error}");
    assert_eq!(
        json_file(&receipts.path().join("receipt.json"))["admitted"],
        false
    );
    assert!(finish(checkout.path(), receipts.path(), Cohort::HostedXtask, &env).is_err());
}

#[test]
fn missing_trusted_entry_records_blocked_evidence_and_no_receipt() {
    let checkout = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let env = sample_env("exact-warm", "false");
    record_keys(receipts.path());
    mark_all(receipts.path());
    let error = admit(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap_err();
    assert!(error.starts_with("BLOCKED: "), "{error}");
    let blocked = json_file(&receipts.path().join("blocked.json"));
    assert_eq!(blocked["outcome"], "blocked");
    assert_eq!(blocked["key"], RUNTIME_KEY);
    assert!(finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).is_err());
    assert!(!receipts.path().join("receipt.json").exists());
}

#[test]
fn cold_receipt_requires_dispatch_and_a_produced_output() {
    let checkout = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let env = sample_env("cold", "");
    record_keys(receipts.path());
    mark_all(receipts.path());
    let pushed = |name: &str| {
        if name == "GITHUB_EVENT_NAME" {
            "push".to_owned()
        } else {
            env(name)
        }
    };
    assert!(admit(checkout.path(), receipts.path(), Cohort::CpuRuntime, pushed).is_err());
    admit(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap();
    assert!(finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).is_err());
    fs::create_dir_all(checkout.path().join("runtime-input/cpu")).unwrap();
    fs::write(
        checkout.path().join("runtime-input/cpu/manifest.json"),
        "{}",
    )
    .unwrap();
    finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap();
}

#[test]
fn failed_preparation_and_bad_phases_write_a_refused_receipt() {
    let checkout = tempfile::tempdir().unwrap();
    let receipts = tempfile::tempdir().unwrap();
    let env = sample_env("cold", "");
    record_keys(receipts.path());
    mark_all(receipts.path());
    admit(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap();
    fs::create_dir_all(checkout.path().join("runtime-input/cpu")).unwrap();
    fs::write(
        checkout.path().join("runtime-input/cpu/manifest.json"),
        "{}",
    )
    .unwrap();
    let failed = |name: &str| {
        if name == "PREPARE_OUTCOME" {
            "failure".to_owned()
        } else {
            env(name)
        }
    };
    let error = finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, failed).unwrap_err();
    assert!(error.contains("did not succeed"), "{error}");
    let receipt = json_file(&receipts.path().join("receipt.json"));
    assert_eq!(receipt["admitted"], false);
    assert_eq!(receipt["prepare_outcome"], "failure");
    assert_eq!(receipt["seconds"]["primary"], 15.0);

    fs::remove_file(receipts.path().join("phase-prepare_end.txt")).unwrap();
    let error = finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap_err();
    assert!(error.contains("prepare_end is missing"), "{error}");
    assert!(json_file(&receipts.path().join("receipt.json"))["phases"]["prepare_end"].is_null());

    let early = timestamp("2026-11-18T09:59:00Z").ok().flatten().unwrap();
    mark(receipts.path(), "prepare_end", early).unwrap();
    let error = finish(checkout.path(), receipts.path(), Cohort::CpuRuntime, &env).unwrap_err();
    assert!(error.contains("out of order"), "{error}");
}

#[test]
fn snapshot_observes_rewritten_identical_bytes_and_single_files() {
    let directory = tempfile::tempdir().unwrap();
    let binary = directory.path().join("xtask");
    assert_eq!(snapshot(&binary).unwrap().files, 0);
    fs::write(&binary, "elf").unwrap();
    let before = snapshot(&binary).unwrap();
    assert_eq!(before.files, 1);
    fs::File::options()
        .write(true)
        .open(&binary)
        .unwrap()
        .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1))
        .unwrap();
    assert_ne!(before, snapshot(&binary).unwrap());
}

#[test]
fn phases_are_closed_and_recorded_once() {
    let receipts = tempfile::tempdir().unwrap();
    let now = Instant::now();
    assert!(mark(receipts.path(), "queue", now).is_err());
    mark(receipts.path(), "lookup_start", now).unwrap();
    assert!(mark(receipts.path(), "lookup_start", now).is_err());
}

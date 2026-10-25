use super::*;
const CHECK: &str = "  - if: ${{ matrix.check.kind == 'skippy-probes' }}\n    run: |\n      $ctest = 'C:\\Program Files\\CMake\\bin\\ctest.exe'\n      if ($machine -ne 0x8664) { throw 'not x64' }\n";
const HEADROOM: &str = "  - if: ${{ matrix.check.kind == 'skippy-probes' }}\n    run: |\n      $os.TotalVirtualMemorySize - $os.FreeVirtualMemory\n";
const BUILD_STEP: &str = "  - if: ${{ matrix.check.kind == 'skippy-probes' }}\n    env:\n      LLAMA_STAGE_BUILD_TESTS: \"ON\"\n      LLAMA_STAGE_BUILD_DIR: .deps/probes\n      CTEST_PARALLEL_LEVEL: ${{ matrix.check.ctest_parallel }}\n    run: |\n      PATH=\"/c/Program Files/CMake/bin:$PATH\" scripts/build-llama.sh\n";
const LIB_STEP: &str = "  - if: ${{ matrix.check.kind == 'skippy-probes' }}\n    env:\n      LLAMA_STAGE_LINK_MODE: static\n      LLAMA_STAGE_BUILD_DIR: .deps/probes\n    run: |\n      cargo test --locked -p skippy-runtime --lib\n      cargo test --locked -p skippy-serving --lib\n";
fn job(timeout: &str, steps: &[&str]) -> Node {
    workflow_yaml::parse(&format!(
        "timeout-minutes: {timeout}\nsteps:\n{}",
        steps.concat()
    ))
    .unwrap()
}
fn row(edit: impl FnOnce(&mut Value)) -> Value {
    let mut row = serde_json::json!({"id": ROW, "platform": "windows", "kind": "skippy-probes", "ctest_parallel": 1, "timeout_minutes": 150});
    edit(&mut row);
    serde_json::json!({"platform_rows": [row]})
}
#[test]
fn windows_skippy_probes_actual_workflow_and_catalog_hold() {
    check(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
}
#[test]
fn windows_skippy_probes_job_checks_ctest_and_headroom_before_the_build() {
    probe_job(&job(JOB_TIMEOUT, &[CHECK, HEADROOM, BUILD_STEP, LIB_STEP])).unwrap();
    for steps in [
        &[BUILD_STEP, CHECK, HEADROOM, LIB_STEP][..],
        &[CHECK, BUILD_STEP, HEADROOM][..],
        &[HEADROOM, BUILD_STEP][..],
        &[CHECK, BUILD_STEP][..],
        &[CHECK, HEADROOM][..],
    ] {
        assert!(probe_job(&job(JOB_TIMEOUT, steps)).is_err(), "{steps:?}");
    }
    let other_kind = CHECK.replace("skippy-probes", "unit");
    assert!(probe_job(&job(JOB_TIMEOUT, &[&other_kind, HEADROOM, BUILD_STEP])).is_err());
}
#[test]
fn windows_skippy_probes_build_uses_x64_cmake_tests_and_row_parallelism() {
    for (from, to) in [
        ("0x8664", "0x014C"),
        ("PATH=\"/c/Program Files/CMake/bin:$PATH\" ", ""),
        ("\"ON\"", "\"OFF\""),
        ("${{ matrix.check.ctest_parallel }}", "4"),
    ] {
        let check = CHECK.replace(from, to);
        let build = BUILD_STEP.replace(from, to);
        assert!(check != CHECK || build != BUILD_STEP, "{from}");
        assert!(
            probe_job(&job(JOB_TIMEOUT, &[&check, HEADROOM, &build, LIB_STEP])).is_err(),
            "{from}"
        );
    }
    assert!(probe_job(&job("75", &[CHECK, HEADROOM, BUILD_STEP, LIB_STEP])).is_err());
}
#[test]
fn windows_skippy_probes_lib_tests_follow_the_build_on_its_static_abi() {
    for steps in [
        &[CHECK, HEADROOM, BUILD_STEP][..],
        &[CHECK, HEADROOM, LIB_STEP, BUILD_STEP][..],
    ] {
        assert!(probe_job(&job(JOB_TIMEOUT, steps)).is_err(), "{steps:?}");
    }
    for (from, to) in [
        ("-p skippy-serving", "-p skippy-server"),
        ("static", "dynamic"),
        (
            "LLAMA_STAGE_BUILD_DIR: .deps/probes\n    run",
            "LLAMA_STAGE_BUILD_DIR: .deps/other\n    run",
        ),
        ("skippy-probes", "unit"),
        (
            "      cargo test --locked -p skippy-runtime",
            "      # cargo test --locked -p skippy-runtime",
        ),
        (
            "      cargo test --locked -p skippy-serving",
            "      # cargo test --locked -p skippy-serving",
        ),
    ] {
        let lib = LIB_STEP.replace(from, to);
        assert_ne!(lib, LIB_STEP, "{from}");
        assert!(
            probe_job(&job(JOB_TIMEOUT, &[CHECK, HEADROOM, BUILD_STEP, &lib])).is_err(),
            "{from}"
        );
    }
}
#[test]
fn windows_skippy_probes_row_is_windows_and_extends_the_timeout() {
    probe_row(&row(|_| {})).unwrap();
    let edits: [fn(&mut Value); 6] = [
        |row| row["id"] = "windows-unit".into(),
        |row| row["platform"] = "macos".into(),
        |row| row["kind"] = "unit".into(),
        |row| row["ctest_parallel"] = 0.into(),
        |row| {
            row.as_object_mut().unwrap().remove("ctest_parallel");
        },
        |row| row["timeout_minutes"] = DEFAULT_TIMEOUT.into(),
    ];
    for edit in edits {
        assert!(probe_row(&row(edit)).is_err());
    }
}

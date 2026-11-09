use super::fixture::Fixture;
use std::fs;

const COMMON: &[&str] = &[
    "run_for",
    "remaining_verification_seconds",
    "run_verification_logged",
];

#[test]
fn actual_full_build_delegates_both_backends_to_the_shared_typed_plan() {
    for (backend, mode) in [("metal", "verify-build"), ("cuda", "native-build")] {
        let fixture = Fixture::new();
        fixture.build_tools();
        let mut names = COMMON.to_vec();
        names.push("run_full_build");
        let report = fixture.run(
            mode,
            &names,
            &format!("LLAMA_STAGE_BACKEND={backend}"),
            "run_full_build",
            &[(
                "MESH_LLM_AUTOMATION_BIN",
                fixture.root.join("tools/controller").display().to_string(),
            )],
        );
        assert_eq!(
            report.process.status.unwrap().code(),
            Some(0),
            "{}",
            fixture.diagnostic(&report)
        );
        assert_eq!(fixture.order(), ["repair-check", "cargo"]);
        let args = fixture.args("repair-check");
        for pair in [["--phase", "full"], ["--backend", backend]] {
            assert!(args.windows(2).any(|arguments| arguments == pair));
        }
        assert!(args.windows(2).any(|arguments| arguments[0] == "--root"
            && arguments[1] == fixture.root.display().to_string()));
        assert!(fixture.root.join("mm-build.jsonl").is_file());
        fixture.finish();
    }
}

#[test]
fn actual_prepare_admits_relative_script_in_all_five_modes_and_refuses_wrong_stamp() {
    for mode in [
        "repair",
        "verify",
        "repair-build",
        "verify-build",
        "pinned-build",
    ] {
        for wrong in [false, true] {
            let fixture = Fixture::new();
            fixture.tool(
                "scripts/update-llama-pin.sh",
                "update",
                r#"printf '%s\n' "$1" > "$INERT_ROOT/skippy/llama_cpp/upstream.txt""#,
            );
            fixture.tool("scripts/prepare-llama.sh", "prepare", r#"printf '%s\n' "$INERT_PREPARED_SHA" > "$INERT_ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha""#);
            let mut names = COMMON.to_vec();
            names.extend(["write_repair_pin", "verify_repair_pin", "run_prepare"]);
            let report = fixture.run(
                mode,
                &names,
                "",
                "run_prepare",
                &[(
                    "INERT_PREPARED_SHA",
                    if wrong { "b" } else { "a" }.repeat(40),
                )],
            );
            assert_eq!(
                report.process.status.unwrap().code(),
                Some(i32::from(wrong)),
                "mode={mode}, wrong={wrong}: {}",
                fixture.diagnostic(&report)
            );
            assert_eq!(fixture.args("prepare"), ["pinned"]);
            if mode == "pinned-build" {
                assert_eq!(fixture.order(), ["prepare"]);
                assert!(!fixture.root.join("update.args").exists());
            } else {
                assert_eq!(fixture.order(), ["update", "prepare"]);
                assert_eq!(fixture.args("update"), ["a".repeat(40)]);
            }
            assert_eq!(
                fs::read_to_string(fixture.root.join("skippy/llama_cpp/upstream.txt")).unwrap(),
                format!("{}\n", "a".repeat(40))
            );
            if wrong {
                assert!(
                    fs::read_to_string(fixture.root.join("prepare.log"))
                        .unwrap()
                        .contains("prepared upstream is")
                );
            }
            fixture.finish();
        }
        // The same actual timeout handoff preserves absolute argv and refuses a shell builtin.
        let fixture = Fixture::new();
        let report = fixture.run(mode, &["run_for"], "", r#"
run_for 'absolute argument custody' 3 /bin/bash -c 'printf "%s\0" "$@" > absolute.args' child '' 'space value' '*.txt'
if run_for 'builtin refusal' 3 printf forbidden; then exit 99; else status=$?; fi
[[ "$status" == 125 ]]
"#, &[]);
        assert_eq!(
            report.process.status.unwrap().code(),
            Some(0),
            "{}",
            fixture.diagnostic(&report)
        );
        assert_eq!(
            fixture.values("absolute.args"),
            ["", "space value", "*.txt"]
        );
        assert!(
            report
                .stderr
                .as_ref()
                .unwrap()
                .as_bytes()
                .windows(b"requires an absolute executable".len())
                .any(|part| part == b"requires an absolute executable")
        );
        fixture.finish();
    }
    let fixture = Fixture::new();
    let pin = fixture.root.join("skippy/llama_cpp/upstream.txt");
    fs::write(&pin, format!("{}\n", "b".repeat(40))).unwrap();
    let mut names = COMMON.to_vec();
    names.extend(["write_repair_pin", "verify_repair_pin", "run_prepare"]);
    let report = fixture.run("pinned-build", &names, "", "run_prepare", &[]);
    assert_eq!(report.process.status.unwrap().code(), Some(1));
    assert!(!fixture.root.join("order").exists());
    assert_eq!(
        fs::read_to_string(pin).unwrap(),
        format!("{}\n", "b".repeat(40))
    );
    fixture.finish();
}

#[test]
fn actual_full_build_preserves_shared_check_failure_without_starting_later_gates() {
    let fixture = Fixture::new();
    fixture.build_tools();
    let mut names = COMMON.to_vec();
    names.push("run_full_build");
    let report = fixture.run(
        "verify-build",
        &names,
        "",
        "run_full_build",
        &[
            (
                "MESH_LLM_AUTOMATION_BIN",
                fixture.root.join("tools/controller").display().to_string(),
            ),
            ("INERT_FAIL", "repair-check".into()),
        ],
    );
    assert_eq!(report.process.status.unwrap().code(), Some(17));
    assert_eq!(fixture.order(), ["repair-check"]);
    assert!(!fixture.root.join("mm-build.jsonl").exists());
    fixture.finish();
}

#[test]
fn actual_full_build_uses_controller_jinja_project_for_historical_source_without_manifest() {
    let fixture = Fixture::new();
    fixture.build_tools();
    let controller_manifest = fixture.root.join("ci/llama-canary/jinja/pyproject.toml");
    fs::create_dir_all(controller_manifest.parent().unwrap()).unwrap();
    fs::write(
        &controller_manifest,
        "[project]\nname = \"controller-jinja\"\n",
    )
    .unwrap();
    let historical_source = fixture.root.join("historical-source");
    fs::create_dir_all(&historical_source).unwrap();
    assert!(
        !historical_source
            .join("ci/llama-canary/jinja/pyproject.toml")
            .exists()
    );

    let source = super::fixture::source();
    let project_binding = source
        .find("JINJA_PROJECT=\"$TRUSTED_ROOT/ci/llama-canary/jinja\"")
        .unwrap();
    let selected_root_switch = source.find("ROOT=\"${CANARY_SOURCE_ROOT:").unwrap();
    assert!(project_binding < selected_root_switch);

    let mut names = COMMON.to_vec();
    names.push("run_full_build");
    let report = fixture.run(
        "pinned-build",
        &names,
        "ROOT=\"$ROOT/historical-source\"",
        "run_full_build",
        &[(
            "MESH_LLM_AUTOMATION_BIN",
            fixture.root.join("tools/controller").display().to_string(),
        )],
    );
    assert_eq!(
        report.process.status.unwrap().code(),
        Some(0),
        "{}",
        fixture.diagnostic(&report)
    );

    let args = fixture.args("repair-check");
    assert!(controller_manifest.is_file());
    assert!(args.windows(2).any(
        |pair| pair[0] == "--controller-root" && pair[1] == fixture.root.display().to_string()
    ));
    assert!(
        args.windows(2)
            .any(|pair| pair[0] == "--root" && pair[1] == historical_source.display().to_string())
    );
    fixture.finish();
}

mod fixture;
mod models;
use fixture::Fixture;
use std::fs;
#[test]
fn actual_copied_gate_executes_with_complete_argv_and_absolute_writer_reader_paths() {
    for relative in [false, true] {
        let fixture = Fixture::new("executed");
        let report = fixture.run(fixture.args(relative));
        assert!(report.process.success(), "{:?}", report.process);
        let environment = fixture.captured("environment");
        assert_eq!(
            environment[0],
            fixture.directory.join("source").to_str().unwrap()
        );
        assert_eq!(environment[1], "1");
        assert_eq!(
            environment[2],
            fixture.caller.join("bundle directory").to_str().unwrap()
        );
        assert_eq!(
            environment[3],
            fixture.caller.join("model file.gguf").to_str().unwrap()
        );
        assert_eq!(environment[4], fixture.evidence().to_str().unwrap());
        let args = fixture.captured("argv");
        assert_eq!(args[0], "test");
        assert!(args.iter().any(|value| value == "--locked"));
        for pair in [
            ["-p", "skippy-runtime"],
            ["--features", "dynamic-native-runtime"],
            ["--test", "runtime_events_native"],
        ] {
            assert!(args.windows(2).any(|values| values == pair));
        }
        assert_eq!(
            fs::read_to_string(fixture.evidence()).unwrap(),
            "executed\n"
        );
        assert!(String::from_utf8_lossy(report.stdout.unwrap().as_bytes()).contains("executed"));
        assert!(!fixture.directory.join("capture/nested evidence").exists());
    }
}
#[test]
fn green_child_without_execution_and_stale_success_markers_cannot_pass() {
    for mode in ["blocked", "absent"] {
        let fixture = Fixture::new(mode);
        fs::create_dir_all(fixture.evidence().parent().unwrap()).unwrap();
        fs::write(fixture.evidence(), "executed\nfrom earlier run\n").unwrap();
        let report = fixture.run(fixture.args(true));
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        let evidence = fs::read_to_string(fixture.evidence()).unwrap();
        assert!(!evidence.contains("executed") && !evidence.contains("earlier"));
        assert!(!fixture.captured("argv").is_empty());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("did not execute")
        );
    }
}
#[test]
fn child_failure_remains_failure_without_an_executed_marker() {
    let fixture = Fixture::new("failure");
    let report = fixture.run(fixture.args(false));
    assert_eq!(report.process.status.unwrap().code(), Some(7));
    assert!(fs::read(fixture.evidence()).unwrap().is_empty());
}
#[test]
fn missing_or_empty_native_inputs_reject_before_any_cargo_or_evidence_write() {
    for fault in ["bundle", "model", "empty-model", "arguments"] {
        let fixture = Fixture::new("executed");
        let mut args = fixture.args(true);
        match fault {
            "bundle" => args[1] = "missing bundle".into(),
            "model" => args[3] = "missing.gguf".into(),
            "empty-model" => {
                fs::write(fixture.caller.join("model file.gguf"), b"").unwrap();
            }
            _ => args.clear(),
        }
        let report = fixture.run(args);
        assert_ne!(report.process.status.unwrap().code(), Some(0));
        assert!(!fixture.directory.join("capture/argv").exists());
        assert!(!fixture.evidence().exists());
    }
}

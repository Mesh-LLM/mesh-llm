use super::write_with;
use serde_json::Value;
use std::fs;

#[test]
fn disk_receipt_preserves_both_labels_existing_ancestors_and_equality_boundary() {
    let state = tempfile::tempdir().unwrap();
    let models = state.path().join("models");
    fs::create_dir(&models).unwrap();
    let output = state.path().join("environment.json");
    for (model_free, success) in [(1 << 30, true), ((1 << 30) - 1, false)] {
        let mut paths = Vec::new();
        let result = write_with(
            &state.path().join("absent/artifacts"),
            &models.join("absent/model"),
            "1",
            &output,
            |path| {
                paths.push(path.to_owned());
                Ok(if path == models { model_free } else { 1 << 30 })
            },
        );
        assert_eq!(result.is_ok(), success);
        assert_eq!(paths, [state.path().to_owned(), models.clone()]);
        let bytes = fs::read(&output).unwrap();
        assert!(bytes.ends_with(b"\n"));
        let report: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report["ports"]["allocation"], "os-assigned-at-launch");
        assert_eq!(report["filesystems"].as_array().unwrap().len(), 2);
        assert_eq!(report["filesystems"][0]["label"], "artifacts");
        assert_eq!(report["filesystems"][0]["sufficient"], true);
        assert_eq!(report["filesystems"][1]["label"], "models");
        assert_eq!(report["filesystems"][1]["free_bytes"], model_free);
        assert_eq!(report["filesystems"][1]["minimum_free_bytes"], 1_u64 << 30);
        assert_eq!(report["filesystems"][1]["sufficient"], success);
    }
}

#[test]
fn malformed_overflowing_or_failed_probe_preserves_existing_receipt() {
    let state = tempfile::tempdir().unwrap();
    let output = state.path().join("environment.json");
    fs::write(&output, b"unchanged").unwrap();
    for minimum in [
        "",
        "+1",
        "-1",
        "1.5",
        " 1",
        "18446744073709551616",
        "17179869184",
    ] {
        assert!(
            write_with(state.path(), state.path(), minimum, &output, |_| panic!(
                "invalid minimum must fail before probe"
            ))
            .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"unchanged");
    }
    let mut calls = 0;
    assert!(
        write_with(state.path(), state.path(), "0", &output, |_| {
            calls += 1;
            if calls == 1 {
                Ok(0)
            } else {
                Err("probe failed".into())
            }
        })
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), b"unchanged");
    write_with(state.path(), state.path(), "0", &output, |_| Ok(0)).unwrap();
}

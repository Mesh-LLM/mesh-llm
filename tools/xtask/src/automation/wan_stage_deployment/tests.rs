use super::*;
fn controls(index: u32) -> Controls {
    Controls {
        index,
        count: 4,
        port: 19000,
        run_id: "run".into(),
        topology_id: "topology".into(),
        batch: Some(1024),
        ubatch: None,
        cache_k: "f16".into(),
        cache_v: "q8_0".into(),
        flash: "disabled".into(),
    }
}
fn admitted(index: u32) -> Value {
    json!({"stage_id":format!("stage-{index}"),"stage_index":index,
        "source_model_sha256":"source-sha", "source_model_bytes":1234,
        "layer_start":12,"layer_end":24,"activation_codec":"bf16-rne-v1",
        "model_part_paths":["model.gguf"], "resident_tensor_names":["blk.12.attn_q.weight"],
        "execution_contract":{"frontiers":[{"id":"exact"}]},
        "activation_import_identities":["boundary-12"],"activation_export_identities":["boundary-24"],
        "future_source_field":{"nested":[1,false,null,"kept"]}})
}
#[test]
fn deployment_changes_only_ten_allowed_fields_and_preserves_admitted_contracts() {
    for index in 0..4 {
        let original = admitted(index);
        let projected = project(original.clone(), &controls(index)).unwrap();
        for (key, value) in original.as_object().unwrap() {
            assert_eq!(projected.get(key), Some(value), "{key}");
        }
        assert_eq!(projected["bind_addr"], "0.0.0.0:19000");
        assert_eq!(projected["n_batch"], 1024);
        assert!(projected["n_ubatch"].is_null());
        assert_eq!(
            projected.as_object().unwrap().len(),
            original.as_object().unwrap().len() + 10
        );
        assert_eq!(
            projected["upstream"],
            index.checked_sub(1).map_or(Value::Null, |i| peer(i, 19000))
        );
        assert_eq!(
            projected["downstream"],
            if index == 3 {
                Value::Null
            } else {
                peer(index + 1, 19000)
            }
        );
    }
}
#[test]
fn wrong_stage_or_nonobject_is_refused_without_rebuilding_admission() {
    for value in [
        admitted(2),
        json!({"stage_id":"stage-1"}),
        json!({"stage_id":"other","stage_index":1}),
        json!([]),
    ] {
        assert!(project(value, &controls(1)).is_err());
    }
}
#[test]
fn closed_typed_controls_reject_malformed_values_and_duplicate_flags() {
    let base = [
        "--input",
        "input",
        "--output",
        "output",
        "--stage-index",
        "1",
        "--stage-count",
        "4",
    ];
    for extra in [
        vec!["--bind-port", "0"],
        vec!["--stage-count", "1"],
        vec!["--stage-index", "4"],
        vec!["--n-batch", "-1"],
        vec!["--n-ubatch", "4294967296"],
        vec!["--flash-attn-type", "maybe"],
        vec!["--cache-type-k", "\n"],
        vec!["--unknown", "x"],
    ] {
        let args = base
            .iter()
            .chain(extra.iter())
            .map(|s| (*s).into())
            .collect::<Vec<_>>();
        assert!(parse(&args).is_err());
    }
    let args = base.iter().map(|s| (*s).into()).collect::<Vec<_>>();
    assert!(parse(&args).is_ok());
}
#[test]
fn generated_output_is_atomically_replaced_and_unavailable_parent_refused() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("output");
    publish(&output, b"first\n").unwrap();
    publish(&output, b"replacement\n").unwrap();
    assert_eq!(fs::read(&output).unwrap(), b"replacement\n");
    assert!(publish(&temp.path().join("missing/output"), b"new\n").is_err());
    let directory = temp.path().join("directory");
    fs::create_dir(&directory).unwrap();
    assert!(publish(&directory, b"new\n").is_err());
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
}
#[cfg(unix)]
#[test]
fn generated_output_symlink_refusal_preserves_target_and_mode_survives_replacement() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target");
    fs::write(&target, b"original\n").unwrap();
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(publish(&link, b"replacement\n").is_err());
    assert_eq!(fs::read(&target).unwrap(), b"original\n");
    fs::set_permissions(&target, fs::Permissions::from_mode(0o640)).unwrap();
    publish(&target, b"replacement\n").unwrap();
    assert_eq!(
        fs::metadata(&target).unwrap().permissions().mode() & 0o777,
        0o640
    );
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
}

#[test]
fn bounded_input_refuses_directory_oversize_and_malformed_json() {
    let temp = tempfile::tempdir().unwrap();
    assert!(read_plan(temp.path()).is_err());
    let input = temp.path().join("input");
    fs::write(&input, b"{").unwrap();
    assert!(read_plan(&input).is_err());
    fs::File::create(&input)
        .unwrap()
        .set_len(INPUT_LIMIT + 1)
        .unwrap();
    assert!(read_plan(&input).is_err());
}
#[cfg(unix)]
#[test]
fn input_symlink_refused() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::write(&input, b"{}").unwrap();
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&input, &link).unwrap();
    assert!(read_plan(&link).is_err());
}

#[test]
fn stage_bounds_are_checked_without_duplicate_flag_masking() {
    for (index, count) in [
        ("4", "4"),
        ("0", "1"),
        ("0", "10001"),
        ("-1", "4"),
        ("true", "4"),
    ] {
        let args = [
            "--input",
            "input",
            "--output",
            "output",
            "--stage-index",
            index,
            "--stage-count",
            count,
        ]
        .map(String::from);
        assert!(parse(&args).is_err());
    }
    let args = [
        "--input",
        "input",
        "--output",
        "output",
        "--stage-index",
        "9999",
        "--stage-count",
        "10000",
    ]
    .map(String::from);
    assert!(parse(&args).is_ok());
}

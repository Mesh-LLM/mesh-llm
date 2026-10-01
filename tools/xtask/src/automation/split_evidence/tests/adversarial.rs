use super::{execute, mutate, request};
use crate::automation::{
    codepoint_json::{parser, value::Value},
    split_evidence::{args::Mode, types::Integer, verify},
};
use serde_json::json;
use std::fs;

#[test]
fn rejects_field_types_indexes_ranges_and_model_duplicates() {
    for case in 0..12 {
        let root = tempfile::tempdir().unwrap();
        let request = request(root.path());
        match case {
            0 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["stage_index"] = json!(true)
            }),
            1 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["layer_end"] = json!(-1)
            }),
            2 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["layer_end"] = json!(12.0)
            }),
            3 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][1]["stage_index"] = json!(0)
            }),
            4 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["layer_start"] = json!(1)
            }),
            5 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["layer_end"] = json!(0)
            }),
            6 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][1]["stage_id"] = json!("stage-0")
            }),
            7 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][1]["node_id"] = json!("seed-node-0001")
            }),
            8 => mutate(&request, 2, |value| {
                value["data"] = json!([{"id":"model-a"},{"id":"model-a"}])
            }),
            9 => mutate(&request, 0, |value| value["peers"] = json!([])),
            10 => mutate(&request, 0, |value| {
                value["peers"][0]["id"] = json!("foreign")
            }),
            11 => mutate(&request, 1, |value| {
                value["topologies"][0]["stages"][0]["endpoint"] = json!(null)
            }),
            _ => unreachable!(),
        }
        assert!(execute(&request).is_err(), "case {case}");
    }
}

#[test]
fn retains_arbitrary_width_integers_and_rejects_boolean() {
    let huge = parser::parse(b"18446744073709551615").unwrap();
    let integer = Integer::parse(Some(&huge), "layer_end").unwrap();
    assert!(matches!(integer.value(), Value::BigInt(ref text) if text == "18446744073709551615"));
    assert!(Integer::parse(Some(&Value::Bool(false)), "index").is_err());
}

#[test]
fn duplicate_last_value_and_unknown_fields_are_accepted() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    fs::write(
        &request.paths[2],
        b"{\"data\":false,\"d\\u0061ta\":[{\"id\":\"model-a\"}],\"ignored\":null}",
    )
    .unwrap();
    assert!(execute(&request).is_ok());
}

#[test]
fn sorted_reordered_stages_and_statuses_are_accepted() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    mutate(&request, 4, |value| {
        value["topologies"][0]["stages"]
            .as_array_mut()
            .unwrap()
            .reverse();
        value["statuses"].as_array_mut().unwrap().reverse();
    });
    assert!(execute(&request).is_ok());
}

#[test]
fn verification_preserves_bytes_when_numeric_replacements_match_python() {
    for (replacement, accepted) in [(json!(true), true), (json!(1.0), true), (json!(2), false)] {
        let root = tempfile::tempdir().unwrap();
        let mut request = request(root.path());
        let path = root.path().join("split-evidence.json");
        let mut evidence: serde_json::Value = serde_json::from_slice(
            &fs::read(super::fixture().join("expected-ready.json")).unwrap(),
        )
        .unwrap();
        evidence["schema_version"] = replacement;
        let original = serde_json::to_vec(&evidence).unwrap();
        fs::write(&path, &original).unwrap();
        request.mode = Mode::Verify(path.clone());
        let result = execute(&request);
        assert_eq!(result.is_ok(), accepted);
        assert_eq!(fs::read(path).unwrap(), original);
    }
    assert!(!verify::equal(
        &parser::parse(b"{\"a\":1}").unwrap(),
        &parser::parse(b"{\"a\":1,\"b\":2}").unwrap()
    ));
}

#[test]
fn invalid_utf8_persists_failure() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    fs::write(&request.paths[5], [0xff, 0x80, 0xff]).unwrap();
    assert!(execute(&request).is_err());
}

#[test]
fn output_alias_reads_all_snapshots_before_replacing() {
    let root = tempfile::tempdir().unwrap();
    let mut request = request(root.path());
    request.mode = Mode::Output(request.paths[5].clone());
    assert!(execute(&request).is_ok());
    let evidence: serde_json::Value =
        serde_json::from_slice(&fs::read(&request.paths[5]).unwrap()).unwrap();
    assert_eq!(
        evidence["snapshots"]["worker_models"]["path"],
        "worker-models.json"
    );
    assert_eq!(
        evidence["snapshots"]["worker_models"]["sha256"],
        "6d56dc8395e8c032a254d0a8f8497452f09549e5c283785cc675deaaed295680"
    );
}

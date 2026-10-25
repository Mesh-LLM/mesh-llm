use super::{execute, mutate, request};
use serde_json::json;
use std::fs;

#[test]
fn arbitrary_width_layer_ranges_are_not_narrowed() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    for index in [1, 4] {
        let text = fs::read_to_string(&request.paths[index]).unwrap();
        fs::write(
            &request.paths[index],
            text.replace(":12", ":340282366920938463463374607431768211456")
                .replace(":24", ":340282366920938463463374607431768211457"),
        )
        .unwrap();
    }
    assert!(execute(&request).is_err());
}

#[test]
fn prefix_ambiguity_fails_even_with_mutual_sole_peers() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    mutate(&request, 0, |value| value["node_id"] = json!("s"));
    mutate(&request, 3, |value| value["peers"][0]["id"] = json!("s"));
    for index in [1, 4] {
        mutate(&request, index, |value| {
            value["topologies"][0]["stages"][1]["node_id"] = json!("second-worker-node");
            value["statuses"][1]["node_id"] = json!("second-worker-node");
        });
    }
    assert!(
        execute(&request)
            .unwrap_err()
            .to_string()
            .contains("seed observer node ID must match exactly one")
    );
}

#[test]
fn whitespace_endpoints_package_labels_and_lone_surrogates_remain_accepted() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    for index in [1, 4] {
        let text = fs::read_to_string(&request.paths[index]).unwrap();
        fs::write(
            &request.paths[index],
            text.replace("hf:test/model@revision", " ")
                .replace("127.0.0.1:5501", "\\ud800"),
        )
        .unwrap();
    }
    assert!(execute(&request).is_err());
}

#[test]
fn byte_json_accepts_utf8_bom_and_utf16_while_hashing_raw_bytes() {
    for encoding in [0, 1] {
        let root = tempfile::tempdir().unwrap();
        let request = request(root.path());
        let text = fs::read_to_string(&request.paths[2]).unwrap();
        let bytes = if encoding == 0 {
            [b"\xef\xbb\xbf".as_slice(), text.as_bytes()].concat()
        } else {
            [
                vec![0xff, 0xfe],
                text.encode_utf16().flat_map(u16::to_le_bytes).collect(),
            ]
            .concat()
        };
        fs::write(&request.paths[2], &bytes).unwrap();
        assert!(execute(&request).is_ok());
        let evidence: serde_json::Value =
            serde_json::from_slice(&fs::read(root.path().join("split-evidence.json")).unwrap())
                .unwrap();
        assert_ne!(
            evidence["snapshots"]["seed_models"]["sha256"],
            "6d56dc8395e8c032a254d0a8f8497452f09549e5c283785cc675deaaed295680"
        );
    }
}

#[test]
fn first_field_error_respects_legacy_validation_order() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());
    mutate(&request, 1, |value| {
        value["topologies"][0]["stages"][0]["endpoint"] = json!(null);
        value["topologies"][0]["stages"][0]["stage_id"] = json!(null);
    });
    assert!(
        execute(&request)
            .unwrap_err()
            .to_string()
            .contains("stages[0].endpoint must be a JSON object")
    );
}

//! The Agent Action Capsule canonical vectors (`vectors/aac-capsule/`, pinned
//! in `vectors/SOURCES.md`): every `canonical-*` case's `capsule_id`, as the
//! sealing crate computes it, must equal the reference value byte for byte.
//! A record id that drifts from the reference would make this node's records
//! unverifiable by every other implementation.

use capsule_emit_lib::jcs::compute_capsule_id;
use serde_json::Value;
use std::path::PathBuf;

fn vectors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vectors/aac-capsule")
}

fn read_json(path: &std::path::Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn canonical_vectors_match_the_reference_capsule_ids() {
    let dir = vectors_dir();
    let manifest = read_json(&dir.join("vectors.json"));
    let names: Vec<&str> = manifest["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .filter_map(|c| c["name"].as_str())
        .filter(|n| n.starts_with("canonical-"))
        .collect();
    assert!(
        names.len() >= 10,
        "expected the canonical cases, found {}",
        names.len()
    );

    for name in names {
        let case = dir.join(name);
        let input = read_json(&case.join("input.json"));
        let expected = read_json(&case.join("expected.json"));
        let want = expected["capsule_id_recomputed"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: expected.json has no capsule_id_recomputed"));
        let got = compute_capsule_id(&input)
            .unwrap_or_else(|e| panic!("{name}: rejected an input the reference digests: {e}"));
        assert_eq!(got, want, "{name}: capsule_id differs from the reference");
    }
}

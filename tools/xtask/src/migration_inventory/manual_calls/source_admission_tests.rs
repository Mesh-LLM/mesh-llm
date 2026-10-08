use super::{manual_source_lines, validate_manual};
use crate::migration_inventory::other_shard::{check_other_shard, source_lines};
use crate::migration_inventory::shard_rows::{OutsideCall, OutsideKind};
use std::collections::BTreeMap;
use std::fs;

const REWRITER_DOC: &str = "skippy/scripts/tools/skippy-stage-rewriter/README.md";
const COMMAND: &str = "uv run --python 3.12 fixed-manual-helper.py";

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join(REWRITER_DOC);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, format!("Reference documentation\n  {COMMAND}  \n")).unwrap();
    root
}

#[test]
fn relocated_manual_tool_document_keeps_exact_source_binding() {
    let root = fixture();
    assert_eq!(
        manual_source_lines(root.path(), REWRITER_DOC).unwrap(),
        ["Reference documentation", COMMAND]
    );
    let error = validate_manual(root.path(), Vec::new(), BTreeMap::new(), Vec::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("1 missing instructions"), "{error}");
    assert!(error.contains(REWRITER_DOC), "{error}");
    let bound = || OutsideCall {
        file: REWRITER_DOC.into(),
        line: 2,
        source_block: COMMAND.into(),
        target: "fixed-manual-helper.py".into(),
        boundary: "Explicit fixture instruction only; no interpreter execution or qualification"
            .into(),
        kind: OutsideKind::Instruction,
    };
    validate_manual(root.path(), vec![bound()], BTreeMap::new(), Vec::new()).unwrap();
    let mut stale = bound();
    stale.source_block = "uv run --python 3.13 changed-helper.py".into();
    let error = validate_manual(root.path(), vec![stale], BTreeMap::new(), Vec::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("stale instruction"), "{error}");
}

#[test]
fn manual_reader_refuses_unlisted_absolute_and_traversal_paths() {
    let root = fixture();
    for path in [
        "/skippy/scripts/tools/skippy-stage-rewriter/README.md",
        "skippy/scripts/tools/skippy-stage-rewriter/../skippy-stage-rewriter/README.md",
        "../skippy/scripts/tools/skippy-stage-rewriter/README.md",
        "skippy/scripts/unlisted/README.md",
        ".github/workflows/unlisted.yml",
    ] {
        let error = manual_source_lines(root.path(), path)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid manual source path"),
            "{path}: {error}"
        );
    }
}

#[test]
fn relocated_manual_document_does_not_expand_other_shard_scope() {
    let root = fixture();
    assert!(source_lines(root.path(), REWRITER_DOC).is_err());
    let ledger = serde_json::json!({
        "schema_version": 1,
        "groups": [{"file": REWRITER_DOC, "members": []}],
        "python_implementation_edges": [],
        "outside_scanner_source_calls": [],
    });
    let error = check_other_shard(root.path(), &ledger.to_string(), &[])
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid source path"), "{error}");
}

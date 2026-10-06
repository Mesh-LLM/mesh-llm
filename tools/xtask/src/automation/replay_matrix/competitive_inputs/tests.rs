use super::{
    projection,
    schema::{Budget, Kind, Request, Source},
};
use crate::process::Cancellation;
use serde_json::Value;
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};
fn source(root: &Path, key: &str) -> Source {
    Source {
        key: key.into(),
        directory: root.canonicalize().unwrap(),
        source_tree_sha256: projection::fixture_tree(root),
        kind: if key == "granite-h1-hybrid" {
            Kind::ImmutableSnapshot
        } else {
            Kind::SuppliedDerivedExport
        },
    }
}
fn request(root: &Path, sources: Vec<Source>) -> Request {
    Request {
        config: root.join("config.json"),
        config_sha256: "a".repeat(64),
        model_keys: vec![],
        sources,
        output_directory: root.join("output"),
        timeout_seconds: 10,
        maximum_source_bytes: 1024 * 1024,
    }
}
fn config() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../evals/skippy-competitive-benchmark.json"
    )))
    .unwrap()
}
#[test]
fn local_family_default_filter_and_derived_source_admission_are_closed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let document = config();
    let sources: Vec<_> = document["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let key = m["key"].as_str().unwrap();
            let dir = root.join(key);
            fs::create_dir(&dir).unwrap();
            fs::write(dir.join("tokenizer.json"), "{}").unwrap();
            source(&dir, key)
        })
        .collect();
    let mut input = request(&root, sources);
    assert_eq!(super::schema::selected(&document, &input).unwrap().len(), 4);
    input.model_keys = vec!["deepseek-v2-moe".into()];
    input.sources.retain(|s| s.key == "deepseek-v2-moe");
    assert_eq!(super::schema::selected(&document, &input).unwrap().len(), 1);
    input.sources[0].kind = Kind::ImmutableSnapshot;
    assert!(super::schema::selected(&document, &input).is_err());
    input.model_keys.push("unknown".into());
    assert!(super::schema::selected(&document, &input).is_err());
    temp.close().unwrap();
}
#[test]
fn granite_full_snapshot_flattens_keeps_extra_files_and_ignores_all_readme_basenames() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let snapshot = root.join("snapshot");
    let expected = root.join("expected");
    fs::create_dir(&snapshot).unwrap();
    fs::create_dir(snapshot.join("nested")).unwrap();
    fs::create_dir(&expected).unwrap();
    for (name, bytes) in [
        ("tokenizer.json", "{}"),
        ("weight.safetensors", "real finite fixture bytes"),
    ] {
        fs::write(snapshot.join(name), bytes).unwrap();
        fs::write(expected.join(name), bytes).unwrap();
    }
    fs::write(snapshot.join("nested/config.json"), "metadata").unwrap();
    fs::write(expected.join("config.json"), "metadata").unwrap();
    fs::write(snapshot.join("README.md"), "ignored").unwrap();
    fs::write(snapshot.join("nested/README.md"), "also ignored").unwrap();
    let cancel = Cancellation::default();
    let budget = Budget {
        deadline: Instant::now() + Duration::from_secs(5),
        cancellation: &cancel,
    };
    let row = projection::materialize(
        &source(&snapshot, "granite-h1-hybrid"),
        &root.join("output"),
        &projection::fixture_tree(&expected),
        1024 * 1024,
        &budget,
    )
    .unwrap();
    assert_eq!(row["files"].as_object().unwrap().len(), 3);
    assert!(!root.join("output/README.md").exists());
    assert_eq!(
        fs::read(root.join("output/weight.safetensors")).unwrap(),
        b"real finite fixture bytes"
    );
    assert_eq!(row["semantic_qualification_performed"], false);
    temp.close().unwrap();
}
#[test]
fn tokenizer_projection_pin_collision_and_byte_budget_refuse_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::write(root.join("tokenizer.json"), "{}").unwrap();
    fs::create_dir(root.join("nested")).unwrap();
    fs::write(root.join("nested/tokenizer.json"), "collision").unwrap();
    let cancel = Cancellation::default();
    let budget = Budget {
        deadline: Instant::now() + Duration::from_secs(5),
        cancellation: &cancel,
    };
    let output = root.join("output");
    assert!(
        projection::materialize(
            &source(root, "granite-h1-hybrid"),
            &output,
            &"a".repeat(64),
            1024,
            &budget
        )
        .unwrap_err()
        .to_string()
        .contains("collision")
    );
    assert!(!output.exists());
    fs::remove_file(root.join("nested/tokenizer.json")).unwrap();
    let supplied = source(root, "llama32-dense");
    assert!(projection::materialize(&supplied, &output, &"a".repeat(64), 1024, &budget).is_err());
    assert!(!output.exists());
    assert!(
        projection::materialize(&supplied, &output, &supplied.source_tree_sha256, 1, &budget)
            .is_err()
    );
    assert!(!output.exists());
    temp.close().unwrap();
}
#[test]
fn local_budget_cancel_and_deadline_refuse_with_no_invented_output() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
    let supplied = source(temp.path(), "llama32-dense");
    for cancelled in [true, false] {
        let cancel = Cancellation::default();
        if cancelled {
            cancel.cancel();
        }
        let budget = Budget {
            deadline: if cancelled {
                Instant::now() + Duration::from_secs(5)
            } else {
                Instant::now()
            },
            cancellation: &cancel,
        };
        let result = projection::materialize(
            &supplied,
            &temp.path().join("output"),
            &supplied.source_tree_sha256,
            1024,
            &budget,
        );
        assert!(result.is_err());
        assert!(!temp.path().join("output").exists());
    }
    temp.close().unwrap();
}
#[cfg(unix)]
#[test]
fn local_snapshot_symlinks_are_refused_without_following_external_bytes() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
    std::os::unix::fs::symlink("tokenizer.json", temp.path().join("linked")).unwrap();
    let supplied = Source {
        key: "granite-h1-hybrid".into(),
        directory: temp.path().canonicalize().unwrap(),
        source_tree_sha256: "a".repeat(64),
        kind: Kind::ImmutableSnapshot,
    };
    let cancel = Cancellation::default();
    let budget = Budget {
        deadline: Instant::now() + Duration::from_secs(5),
        cancellation: &cancel,
    };
    assert!(
        projection::materialize(
            &supplied,
            &temp.path().join("output"),
            &"a".repeat(64),
            1024,
            &budget
        )
        .is_err()
    );
    assert!(!temp.path().join("output").exists());
    temp.close().unwrap();
}
#[test]
fn local_terminal_admission_requires_finish_cancel_deadline_and_preserves_prior_error() {
    let cancel = Cancellation::default();
    let budget = Budget {
        deadline: Instant::now() + Duration::from_secs(5),
        cancellation: &cancel,
    };
    assert!(super::terminal(Ok(()), Ok(()), &budget).is_ok());
    assert!(super::terminal(Ok(()), Err("finish".into()), &budget).is_err());
    assert_eq!(
        super::terminal(Err("prior".into()), Err("finish".into()), &budget)
            .unwrap_err()
            .to_string(),
        "prior"
    );
    cancel.cancel();
    assert!(super::terminal(Ok(()), Ok(()), &budget).is_err());
    let other = Cancellation::default();
    assert!(
        super::terminal(
            Ok(()),
            Ok(()),
            &Budget {
                deadline: Instant::now(),
                cancellation: &other
            }
        )
        .is_err()
    );
}

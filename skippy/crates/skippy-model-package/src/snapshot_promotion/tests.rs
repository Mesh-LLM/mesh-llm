use super::policy::*;
use anyhow::{Context, Result, bail};
use sha2::Digest;
use std::collections::BTreeMap;

struct Transport {
    parent: String,
    main: BTreeMap<String, Vec<u8>>,
    staging: BTreeMap<String, Vec<u8>>,
    created: Option<SnapshotPlan>,
    publish_calls: usize,
    cleanup_calls: usize,
    cleanup_failure: bool,
}

impl Transport {
    fn new() -> Self {
        Self {
            parent: "a".repeat(40),
            main: BTreeMap::from([
                ("shared/metadata.gguf".into(), b"old-artifact".to_vec()),
                ("model-package.json".into(), b"old-manifest".to_vec()),
            ]),
            staging: BTreeMap::new(),
            created: None,
            publish_calls: 0,
            cleanup_calls: 0,
            cleanup_failure: false,
        }
    }
}

impl SnapshotTransport for Transport {
    fn main_revision(&mut self) -> Result<String> {
        Ok(self.parent.clone())
    }
    fn create_staging(&mut self, plan: &SnapshotPlan) -> Result<()> {
        assert_eq!(plan.parent_commit(), self.parent);
        self.staging = self.main.clone();
        self.created = Some(plan.clone());
        Ok(())
    }
    fn publish(&mut self, plan: &PromotionPlan) -> Result<()> {
        self.publish_calls += 1;
        if plan.parent_commit() != self.parent {
            bail!("parent conflict");
        }
        let mut replacement = self.main.clone();
        for path in plan.paths() {
            replacement.insert(
                path.clone(),
                self.staging
                    .get(path)
                    .context("missing staged artifact")?
                    .clone(),
            );
        }
        self.main = replacement;
        self.parent = "c".repeat(40);
        Ok(())
    }
    fn delete_staging(&mut self, _: &str) -> Result<()> {
        self.cleanup_calls += 1;
        if self.cleanup_failure {
            bail!("cleanup failed");
        }
        self.staging.clear();
        self.created = None;
        Ok(())
    }
}

fn manifest() -> Vec<u8> {
    super::fixtures::manifest(
        "shared/metadata.gguf",
        b"new-artifact".len() as u64,
        sha2::Sha256::digest(b"new-artifact")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    )
}

fn stage(transport: &mut Transport) -> PromotionPlan {
    let state = prepare_with(transport, &"b".repeat(40), "run-2").unwrap();
    transport
        .staging
        .insert("shared/metadata.gguf".into(), b"new-artifact".to_vec());
    transport
        .staging
        .insert("model-package.json".into(), manifest());
    promote(&manifest(), state.staging_revision(), state.parent_commit()).unwrap()
}

#[test]
fn partial_staging_leaves_main_artifacts_and_manifest_unchanged() {
    let mut transport = Transport::new();
    let before = transport.main.clone();
    let state = prepare_with(&mut transport, &"b".repeat(40), "run-1").unwrap();
    transport
        .staging
        .insert("shared/metadata.gguf".into(), b"new-artifact".to_vec());
    assert_eq!(state.parent_commit(), "a".repeat(40));
    assert_eq!(transport.main, before);
    assert_eq!(transport.publish_calls, 0);
}

#[test]
fn complete_staging_promotes_artifacts_and_manifest_in_one_parent_guarded_commit() {
    let mut transport = Transport::new();
    let plan = stage(&mut transport);
    assert!(promote_with(&mut transport, &plan).unwrap().is_none());
    assert_eq!(transport.main["shared/metadata.gguf"], b"new-artifact");
    assert_eq!(transport.main["model-package.json"], manifest());
    assert_eq!(transport.publish_calls, 1);
    assert_eq!(transport.cleanup_calls, 1);
    assert!(transport.staging.is_empty());
}

#[test]
fn stale_parent_refuses_publication_and_retains_staging() {
    let mut transport = Transport::new();
    let plan = stage(&mut transport);
    let before = transport.main.clone();
    transport.parent = "d".repeat(40);
    assert!(promote_with(&mut transport, &plan).is_err());
    assert_eq!(transport.main, before);
    assert_eq!(transport.cleanup_calls, 0);
    assert!(transport.created.is_some());
}

#[test]
fn incomplete_staging_cannot_partially_publish_main() {
    let mut transport = Transport::new();
    let plan = stage(&mut transport);
    let before = transport.main.clone();
    transport.staging.remove("model-package.json");
    assert!(promote_with(&mut transport, &plan).is_err());
    assert_eq!(transport.main, before);
    assert_eq!(transport.cleanup_calls, 0);
}

#[test]
fn cleanup_failure_retains_successful_publication_with_warning() {
    let mut transport = Transport::new();
    let plan = stage(&mut transport);
    transport.cleanup_failure = true;
    assert_eq!(
        promote_with(&mut transport, &plan).unwrap(),
        Some("cleanup failed".into())
    );
    assert_eq!(transport.main["shared/metadata.gguf"], b"new-artifact");
    assert_eq!(transport.publish_calls, 1);
}

#[test]
fn invalid_catalog_paths_and_revisions_refuse_before_publication() {
    for paths in [
        vec![],
        vec!["../escape"],
        vec!["/absolute"],
        vec!["a\\b"],
        vec!["model-package.json"],
        vec!["duplicate", "duplicate"],
    ] {
        let mut doc: skippy_package_format::PackageManifest =
            serde_json::from_slice(&manifest()).unwrap();
        doc.artifact_catalog.entries = paths
            .iter()
            .enumerate()
            .map(|(index, path)| skippy_package_format::Artifact {
                id: if index == 0 {
                    "metadata".into()
                } else {
                    format!("artifact-{index}")
                },
                path: (*path).into(),
                byte_size: 1,
                sha256: "b".repeat(64),
            })
            .collect();
        doc.package_id = doc.computed_package_id().unwrap();
        assert!(
            promote(
                &serde_json::to_vec(&doc).unwrap(),
                "automation/republish-valid",
                &"a".repeat(40)
            )
            .is_err()
        );
    }
    assert!(prepare("main", "token", &"a".repeat(40)).is_err());
    assert!(prepare(&"b".repeat(40), "token", "main").is_err());
    assert!(prepare(&"b".repeat(40), "!!!", &"a".repeat(40)).is_err());
}

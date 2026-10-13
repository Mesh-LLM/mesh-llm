use super::*;
use crate::competitive_acquisition::{contract::digest, full_chain::Peer, listing::Listing};
use contract::{Request, Source};
use std::{collections::BTreeMap, path::Path};
fn fixture(root: &Path, existing: bool) -> (Request, BTreeMap<String, Vec<u8>>) {
    let revision = "a".repeat(40);
    let mut checkpoint = BTreeMap::from([
        ("config.json".to_string(), b"{}".to_vec()),
        (
            "model.safetensors".into(),
            b"finite checkpoint bytes".to_vec(),
        ),
    ]);
    let tokenizer = BTreeMap::from([
        ("tokenizer.json".to_string(), b"base tokenizer".to_vec()),
        ("tokenizer_config.json".into(), b"{}".to_vec()),
        ("special_tokens_map.json".into(), b"{}".to_vec()),
    ]);
    if existing {
        checkpoint.insert(
            "tokenizer.json".into(),
            b"checkpoint tokenizer wins".to_vec(),
        );
    }
    let mut pins: BTreeMap<_, _> = tokenizer
        .iter()
        .map(|(n, b)| (n.clone(), digest(b)))
        .collect();
    pins.extend(checkpoint.iter().map(|(n, b)| (n.clone(), digest(b))));
    let profile = json!({"schema_version":1,"config_sha256":pins["config.json"],"tokenizer_sha256":pins["tokenizer.json"],"tokenizer_config_sha256":pins["tokenizer_config.json"],"chat_template_sha256":null,"pre":"qwen2"});
    let bytes = serde_json::to_vec(&profile).unwrap();
    let profile_path = root.join("profile.json");
    std::fs::write(&profile_path, &bytes).unwrap();
    let mut files = BTreeMap::new();
    for (repo, roster) in [
        ("owner/checkpoint", &checkpoint),
        ("owner/tokenizer", &tokenizer),
    ] {
        files.insert(
            format!("/api/models/{repo}/tree/{revision}"),
            serde_json::to_vec(
                &roster
                    .iter()
                    .map(|(n, b)| json!({"type":"file","path":n,"oid":"fixture","size":b.len()}))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        );
        for (n, b) in roster {
            files.insert(format!("/{repo}/resolve/{revision}/{n}"), b.clone());
        }
    }
    (
        Request {
            schema_version: 1,
            checkpoint: Source {
                repo: "owner/checkpoint".into(),
                revision: revision.clone(),
                files: checkpoint
                    .iter()
                    .map(|(n, b)| (n.clone(), digest(b)))
                    .collect(),
            },
            tokenizer_source: Source {
                repo: "owner/tokenizer".into(),
                revision,
                files: tokenizer
                    .iter()
                    .map(|(n, b)| (n.clone(), digest(b)))
                    .collect(),
            },
            tokenizer_profile: profile_path,
            tokenizer_profile_sha256: digest(&bytes),
            output_directory: root.join("staged"),
            credential_file: None,
            timeout_seconds: 5,
            maximum_bytes: 1024 * 1024,
        },
        files,
    )
}
fn execute(input: &Request, files: BTreeMap<String, Vec<u8>>) -> (anyhow::Result<()>, Value) {
    let peer = Peer::new(files, "a".repeat(40));
    let client = hf_hub::HFClient::builder()
        .endpoint(&peer.endpoint)
        .token("finite-fixture")
        .cache_enabled(false)
        .retry_max_attempts(0)
        .client(
            reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let clients = stitch::Clients {
        client,
        listing: Listing::fixture(&peer.endpoint),
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut evidence = json!({"output_owned":false});
    let result = rt.block_on(stitch::execute(
        input,
        Instant::now() + Duration::from_secs(5),
        &mut evidence,
        Some(clients),
    ));
    let seen = peer.close();
    assert!(!seen.is_empty());
    (result, evidence)
}
#[test]
fn immutable_native_checkpoint_stage_preserves_precedence_complete_roster_and_profile_lineage() {
    for existing in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (input, files) = fixture(&root, existing);
        let (result, e) = execute(&input, files);
        result.unwrap();
        let staged = root.join("staged/mtp-src");
        assert_eq!(
            std::fs::read(staged.join("tokenizer.json")).unwrap(),
            if existing {
                b"checkpoint tokenizer wins".as_slice()
            } else {
                b"base tokenizer".as_slice()
            }
        );
        assert_eq!(
            e["lineage"]["tokenizer.json"],
            if existing {
                "checkpoint"
            } else {
                "tokenizer-source"
            }
        );
        let rows = e["checkpoint_files"].as_array().unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(e["loaded_model_qualification"], false);
        assert_eq!(
            std::fs::read(root.join("profile.json")).unwrap(),
            std::fs::read(root.join("staged/tokenizer-profile.json")).unwrap()
        );
        temp.close().unwrap();
    }
}
#[test]
fn immutable_stage_refuses_roster_hash_and_combined_budget_failures_without_conversion_claim() {
    for mode in ["missing", "hash", "budget"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let (mut input, files) = fixture(&root, false);
        match mode {
            "missing" => {
                input
                    .checkpoint
                    .files
                    .insert("absent.json".into(), "a".repeat(64));
            }
            "hash" => {
                input
                    .checkpoint
                    .files
                    .insert("model.safetensors".into(), "b".repeat(64));
            }
            _ => input.maximum_bytes = 40,
        };
        let (result, e) = execute(&input, files);
        assert!(result.is_err());
        assert!(e["checkpoint_directory"].is_null());
        assert!(!root.join("staged/mtp-src").exists());
        temp.close().unwrap();
    }
}
#[test]
fn profile_mismatch_and_expired_stage_refuse_before_output_creation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (mut input, _) = fixture(&root, false);
    input.tokenizer_profile_sha256 = "a".repeat(64);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut e = json!({});
    assert!(
        rt.block_on(stitch::execute(
            &input,
            Instant::now() + Duration::from_secs(5),
            &mut e,
            None
        ))
        .is_err()
    );
    assert!(!input.output_directory.exists());
    assert!(
        rt.block_on(stitch::execute(&input, Instant::now(), &mut e, None))
            .is_err()
    );
    assert!(!input.output_directory.exists());
    temp.close().unwrap();
}

#[test]
fn tokenizer_roster_exceeding_remaining_whole_budget_refuses_before_tokenizer_file_requests() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (mut input, files) = fixture(&root, false);
    let first_bytes = files
        .iter()
        .filter(|(p, _)| p.starts_with("/owner/checkpoint/resolve/"))
        .map(|(_, b)| b.len() as u64)
        .sum::<u64>();
    input.maximum_bytes = first_bytes + 1;
    let peer = Peer::new(files, "a".repeat(40));
    let client = hf_hub::HFClient::builder()
        .endpoint(&peer.endpoint)
        .token("finite-fixture")
        .cache_enabled(false)
        .retry_max_attempts(0)
        .client(
            reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap(),
        )
        .build()
        .unwrap();
    let clients = stitch::Clients {
        client,
        listing: Listing::fixture(&peer.endpoint),
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut e = json!({"output_owned":false});
    let result = rt.block_on(stitch::execute(
        &input,
        Instant::now() + Duration::from_secs(5),
        &mut e,
        Some(clients),
    ));
    let seen = peer.close();
    assert!(result.is_err());
    assert_eq!(e["checkpoint_source"]["bytes"], first_bytes);
    assert!(seen.iter().any(
        |r| r.starts_with("GET /owner/checkpoint/resolve/") && r.contains("model.safetensors")
    ));
    assert!(
        seen.iter()
            .any(|r| r.starts_with("GET /api/models/owner/tokenizer/tree/"))
    );
    assert!(
        !seen
            .iter()
            .any(|r| r.contains(" /owner/tokenizer/resolve/"))
    );
    assert!(!root.join("staged/tokenizer-source").exists());
    assert!(!root.join("staged/mtp-src").exists());
    temp.close().unwrap();
}

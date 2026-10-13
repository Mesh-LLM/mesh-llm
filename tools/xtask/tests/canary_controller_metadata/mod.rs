use super::{Fixture, Value, fs, json};
use sha2::{Digest, Sha256};
use std::path::Path;

fn gguf(architecture: &str, fields: &[(&str, u64)]) -> Vec<u8> {
    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend(u64::try_from(text.len()).unwrap().to_le_bytes());
        bytes.extend(text.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(u64::try_from(fields.len() + 1).unwrap().to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, architecture);
    for (key, value) in fields {
        string(&mut bytes, key);
        bytes.extend(10_u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn artifact(root: &Path, name: &str, bytes: &[u8]) -> Value {
    let repository = root.join("hub/models--fixture--metadata");
    let revision = "a".repeat(40);
    let snapshot = repository.join("snapshots").join(&revision);
    fs::create_dir_all(&snapshot).unwrap();
    fs::create_dir_all(repository.join("blobs")).unwrap();
    let digest = hex::encode(Sha256::digest(bytes));
    let blob = repository.join("blobs").join(&digest);
    fs::write(&blob, bytes).unwrap();
    std::os::unix::fs::symlink(&blob, snapshot.join(name)).unwrap();
    json!({"repo":"fixture/metadata","revision":revision,"files":[name],
        "selector":"fixture","file_integrity":{name:{"size_bytes":bytes.len(),"blob_id":digest}}})
}

pub(super) fn fixture(target: &[u8], draft: &[u8]) -> Fixture {
    let mut fixture = Fixture::new();
    let root = fixture.directory.path();
    let cache = root.join("cache");
    let manifest_path = root.join("selected/ci/llama-canary/family-certified.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["models"].as_array_mut().unwrap().truncate(1);
    let model = &mut manifest["models"][0];
    model["artifact"] = artifact(&cache, "target.gguf", target);
    model["draft_artifact"] = artifact(&cache, "draft.gguf", draft);
    model["mmproj_artifact"] = artifact(&cache, "projector.gguf", b"projector");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    fs::write(
        root.join("selected/scripts/skippy-family-battery.sh"),
        "set -eu\nmkdir -p \"$FAMILY_BATTERY_ARTIFACT_ROOT\"\nprintf 'executed' > \"$FAMILY_BATTERY_ARTIFACT_ROOT/../battery-ran\"\n",
    )
    .unwrap();
    fixture.input["cache"] = json!({"mode":"gguf_metadata","root":cache});
    fixture
}

pub(super) fn valid_target() -> Vec<u8> {
    gguf(
        "qwen3",
        &[("qwen3.block_count", 1), ("qwen3.embedding_length", 1)],
    )
}

pub(super) fn valid_draft() -> Vec<u8> {
    gguf(
        "llama",
        &[("llama.block_count", 4), ("llama.embedding_length", 512)],
    )
}

fn identity(fixture: &Fixture) -> Value {
    serde_json::from_slice(&fs::read(fixture.output("source-plan.json")).unwrap()).unwrap()
}

#[test]
fn metadata_admission_when_target_and_independent_draft_are_valid() {
    let fixture = fixture(&valid_target(), &valid_draft());

    let result = fixture.run("source-plan");

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(identity(&fixture)["gguf_admission"], "metadata_admitted");
    assert_eq!(identity(&fixture)["cache_admission"], "gguf_metadata");
    assert!(fixture.output("battery-ran").is_file());
    assert!(fixture.run("verify-source-plan").status.success());
}

#[test]
fn admission_rejects_invalid_metadata_before_battery_or_output() {
    let mut duplicate = valid_target();
    duplicate[16..24].copy_from_slice(&4_u64.to_le_bytes());
    let extra = gguf("qwen3", &[("qwen3.block_count", 1)]);
    let architecture_bytes = gguf("qwen3", &[]).len();
    duplicate.extend(&extra[architecture_bytes..]);
    let mut truncated = valid_target();
    truncated.pop();
    for (target, draft) in [
        (b"malformed".to_vec(), valid_draft()),
        (truncated, valid_draft()),
        (duplicate, valid_draft()),
        (
            gguf(
                "qwen3",
                &[("qwen3.block_count", 1), ("qwen3.embedding_length", 2)],
            ),
            valid_draft(),
        ),
        (valid_target(), gguf("llama", &[])),
        (
            valid_target(),
            gguf(
                "llama",
                &[("llama.block_count", 0), ("llama.embedding_length", 512)],
            ),
        ),
    ] {
        let fixture = fixture(&target, &draft);

        let result = fixture.run("source-plan");

        assert!(!result.status.success());
        assert!(!fixture.output("plan.json").exists());
        assert!(!fixture.output("source-plan.json").exists());
        assert!(!fixture.output("battery-ran").exists());
    }
}

#[test]
fn corrupt_projector_rejects_admission_before_battery_or_output() {
    let fixture = fixture(&valid_target(), &valid_draft());
    let digest = hex::encode(Sha256::digest(b"projector"));
    fs::write(
        fixture
            .directory
            .path()
            .join("cache/hub/models--fixture--metadata/blobs")
            .join(digest),
        b"corrupted",
    )
    .unwrap();

    let result = fixture.run("source-plan");

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SHA-256 mismatch"));
    assert!(!fixture.output("plan.json").exists());
    assert!(!fixture.output("battery-ran").exists());
}

#[test]
fn verification_rejects_cache_policy_downgrades() {
    for mode in ["not_checked", "blob_identity"] {
        let mut fixture = fixture(&valid_target(), &valid_draft());
        assert!(fixture.run("source-plan").status.success());
        fixture.input["cache"]["mode"] = mode.into();
        if mode == "not_checked" {
            fixture.input["cache"]
                .as_object_mut()
                .unwrap()
                .remove("root");
        }

        let result = fixture.run("verify-source-plan");

        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("cache admission policy differs"));
    }
}

#[test]
fn verification_rejects_admission_state_downgrade() {
    let fixture = fixture(&valid_target(), &valid_draft());
    assert!(fixture.run("source-plan").status.success());
    let mut identity = identity(&fixture);
    identity["gguf_admission"] = "pending".into();
    fs::write(
        fixture.output("source-plan.json"),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();

    let result = fixture.run("verify-source-plan");

    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("GGUF admission state differs"));
}

#[test]
fn blob_only_admission_when_dimensions_are_unchecked_remains_pending() {
    let mut fixture = fixture(b"unchecked target", b"unchecked draft");
    fixture.input["cache"]["mode"] = "blob_identity".into();

    let result = fixture.run("source-plan");

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(identity(&fixture)["gguf_admission"], "pending");
    assert!(fixture.run("verify-source-plan").status.success());
}

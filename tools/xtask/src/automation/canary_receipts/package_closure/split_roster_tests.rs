use super::*;
use serde_json::json;

fn manifest() -> Value {
    json!({"policy":{"profiles":{"cert":{"status":"certified","required_lanes":["single-step","chain","state-handoff"]},"partial":{"status":"certified","required_lanes":["single-step"]}}},"models":[{"class":"causal_generation","profile":"cert","architecture":"zeta"},{"class":"causal_generation","profile":"cert","architecture":"alpha"},{"class":"causal_generation","profile":"cert","architecture":"alpha"},{"class":"causal_generation","profile":"partial","architecture":"excluded"},{"class":"embedding","profile":"workload-oracle","architecture":"auxiliary"}]})
}
#[test]
fn roster_requires_causal_certified_core_lanes_and_excludes_nonchat_profiles() {
    assert_eq!(architectures(&manifest()).unwrap(), ["alpha", "zeta"]);
    for (class, profile) in [
        ("embedding", "cert"),
        ("causal_generation", "workload-oracle"),
        ("unknown", "cert"),
    ] {
        let mut document = manifest();
        document["models"][0]["class"] = json!(class);
        document["models"][0]["profile"] = json!(profile);
        assert!(architectures(&document).is_err());
    }
    let mut document = manifest();
    document["models"][0]["architecture"] = json!("");
    assert!(architectures(&document).is_err());
}
#[test]
fn roster_bytes_bind_independent_v2_patch_frames_pin_and_all_abi_components() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let patches = root.join("third_party/llama.cpp/patches");
    fs::create_dir_all(patches.join("model_support")).unwrap();
    fs::create_dir(patches.join("generated")).unwrap();
    let members = [
        ("0001-core.patch", b"core".as_slice()),
        ("model_support/0001-family.patch", b"family".as_slice()),
        ("generated/0001-family-test.patch", b"graph".as_slice()),
    ];
    for (name, bytes) in members {
        fs::write(patches.join(name), bytes).unwrap();
    }
    fs::write(patches.join("model_support/series"), "0001-family.patch\n").unwrap();
    fs::write(patches.join("generated/series"), "0001-family-test.patch\n").unwrap();
    let mut framed = b"mesh-llm-skippy-patch-queue-v2\0".to_vec();
    framed.extend(3_u64.to_le_bytes());
    for (name, bytes) in members {
        framed.extend(u64::try_from(name.len()).unwrap().to_le_bytes());
        framed.extend(name.as_bytes());
        framed.extend(u64::try_from(bytes.len()).unwrap().to_le_bytes());
        framed.extend(bytes);
    }
    let expected = Digest::of_bytes(&framed);
    assert_eq!(queue(root).unwrap(), expected);
    fs::write(
        root.join("third_party/llama.cpp/upstream.txt"),
        "a".repeat(40),
    )
    .unwrap();
    fs::create_dir_all(root.join("crates/skippy-ffi/src")).unwrap();
    let ffi = root.join("crates/skippy-ffi/src/lib.rs");
    fs::write(&ffi,"pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n").unwrap();
    let bytes = render(root, &serde_json::to_vec(&manifest()).unwrap()).unwrap();
    let expected_bytes = format!(
        "{{\n  \"schema_version\": 2,\n  \"native_recipe\": {{\n    \"llama_upstream_sha\": \"{}\",\n    \"skippy_abi\": \"1.2.3\",\n    \"patch_queue_sha256\": \"{}\"\n  }},\n  \"architectures\": [\n    \"alpha\",\n    \"zeta\"\n  ]\n}}\n",
        "a".repeat(40),
        expected.as_str()
    );
    assert_eq!(bytes, expected_bytes.as_bytes());
    fs::write(patches.join("generated/0001-family-test.patch"), b"changed").unwrap();
    assert_ne!(queue(root).unwrap(), expected);
    fs::write(ffi, "pub const ABI_VERSION_MAJOR: u32 = 1;\n").unwrap();
    assert!(render(root, &serde_json::to_vec(&manifest()).unwrap()).is_err());
}

use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub(super) fn executable(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, contents)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

pub(super) fn inputs(root: &Path, model: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    for (key, text) in [
        ("general.architecture", Some("fixture")),
        ("fixture.context_length", None),
    ] {
        bytes.extend(u64::try_from(key.len())?.to_le_bytes());
        bytes.extend(key.as_bytes());
        if let Some(text) = text {
            bytes.extend(8_u32.to_le_bytes());
            bytes.extend(u64::try_from(text.len())?.to_le_bytes());
            bytes.extend(text.as_bytes());
        } else {
            bytes.extend(4_u32.to_le_bytes());
            bytes.extend(131072_u32.to_le_bytes());
        }
    }
    fs::write(model, &bytes)?;
    let dataset = b"pinned synthetic dataset";
    fs::write(root.join("data.parquet"), dataset)?;
    let mut matrix: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../fixtures/migration/optional_replay/valid.json"
    ))?;
    matrix["models"] = serde_json::json!([{"family":"granite-3.1-2b","class":"dense","repo":"fixture/model",
        "revision":"0123456789abcdef0123456789abcdef01234567","file":"model.gguf","quant":"fixture",
        "sha256":hex::encode(Sha256::digest(&bytes)),"native_context_tokens":131072}]);
    let replay = matrix["replay"].as_object_mut().ok_or("replay block")?;
    for (key, value) in [
        ("sessions_per_concurrency", 3),
        ("minimum_worker_waves", 2),
        ("minimum_session_prompt_tokens", 1),
        ("min_isl", 1),
        ("max_isl", 100),
        ("min_turns", 1),
        ("passes", 1),
        ("warmup_turns", 1),
    ] {
        replay.insert(key.into(), value.into());
    }
    replay.insert("concurrency".into(), serde_json::json!([1]));
    replay.insert(
        "dataset_sha256".into(),
        hex::encode(Sha256::digest(dataset)).into(),
    );
    fs::write(root.join("matrix.json"), serde_json::to_vec(&matrix)?)?;
    let trajectory = |id: &str, framework: &str| {
        serde_json::json!({"session_id":id,
        "source_dataset":"fixture","agent_framework":framework,"recorded_model":null,
        "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
            {"role":"user","content":"next"},{"role":"assistant","content":"final"}]})
    };
    let manifest = serde_json::json!({"cohorts":{
        "warmup":[trajectory("warmup","swe-agent")],
        "1":[trajectory("first","swe-agent"),trajectory("second","mini-swe-agent"),trajectory("third","openhands")]
    }});
    fs::write(root.join("manifest.json"), serde_json::to_vec(&manifest)?)?;
    Ok(())
}

pub(super) fn builds(root: &Path, worktrees: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let bin = root.join("bin");
    fs::create_dir(&bin)?;
    executable(
        &bin.join("git"),
        "#!/bin/sh\ncase \"$1\" in rev-parse) printf '%s\\n' '0123456789abcdef0123456789abcdef01234567';; status) exit 0;; *) exit 9;; esac\n",
    )?;
    executable(
        &bin.join("just"),
        &format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> {}/build-calls.txt\n",
            quote(root)
        ),
    )?;
    let worktree = worktrees.join("main-0123456789");
    let binary = worktree.join("target/release/mesh-llm");
    fs::create_dir_all(binary.parent().ok_or("binary parent")?)?;
    fs::copy(env!("CARGO_BIN_EXE_laya-product-fixture"), binary)?;
    let runtime = worktree.join("dist/native-runtimes/fixture");
    fs::create_dir_all(&runtime)?;
    fs::write(runtime.join("runtime.so"), b"fixture runtime")?;
    fs::write(
        runtime.join("manifest.json"),
        br#"{"runtime":{"backend":{"kind":"metal"}}}"#,
    )?;
    Ok(())
}

fn quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

//! Closed inert publisher peer: tests subprocess transport custody, never HF mutation.
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: PathBuf,
    path_in_repo: String,
    sha256: String,
    byte_size: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    shards: Vec<Artifact>,
    sidecars: Vec<Artifact>,
    credential_file: Option<PathBuf>,
    execution_timeout_ms: u64,
}
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let [verb, a, path, b, output] = args else {
        return Err("publisher fixture closed argv".into());
    };
    if verb != "publish" || a != "--input" || b != "--output-directory" {
        return Err("publisher fixture closed argv".into());
    }
    let mut bytes = Vec::new();
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("fixture regular input".into());
    }
    std::fs::File::open(path)?
        .take(512 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 512 * 1024 {
        return Err("publisher fixture input bound".into());
    }
    let input: Input = serde_json::from_slice(&bytes)?;
    let hash = hex::encode(Sha256::digest(serde_json::to_vec(&input)?));
    let output = PathBuf::from(output);
    std::fs::create_dir(&output)?;
    let mut marker = Vec::new();
    std::fs::File::open(&input.shards[0].path)?
        .take(257)
        .read_to_end(&mut marker)?;
    if marker.len() > 256 {
        return Err("fixture mode bytes bound".into());
    }
    let mode = std::str::from_utf8(&marker)?
        .strip_prefix("GGUF")
        .ok_or("fixture GGUF mode")?;
    crate::signals::install()?;
    let paths = input
        .shards
        .iter()
        .chain(&input.sidecars)
        .map(|a| a.path_in_repo.clone())
        .collect::<Vec<_>>();
    let mut publication = json!({"schema_version":1,"repo":input.repo,"parent_commit":input.parent_commit,
        "ordered_paths":paths,"objects":input.shards.iter().map(|a|json!({"oid":a.sha256,"size":a.byte_size,"mutation_attempted":true,
            "uploaded_parts":1,"object_present":true,"source_custody_verified":true,"completed":true,"error":null})).collect::<Vec<_>>(),
        "object_attempted_paths":input.shards.iter().map(|a|a.path_in_repo.clone()).collect::<Vec<_>>(),
        "commit_attempted":true,"commit_oid":"c".repeat(40),"remote_verified_paths":paths,"final_source_custody_verified":true,"completed":false,"error":null});
    let mut receipt = json!({"schema_version":1,"request_sha256":hash,"status":"IN_PROGRESS","publication":publication,
        "source_custody_verified":false,"error":null});
    std::fs::write(output.join("progress.json"), serde_json::to_vec(&receipt)?)?;
    std::fs::write(
        output.join("started.json"),
        b"owned publisher fixture admitted",
    )?;
    if mode == "held" || mode == "forced" {
        let deadline = Instant::now() + Duration::from_secs(15);
        while (mode == "forced" || !crate::signals::stopped()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        receipt["status"] = json!("FAILED");
        receipt["error"] = json!("fixture cancelled or deadline");
        std::fs::write(
            output.join("publication.json"),
            serde_json::to_vec(&receipt)?,
        )?;
        std::process::exit(1);
    }
    publication = receipt["publication"].clone();
    publication["completed"] = json!(true);
    receipt["publication"] = publication;
    receipt["status"] = json!("PUBLISHED");
    receipt["source_custody_verified"] = json!(true);
    match mode {
        "outer-error" => receipt["error"] = json!("contradictory successful outer error"),
        "object-error" => {
            receipt["publication"]["objects"][0]["error"] =
                json!("contradictory completed object error")
        }
        "partial-errors" => {
            receipt["status"] = json!("FAILED");
            receipt["error"] = json!("partial outer error");
            receipt["publication"]["completed"] = json!(false);
            receipt["publication"]["error"] = json!("partial publication error");
            receipt["publication"]["objects"][0]["completed"] = json!(false);
            receipt["publication"]["objects"][0]["error"] = json!("partial object error");
        }
        "wrong-hash" => receipt["request_sha256"] = json!("0".repeat(64)),
        "missing-custody" => receipt["publication"]["final_source_custody_verified"] = json!(false),
        "wrong-roster" => receipt["publication"]["ordered_paths"] = json!(["other.gguf"]),
        "nonzero" => (),
        "source-drift" => std::fs::write(
            std::env::current_dir()?
                .parent()
                .ok_or("fixture parent")?
                .join("source.txt"),
            b"changed fixture source",
        )?,
        "ok" => (),
        _ => return Err("unknown publisher fixture mode".into()),
    }
    std::fs::write(
        output.join("publication.json"),
        serde_json::to_vec(&receipt)?,
    )?;
    if mode == "nonzero" {
        std::process::exit(7);
    }
    Ok(())
}

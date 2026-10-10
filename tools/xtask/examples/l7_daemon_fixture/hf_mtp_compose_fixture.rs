//! Owned inert compose/attach process peer; no native GGUF loading.
use std::{io::Write, path::Path};
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let value = |key: &str| {
        args.windows(2)
            .find(|p| p[0] == key)
            .map(|p| p[1].clone())
            .ok_or("fixture flag")
    };
    let draft = value(if args[0] == "compose-mtp" {
        "--mtp-gguf"
    } else {
        "--mtp-draft"
    })?;
    let bytes = std::fs::read(draft)?;
    let mode = std::str::from_utf8(bytes.get(4..).ok_or("GGUF fixture")?)?.trim();
    let mut records = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("invocations.jsonl")?;
    records.write_all(serde_json::to_string(args)?.as_bytes())?;
    records.write_all(b"\n")?;
    records.flush()?;
    if mode == "held" {
        crate::signals::install()?;
        std::fs::write("held-marker", "ready")?;
        while !crate::signals::stopped() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        return Ok(());
    }
    if mode == "nonzero" {
        return Err("inert compose failure".into());
    }
    if mode == "malformed" {
        println!("not json");
        return Ok(());
    }
    if args[0] == "compose-mtp" {
        let target = value("--target-shard")?;
        let metadata = value("--metadata-shard")?;
        let output = value("--output")?;
        let first = value("--metadata-output")?;
        if value("--set-kv")? != "nemotron_h_moe.nextn_predict_layers=1"
            || args.last().is_none_or(|a| a != "--json")
        {
            return Err("fixture compose argv".into());
        }
        let mut first_bytes = std::fs::read(&metadata)?;
        first_bytes.extend_from_slice(b"patched-metadata");
        let mut last_bytes = std::fs::read(&target)?;
        last_bytes.extend_from_slice(b"composed-tensors");
        std::fs::write(&first, first_bytes)?;
        if mode != "missing-output" {
            std::fs::write(&output, last_bytes)?;
        }
        let report = serde_json::json!({"target_shard":target,"mtp_gguf":value("--mtp-gguf")?,"output":if mode=="wrong-path"{"/unrelated.gguf"}else{&output},"metadata_shard":first,"target_tensors":1,"mtp_tensors":1,"appended_bytes":16,"block_count":value("--mtp-block")?.parse::<u32>()?+1});
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let models = args
            .windows(2)
            .filter(|p| p[0] == "--model")
            .map(|p| p[1].clone())
            .collect::<Vec<_>>();
        if args.iter().any(|v| v == "--projector") {
            return Err("compose must not introduce projector".into());
        }
        if mode == "mutate-middle" {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&models[1])?
                .write_all(b"changed")?;
        }
        if mode == "mutate-output" {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&models[0])?
                .write_all(b"changed")?;
        }
        let report = serde_json::json!({"model_parts":models,"mtp_draft":value("--mtp-draft")?,"projector":null,"layer_count":value("--layer-count")?.parse::<u32>()?,"mtp_layer_count":1,"ctx_size":64,"native_mtp_multimodal_feature":mode!="no-feature","session_created":true});
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    let _ = Path::new(".");
    Ok(())
}

//! Owned inert native conversion/verify process, not tensor or runtime validation.
use std::{io::Write, path::Path};
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let value = |key: &str| {
        args.windows(2)
            .find(|pair| pair[0] == key)
            .map(|pair| pair[1].clone())
            .ok_or("raw fixture flag")
    };
    let mut record = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("conversion-invocations.jsonl")?;
    record.write_all(serde_json::to_string(args)?.as_bytes())?;
    record.write_all(b"\n")?;
    record.flush()?;
    if args[0] == "convert" {
        if value("--backend")? != "native-rust"
            || !args.iter().any(|a| a == "--mtp")
            || value("--output-type")? != "bf16"
            || value("--expected-splits")? != "1"
            || value("--window-size")? != "1"
            || value("--split-max-size")? != "0"
            || !value("--target-prefix")?.is_empty()
            || !args.iter().any(|a| a == "--no-verify-on-complete")
        {
            return Err("raw fixture closed native conversion argv".into());
        }
        let profile = value("--nemotron-mtp-tokenizer-profile")?;
        let _: serde_json::Value = serde_json::from_slice(&std::fs::read(profile)?)?;
        let source = Path::new(args.last().ok_or("raw source")?);
        let mode = std::fs::read_to_string(source.join("fixture-mode"))?;
        if mode == "held" {
            crate::signals::install()?;
            std::fs::write("conversion-held-marker", "ready")?;
            while !crate::signals::stopped() {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            return Ok(());
        }
        if mode == "nonzero" {
            std::fs::write("partial-spool", "owned partial bytes")?;
            return Err("inert native conversion refusal".into());
        }
        let output = value("--outfile")?;
        let root = Path::new(&output).parent().ok_or("raw output root")?;
        let manifest = serde_json::json!({"schema_version":1,"kind":"CONVERT_HF","source":source,"source_prefix":null,"target":root,"target_prefix":"","output_basename":"mtp","expected_splits":1,"window_size":1,"quant":null,"output_type":"BF16","tensor_type_file":null,"tensor_type_recipe":null});
        std::fs::write(value("--manifest")?, serde_json::to_vec(&manifest)?)?;
        if mode != "missing" {
            std::fs::write(output, b"GGUFsuccess")?;
        }
        if mode == "drift" {
            std::fs::write(source.join("config.json"), br#"{"inert_changed":true}"#)?;
        }
        println!("inert native conversion observed; not model qualification");
    } else {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(value("--manifest")?)?)?;
        let root = Path::new(manifest["target"].as_str().ok_or("root")?);
        if !root.join("mtp.gguf").is_file() {
            return Err("missing converted fixture".into());
        }
        let mode = std::fs::read_to_string(
            Path::new(manifest["source"].as_str().ok_or("source")?).join("fixture-mode"),
        )?;
        let report = serde_json::json!({"root":root,"prefix":"","basename":if mode=="bad-verify"{"wrong"}else{"mtp"},"expected_splits":1,"completed_count":1,"first_missing":null,"last_present":1,"first_shard":"mtp-00001-of-00001.gguf","last_shard":"mtp-00001-of-00001.gguf","complete":true});
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

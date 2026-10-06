//! Inert product argv/report peer. No GGUF loading, serving or networking.
use std::{io::Write, path::Path};
pub(super) fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let value = |name: &str| {
        args.windows(2)
            .find(|p| p[0] == name)
            .map(|p| p[1].clone())
            .ok_or("fixture missing flag")
    };
    let projector = value("--projector")?;
    let bytes = std::fs::read(&projector)?;
    let mode = std::str::from_utf8(bytes.get(4..).ok_or("fixture magic")?)?.trim();
    std::fs::write("invoked.json", serde_json::to_vec(args)?)?;
    if mode == "held" {
        crate::signals::install()?;
        while !crate::signals::stopped() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        std::fs::write("stopped", "owned fixture stop")?;
        return Ok(());
    }
    if mode == "nonzero" {
        return Err("inert native failure".into());
    }
    if mode == "malformed" {
        println!("not native JSON");
        return Ok(());
    }
    let mut report = if args.first().is_some_and(|a| a == "validate-projector") {
        if args.len() != 4 || args[1] != "--projector" || args[3] != "--json" {
            return Err("projector argv refused".into());
        }
        serde_json::json!({"projector":projector,"warmup":true,"loaded":true})
    } else {
        let models = args
            .windows(2)
            .filter(|p| p[0] == "--model")
            .map(|p| p[1].clone())
            .collect::<Vec<_>>();
        let known = [
            "--model",
            "--mtp-draft",
            "--layer-count",
            "--ctx-size",
            "--mtp-layer-count",
            "--projector",
        ];
        let mut rest = &args[1..];
        while !rest.is_empty() {
            if rest == ["--json"] {
                break;
            }
            if rest.len() < 2 || !known.contains(&rest[0].as_str()) {
                return Err("MTP argv refused".into());
            }
            rest = &rest[2..];
        }
        serde_json::json!({"projector":projector,"model_parts":models,"mtp_draft":value("--mtp-draft")?,"layer_count":value("--layer-count")?.parse::<u32>()?,"ctx_size":value("--ctx-size")?.parse::<u32>()?,"mtp_layer_count":args.windows(2).find(|p|p[0]=="--mtp-layer-count").map(|p|p[1].parse::<u32>()).transpose()?.unwrap_or(1),"session_created":true,"native_mtp_multimodal_feature":true})
    };
    if mode == "wrong-path" {
        report["projector"] = serde_json::json!("/unrelated.gguf")
    }
    if mode == "zero-feature" {
        report["native_mtp_multimodal_feature"] = serde_json::json!(false)
    }
    if mode == "mutate" {
        std::fs::OpenOptions::new()
            .append(true)
            .open(Path::new(&projector))?
            .write_all(b"changed")?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

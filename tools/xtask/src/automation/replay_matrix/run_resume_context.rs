use crate::command::DynResult;
use std::path::Path;

pub(super) fn verify(
    previous: &serde_json::Value,
    current: &serde_json::Value,
    run_path: &Path,
) -> DynResult<()> {
    if current["config"]["context_qualification"] == "captured" {
        if previous["context_preflight"] != current["context_preflight"] {
            return Err("cannot resume: captured context evidence differs".into());
        }
        return Ok(());
    }
    for build in current["builds"]
        .as_array()
        .ok_or("invalid build identities")?
    {
        let label = build["label"].as_str().ok_or("missing build label")?;
        if previous["context_preflight"][label]["passed"] != true {
            return Err("cannot resume: successful context preflight is missing".into());
        }
        let directory = run_path
            .parent()
            .ok_or("missing run directory")?
            .join("context-preflight")
            .join(label);
        super::pass_identity::verify(
            &directory,
            &previous["context_preflight"][label]["artifact_sha256"],
        )?;
        let retained: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("eligibility.json"))?)?;
        if retained != previous["context_preflight"][label] {
            return Err("cannot resume: preflight snapshot differs from retained evidence".into());
        }
    }
    Ok(())
}

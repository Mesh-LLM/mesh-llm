//! Complete deterministic candidate/native TTS waveform comparison.
mod admission;
mod execution;
mod options;
mod pcm;
mod regular_input;
#[cfg(test)]
mod tests;
use crate::command::DynResult;
use serde_json::json;
use std::{fs, io::Write, path::Path};
const PROMPT: &str = "The mesh is ready.";
fn receipt(path: &Path, value: &serde_json::Value) -> DynResult<()> {
    let mut random = [0; 16];
    getrandom::fill(&mut random)
        .map_err(|error| format!("TTS receipt randomness unavailable: {error}"))?;
    let temporary = path.with_file_name(format!(".tts-receipt-{}", hex::encode(random)));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        Ok::<_, Box<dyn std::error::Error>>(fs::rename(&temporary, path)?)
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let options = options::Options::parse(args)?;
    let admission = admission::admit(&options)?;
    let (candidate_path, oracle_path) = execution::fresh(&options.work)?;
    execution::run(&options, &admission, &candidate_path, &oracle_path)?;
    let candidate = pcm::read(&candidate_path)?;
    let oracle = pcm::read(&oracle_path)?;
    let metrics = pcm::compare(&candidate, &oracle)?;
    let result = json!({"status":"pass","class":"speech_synthesis","mode":"deterministic_local_monolithic_pcm_parity","prompt":PROMPT,"seed":7,"top_k":20,"top_p":0.8,"max_frames":512,"pinned_patch_sha":admission.patched,"thresholds":{"max_relative_rms_error":0.02,"min_waveform_cosine":0.9995},"metrics":metrics,"candidate_wav_sha256":candidate.digest,"oracle_wav_sha256":oracle.digest});
    receipt(&options.work.join("tts-oracle-result.json"), &result)?;
    writeln!(
        crate::cli_output::stdout(),
        "speech_synthesis local-monolithic oracle passed: {}",
        serde_json::to_string(&result["metrics"])?
    )?;
    Ok(())
}

use super::contract::{Arm, Input, Workload};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sample {
    pub arm: String,
    pub workload: String,
    pub run: u32,
    pub wall_seconds: f64,
    pub wall_tok_s: f64,
    pub server_tok_s: f64,
    pub predicted_n: u64,
    pub draft_n: u64,
    pub draft_accepted: u64,
    pub ngram_proposer: String,
    pub ngram_tokens: u64,
    pub ngram_accepted_tokens: u64,
    pub proposer_attempts: u64,
    pub proposer_hits: u64,
    pub proposer_match_length_max: u64,
    pub proposer_candidates_examined: u64,
    pub proposer_appended_tokens: u64,
    pub proposer_rebuilds: u64,
    pub proposer_sync_us: u64,
    pub proposer_lookup_us: u64,
    pub finish_reason: String,
    pub output_sha256: String,
}
fn count(v: &Value, key: &str) -> DynResult<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("missing/invalid observed timing {key}").into())
}
pub(super) fn decode(
    v: &Value,
    arm: &Arm,
    w: &Workload,
    run: u32,
    input: &Input,
    wall: f64,
) -> DynResult<Sample> {
    if !v.is_object() || v.get("error").is_some_and(|v| !v.is_null()) {
        return Err("suffix response error/object refusal".into());
    }
    let t = &v["timings"];
    let predicted = count(t, "predicted_n")?;
    let rate = t["predicted_per_second"]
        .as_f64()
        .ok_or("missing observed server rate")?;
    if predicted == 0
        || predicted > input.max_tokens
        || !rate.is_finite()
        || rate <= 0.0
        || !wall.is_finite()
        || wall <= 0.0
    {
        return Err("unusable observed suffix timings".into());
    }
    let choices = v["choices"]
        .as_array()
        .filter(|c| !c.is_empty())
        .ok_or("missing choices")?;
    let c = &choices[0];
    let content = c["message"]["content"].as_str().unwrap_or("");
    let reason = c["finish_reason"].as_str().unwrap_or("unknown");
    let proposer = t["native_mtp_ngram_proposer"]
        .as_str()
        .ok_or("missing proposer source")?;
    if proposer.len() > 128
        || proposer.chars().any(char::is_control)
        || reason.len() > 128
        || reason.chars().any(char::is_control)
    {
        return Err("invalid timing label".into());
    }
    let s = Sample {
        arm: arm.name.clone(),
        workload: w.name.clone(),
        run,
        wall_seconds: wall,
        wall_tok_s: crate::automation::openai_exchange::stream::number(predicted) / wall,
        server_tok_s: rate,
        predicted_n: predicted,
        draft_n: count(t, "draft_n")?,
        draft_accepted: count(t, "draft_n_accepted")?,
        ngram_proposer: proposer.into(),
        ngram_tokens: count(t, "native_mtp_hybrid_ngram_tokens")?,
        ngram_accepted_tokens: count(t, "native_mtp_hybrid_accepted_tail_tokens")?,
        proposer_attempts: count(t, "native_mtp_ngram_proposer_attempts")?,
        proposer_hits: count(t, "native_mtp_ngram_proposer_hits")?,
        proposer_match_length_max: count(t, "native_mtp_ngram_proposer_match_length_max")?,
        proposer_candidates_examined: count(t, "native_mtp_ngram_proposer_candidates_examined")?,
        proposer_appended_tokens: count(t, "native_mtp_ngram_proposer_appended_tokens")?,
        proposer_rebuilds: count(t, "native_mtp_ngram_proposer_rebuilds")?,
        proposer_sync_us: count(t, "native_mtp_ngram_proposer_sync_us")?,
        proposer_lookup_us: count(t, "native_mtp_ngram_proposer_lookup_us")?,
        finish_reason: reason.into(),
        output_sha256: super::evidence::digest(content.as_bytes()),
    };
    if !s.wall_tok_s.is_finite()
        || s.draft_accepted > s.draft_n
        || s.ngram_accepted_tokens > s.ngram_tokens
        || s.proposer_hits > s.proposer_attempts
    {
        return Err("contradictory suffix timing counters".into());
    }
    Ok(s)
}

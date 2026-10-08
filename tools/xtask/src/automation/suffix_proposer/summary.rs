use super::sample::Sample;
use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn number(n: u64) -> f64 {
    crate::automation::openai_exchange::stream::number(n)
}
fn sum(g: &[&Sample], f: impl Fn(&Sample) -> u64) -> DynResult<u64> {
    g.iter().try_fold(0u64, |n, s| {
        n.checked_add(f(s))
            .ok_or_else(|| "suffix counter sum overflow".into())
    })
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        v[m - 1] / 2.0 + v[m] / 2.0
    } else {
        v[m]
    }
}
pub(super) fn summarize(samples: &[Sample], baseline: &str) -> DynResult<Value> {
    let mut groups = BTreeMap::<(&str, &str), Vec<&Sample>>::new();
    for s in samples {
        groups.entry((&s.arm, &s.workload)).or_default().push(s);
    }
    let mut rows = Vec::new();
    for ((arm, workload), g) in groups {
        let hybrid = sum(&g, |s| s.ngram_tokens)?;
        let proposal = if hybrid > 0 {
            hybrid
        } else {
            sum(&g, |s| s.draft_n)?
        };
        let accepted = if hybrid > 0 {
            sum(&g, |s| s.ngram_accepted_tokens)?
        } else {
            sum(&g, |s| s.draft_accepted)?
        };
        let n = number(g.len() as u64);
        let mean = g.iter().map(|s| s.server_tok_s / n).sum::<f64>();
        let variance = if g.len() > 1 {
            g.iter()
                .map(|s| (s.server_tok_s - mean).powi(2) / (n - 1.0))
                .sum::<f64>()
        } else {
            0.0
        };
        if !variance.is_finite() {
            return Err("suffix derived variance overflow".into());
        }
        rows.push(json!({"arm":arm,"workload":workload,"samples":g.len(),"wall_tok_s_median":median(g.iter().map(|s|s.wall_tok_s).collect()),"server_tok_s_median":median(g.iter().map(|s|s.server_tok_s).collect()),"server_tok_s_stdev":variance.sqrt(),"proposal_mode":if hybrid>0{"mtp-hybrid"}else{"standalone"},"ngram_tokens":proposal,"ngram_accepted_tokens":accepted,"ngram_acceptance":if proposal>0{number(accepted)/number(proposal)}else{0.0},"proposer_match_length_max":g.iter().map(|s|s.proposer_match_length_max).max(),"proposer_candidates_examined":sum(&g,|s|s.proposer_candidates_examined)?,"proposer_lookup_us":sum(&g,|s|s.proposer_lookup_us)?}));
    }
    let mut mismatches = Vec::new();
    for s in samples {
        if let Some(b) = samples
            .iter()
            .find(|b| b.arm == baseline && b.workload == s.workload && b.run == s.run)
            && b.output_sha256 != s.output_sha256
        {
            mismatches.push(json!({"arm":s.arm,"workload":s.workload,"run":s.run,"baseline_sha256":b.output_sha256,"output_sha256":s.output_sha256}));
        }
    }
    Ok(json!({"rows":rows,"output_hash_mismatches":mismatches}))
}
pub(super) fn activation(samples: &[Sample], required: Option<&str>) -> DynResult<()> {
    let Some(required) = required else {
        return Ok(());
    };
    let rows: Vec<_> = samples.iter().filter(|s| s.arm == required).collect();
    if rows.is_empty()
        || rows.iter().any(|s| s.ngram_proposer != required)
        || !rows.iter().any(|s| {
            if matches!(s.ngram_proposer.as_str(), "cache" | "suffix") {
                s.proposer_hits > 0
            } else {
                s.draft_n > 0
            }
        })
    {
        return Err("required suffix source/actual proposal activation absent".into());
    }
    Ok(())
}
pub(super) fn markdown(summary: &Value) -> String {
    let mut out = String::from(
        "# Suffix proposer benchmark\n\n| Arm | Workload | N | Wall tok/s median | Server tok/s median | N-gram acceptance | Max match |\n|---|---|---:|---:|---:|---:|---:|\n",
    );
    if let Some(rows) = summary["rows"].as_array() {
        for r in rows {
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {:.2} | {:.3} | {} |\n",
                r["arm"].as_str().unwrap_or(""),
                r["workload"].as_str().unwrap_or(""),
                r["samples"],
                r["wall_tok_s_median"].as_f64().unwrap_or(0.0),
                r["server_tok_s_median"].as_f64().unwrap_or(0.0),
                r["ngram_acceptance"].as_f64().unwrap_or(0.0),
                r["proposer_match_length_max"]
            ));
        }
    }
    out.push_str(&format!(
        "\nOutput hash mismatches: {}\n",
        summary["output_hash_mismatches"]
            .as_array()
            .map_or(0, Vec::len)
    ));
    out
}

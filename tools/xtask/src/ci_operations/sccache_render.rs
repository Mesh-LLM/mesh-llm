//! The bytes `capture.py` writes: `json.dumps(evidence, indent=2,
//! sort_keys=True)`, the appended `--github-output` lines, and the stdout
//! summaries, with floats rendered as Python `repr`.

use crate::ci_operations::sccache_evidence::{
    Assessment, Counters, OUTPUT_COUNTERS, REQUIRED_COUNT_MAPS, REQUIRED_COUNTERS,
};
use crate::prepared_input::python_value::float_repr;
use std::fmt::Write as _;

/// `str(value)` / f-string of an optional float (`None` when absent).
pub(crate) fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "None".to_owned(), float_repr)
}

fn json_float(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_owned(), float_repr)
}

/// `json.dumps(evidence, indent=2, sort_keys=True) + "\n"`.
pub(crate) fn evidence_text(counters: &Counters, assessment: &Assessment) -> String {
    let mut text = String::from("{\n  \"assessment\": {\n");
    let _ = write!(
        text,
        "    \"cache_requests\": {},\n    \"classification\": \"{}\",\n    \
         \"expectation\": \"{}\",\n    \"hit_rate\": {},\n    \
         \"minimum_hit_rate\": {},\n    \"passed\": {}\n  }},\n",
        assessment.requests,
        assessment.classification,
        assessment.expectation,
        json_float(assessment.hit_rate),
        json_float(Some(assessment.minimum_hit_rate)),
        assessment.passed
    );
    text.push_str("  \"schema\": \"mesh-llm.sccache-stats\",\n  \"schema_version\": 2,\n");
    text.push_str("  \"stats\": {\n");
    let mut names: Vec<&str> = REQUIRED_COUNTERS
        .iter()
        .chain(REQUIRED_COUNT_MAPS.iter())
        .copied()
        .collect();
    names.sort_unstable();
    let lines: Vec<String> = names
        .iter()
        .map(|name| {
            let value = counters.get(name);
            if REQUIRED_COUNT_MAPS.contains(name) {
                format!(
                    "    \"{name}\": {{\n      \"counts\": {{\n        \"total\": {value}\n      }}\n    }}"
                )
            } else {
                format!("    \"{name}\": {value}")
            }
        })
        .collect();
    text.push_str(&lines.join(",\n"));
    text.push_str("\n  }\n}\n");
    text
}

/// `write_github_outputs`' appended lines.
pub(crate) fn github_output_text(
    stats_file: &str,
    counters: &Counters,
    assessment: &Assessment,
) -> String {
    let mut text = format!("stats_file={stats_file}\n");
    for name in OUTPUT_COUNTERS {
        let _ = writeln!(text, "{name}={}", counters.get(name));
    }
    let rate = assessment.hit_rate.map_or_else(String::new, float_repr);
    let _ = writeln!(text, "hit_rate={rate}");
    let _ = writeln!(text, "cache_classification={}", assessment.classification);
    let _ = writeln!(text, "cache_passed={}", assessment.passed);
    text
}

/// The two `print` summaries and the zero-request workflow warning.
pub(crate) fn summary_text(counters: &Counters, assessment: &Assessment) -> String {
    let mut text = format!(
        "sccache evidence: requests={} executed={} hits={} misses={} writes={}\n",
        counters.get("compile_requests"),
        counters.get("requests_executed"),
        counters.get("cache_hits"),
        counters.get("cache_misses"),
        counters.get("cache_writes")
    );
    let _ = writeln!(
        text,
        "sccache assessment: expectation={} classification={} hit_rate={}",
        assessment.expectation,
        assessment.classification,
        optional_float(assessment.hit_rate)
    );
    if counters.get("compile_requests") == 0 {
        text.push_str(
            "::warning title=sccache reported zero compile requests::Check RUSTC_WRAPPER \
             wiring unless this job fully reused its restored target cache.\n",
        );
    }
    text
}

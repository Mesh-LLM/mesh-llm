use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    rows: Vec<Row>,
    prompt_token_range: Option<[u64; 2]>,
    min_cache_pct: Option<f64>,
    #[serde(default)]
    require_output_match: bool,
    max_ttft_regression_pct: Option<f64>,
}

#[derive(Deserialize)]
struct Row {
    label: String,
    concurrency: usize,
    failed_requests: Option<u64>,
    prompt_tokens_min: Option<u64>,
    prompt_tokens_max: Option<u64>,
    cache_pct: Option<f64>,
    #[serde(default)]
    content_identity_known: bool,
    #[serde(default)]
    delta_comparable: bool,
    ttft_p50_seconds_delta_pct: Option<f64>,
}

#[derive(Serialize)]
struct Check {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Serialize)]
struct Gates {
    evaluated: bool,
    passed: Option<bool>,
    checks: Vec<Check>,
}

fn evaluate(input: &Input) -> DynResult<Gates> {
    if input
        .prompt_token_range
        .is_some_and(|[minimum, maximum]| minimum > maximum)
        || input
            .min_cache_pct
            .is_some_and(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
        || input
            .max_ttft_regression_pct
            .is_some_and(|value| !value.is_finite())
    {
        return Err("invalid replay acceptance budget".into());
    }
    let evaluated = input.prompt_token_range.is_some()
        || input.min_cache_pct.is_some()
        || input.require_output_match
        || input.max_ttft_regression_pct.is_some();
    let mut checks = Vec::new();
    if !evaluated {
        return Ok(Gates {
            evaluated: false,
            passed: None,
            checks,
        });
    }
    let mut record = |name: String, passed: bool, detail: String| {
        checks.push(Check {
            name,
            passed,
            detail,
        })
    };
    if input.rows.is_empty() {
        record(
            "measured-rows".into(),
            false,
            "observed=0 required>=1".into(),
        );
    }
    for row in &input.rows {
        let cell = format!("{}/c{}", row.label, row.concurrency);
        record(
            format!("failed-requests:{cell}"),
            row.failed_requests == Some(0),
            format!("observed={} required=0", integer(row.failed_requests)),
        );
        if let Some([minimum, maximum]) = input.prompt_token_range {
            let passed = row.prompt_tokens_min.is_some_and(|value| value >= minimum)
                && row.prompt_tokens_max.is_some_and(|value| value <= maximum);
            record(
                format!("prompt-token-range:{cell}"),
                passed,
                format!(
                    "observed={}:{} required={minimum}:{maximum}",
                    integer(row.prompt_tokens_min),
                    integer(row.prompt_tokens_max)
                ),
            );
        }
        if let Some(minimum) = input.min_cache_pct {
            record(
                format!("cached-prompt:{cell}"),
                row.cache_pct
                    .is_some_and(|value| value.is_finite() && value >= minimum),
                format!("observed={} required>={minimum}", number(row.cache_pct)),
            );
        }
        if input.require_output_match {
            let passed = row.content_identity_known && row.delta_comparable;
            record(format!("deterministic-output:{cell}"),passed,if passed {
                "content hashes and failed request identities match baseline"
            } else { "content hashes are missing or differ, or failed request identities differ from baseline" }.into());
        }
        if let Some(maximum) = input.max_ttft_regression_pct {
            let passed = row
                .ttft_p50_seconds_delta_pct
                .is_some_and(|value| value.is_finite() && value <= maximum);
            let detail = match row.ttft_p50_seconds_delta_pct {
                Some(value) => format!("observed={value:.3}% allowed<={maximum:.3}%"),
                None => format!("observed=unavailable allowed<={maximum:.3}%"),
            };
            record(format!("ttft-regression:{cell}"), passed, detail);
        }
    }
    Ok(Gates {
        evaluated: true,
        passed: Some(checks.iter().all(|check| check.passed)),
        checks,
    })
}

fn integer(value: Option<u64>) -> String {
    value.map_or_else(|| "None".into(), |value| value.to_string())
}
fn number(value: Option<f64>) -> String {
    value.map_or_else(|| "None".into(), |value| value.to_string())
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix acceptance-gates --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    let gates = evaluate(&input)?;
    crate::command::write_json_file(
        std::path::Path::new(parsed.last("--output").ok_or("missing --output")?),
        &gates,
    )?;
    if gates.passed == Some(false) {
        Err("replay acceptance gates failed; report retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unconfigured_gates_remain_unevaluated() {
        let input: Input = serde_json::from_value(serde_json::json!({"rows":[]})).unwrap();
        let gates = evaluate(&input).unwrap();
        assert!(!gates.evaluated);
        assert_eq!(gates.passed, None);
    }
    #[test]
    fn configured_gates_fail_without_measurements() {
        let input: Input =
            serde_json::from_value(serde_json::json!({"rows":[],"require_output_match":true}))
                .unwrap();
        let gates = evaluate(&input).unwrap();
        assert_eq!(gates.passed, Some(false));
        assert_eq!(gates.checks.len(), 1);
    }
}

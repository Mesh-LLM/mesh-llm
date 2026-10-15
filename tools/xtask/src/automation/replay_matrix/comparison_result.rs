//! Publish only complete comparisons; preserve engine/case identity in artifacts.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::path::Path;

pub(super) fn write(output: &Path, cases: &[&str], build: &Value) -> DynResult<()> {
    let mut comparison = Vec::new();
    for case in cases {
        let run: Value =
            serde_json::from_slice(&std::fs::read(output.join(case).join("run.json"))?)?;
        if run["gates"]["passed"] != true {
            return Err("incomplete comparison cannot publish benchmark metrics".into());
        }
        comparison.push(json!({"case":case,"results":run["results"]}));
    }
    crate::command::write_json_file(&output.join("comparison.json"), &comparison)?;
    if let Some(path) = std::env::var_os("JIANYANG_RESULT_PATH") {
        let point = point(&comparison, build)?;
        crate::command::write_json_file(
            Path::new(&path),
            &json!({"provenance":{"stack_commit":build["commit"],
            "runtime_flags":{"thinking":false,"output_limit":"server-default",
            "harness_commit":std::env::var("JIANYANG_COMMIT").ok()}},
            "points":[point],"notes":["Point is combined Mesh end-to-end throughput across the two fixed 50-turn cases; not native decode throughput.",
            "Paired upstream/Mesh case metrics, stream-span proxy, terminal overhead, TTFT, cache and integrity evidence are in recorded-replay/comparison.json."]}),
        )?;
    }
    Ok(())
}

fn point(comparison: &[Value], build: &Value) -> DynResult<Value> {
    let cells: Vec<_> = comparison
        .iter()
        .flat_map(|case| case["results"].as_array().into_iter().flatten())
        .filter(|arm| arm["label"] == build["label"])
        .flat_map(|arm| arm["cells"].as_array().into_iter().flatten())
        .collect();
    if cells.len() != 2
        || cells.iter().any(|c| {
            c["requests"] != 50
                || c["successful_requests"] != 50
                || c["output_incomplete_requests"] != 0
        })
    {
        return Err("publication requires both complete 50-turn Mesh cells".into());
    }
    let sum = |field: &str| -> DynResult<f64> {
        cells
            .iter()
            .map(|c| {
                c[field]
                    .as_f64()
                    .ok_or_else(|| format!("missing {field}").into())
            })
            .sum()
    };
    let duration = sum("workload_window_seconds")?;
    if duration <= 0.0 {
        return Err("missing workload duration".into());
    }
    Ok(
        json!({"concurrency":1,"requests_completed":100,"requests_failed":0,"duration_s":duration,
        "tokens":{"input":sum("prompt_tokens")?,"output":sum("completion_tokens")?},
        "throughput":{"output_tok_s":sum("completion_tokens")?/duration}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_pools_fixed_cases_by_time_and_refuses_partial_cells() {
        let cell = |seconds, tokens| json!({"requests":50,"successful_requests":50,"output_incomplete_requests":0,"workload_window_seconds":seconds,"prompt_tokens":1000,"completion_tokens":tokens});
        let mut cases = vec![
            json!({"results":[{"label":"mesh","cells":[cell(2,40)]},{"label":"llama","cells":[cell(1,1000)]}]}),
            json!({"results":[{"label":"mesh","cells":[cell(3,60)]}]}),
        ];
        let build = json!({"label":"mesh"});
        let result = point(&cases, &build).unwrap();
        assert_eq!(result["throughput"]["output_tok_s"], 20.0);
        cases[1]["results"][0]["cells"][0]["requests"] = 2.into();
        assert!(point(&cases, &build).is_err());
    }
}

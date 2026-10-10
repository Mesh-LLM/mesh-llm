//! Source-bound competitive matrix planning; execution and reports are separate owners.
use crate::{
    command::DynResult,
    repository::{check_args::Grammar, check_report::CheckReport},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::OpenOptions, io::Read, path::Path};
#[path = "competitive_config.rs"]
pub(super) mod config;
const USAGE: &str = "cargo xtool automation replay-matrix competitive-plan --config PATH [--platform cuda|metal|rocm]... [--model KEY]... [--workload synthetic|thoughtworks]...";
const GRAMMAR: Grammar = Grammar {
    usage: USAGE,
    values: &["--config", "--platform", "--model", "--workload"],
    flags: &["--help"],
};
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    report(args).emit()
}
fn report(args: &[String]) -> CheckReport {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{USAGE}\n"));
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments");
    }
    let result = (|| {
        let path = Path::new(parsed.last("--config").ok_or("missing --config")?);
        let bytes = read(path)?;
        let document: Value = serde_json::from_slice(&bytes)?;
        config::admit(&document)?;
        let platforms = choices(
            parsed.all("--platform"),
            &["cuda", "metal"],
            &["cuda", "metal", "rocm"],
        )?;
        let workloads = choices(
            parsed.all("--workload"),
            &["synthetic", "thoughtworks"],
            &["synthetic", "thoughtworks"],
        )?;
        let output = build(
            &document,
            &bytes,
            &platforms,
            &parsed.all("--model"),
            &workloads,
        )?;
        Ok::<_, Box<dyn std::error::Error>>(format!("{}\n", serde_json::to_string_pretty(&output)?))
    })();
    result.map_or_else(
        |error| CheckReport::failure(String::new(), format!("{error}\n")),
        CheckReport::success,
    )
}
fn read(path: &Path) -> DynResult<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("competitive config must be a regular file".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened competitive config must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("competitive config exceeds eight MiB".into());
    }
    Ok(bytes)
}
fn choices<'a>(
    values: Vec<&'a str>,
    defaults: &[&'a str],
    allowed: &[&str],
) -> DynResult<Vec<&'a str>> {
    let values = if values.is_empty() {
        defaults.to_vec()
    } else {
        values
    };
    let mut seen = BTreeSet::new();
    if values
        .iter()
        .any(|value| !allowed.contains(value) || !seen.insert(*value))
    {
        return Err("unknown or duplicate competitive selection".into());
    }
    Ok(values)
}
pub(super) fn build(
    document: &Value,
    bytes: &[u8],
    platforms: &[&str],
    keys: &[&str],
    workloads: &[&str],
) -> DynResult<Value> {
    let all = document["models"].as_array().ok_or("models missing")?;
    for key in keys {
        if !all.iter().any(|model| model["key"].as_str() == Some(key)) {
            return Err(format!("unknown competitive model: {key}").into());
        }
    }
    let selected: Vec<_> = all
        .iter()
        .filter(|model| {
            keys.is_empty() || model["key"].as_str().is_some_and(|key| keys.contains(&key))
        })
        .collect();
    let output_count = document["synthetic"]["output_tokens"]
        .as_array()
        .ok_or("output ladder missing")?
        .len();
    let synthetic_count = if workloads.contains(&"synthetic") {
        output_count
            .checked_mul(config::LADDER.len())
            .and_then(|count| count.checked_mul(2))
            .ok_or("matrix size overflow")?
    } else {
        0
    };
    let trace_count = if workloads.contains(&"thoughtworks") {
        config::LADDER.len() * 2
    } else {
        0
    };
    let cell_count = synthetic_count
        .checked_add(trace_count)
        .and_then(|count| count.checked_mul(selected.len()))
        .and_then(|count| count.checked_mul(platforms.len()))
        .ok_or("matrix size overflow")?;
    if cell_count > 100_000 {
        return Err("competitive matrix exceeds 100000 cells".into());
    }
    let mut cells = Vec::with_capacity(cell_count);
    for platform in platforms {
        for model in &selected {
            if workloads.contains(&"synthetic") {
                synthetic(&mut cells, document, platform, model)?;
            }
            if workloads.contains(&"thoughtworks") {
                trace(&mut cells, document, platform, model)?;
            }
        }
    }
    Ok(
        json!({"schema_version":2,"config_hash_kind":"source_bytes_sha256","config_sha256":hex::encode(Sha256::digest(bytes)),"platforms":platforms,"models":selected.iter().map(|model| &model["key"]).collect::<Vec<_>>(),"workloads":workloads,"arms":["llama","mesh"],"cell_count":cells.len(),"cells":cells}),
    )
}
fn synthetic(
    cells: &mut Vec<Value>,
    document: &Value,
    platform: &str,
    model: &Value,
) -> DynResult<()> {
    for arm in ["llama", "mesh"] {
        for output in document["synthetic"]["output_tokens"]
            .as_array()
            .ok_or("output ladder missing")?
        {
            for concurrency in config::LADDER {
                cells.push(json!({"platform":platform,"model":model["key"],"workload":"synthetic","arm":arm,"prompt_tokens":document["synthetic"]["prompt_tokens"],"output_tokens":output,"concurrency":concurrency}));
            }
        }
    }
    Ok(())
}
fn trace(cells: &mut Vec<Value>, document: &Value, platform: &str, model: &Value) -> DynResult<()> {
    let trace = &document["thoughtworks"];
    let positive = config::positive;
    let families = positive(&trace["selection"]["families"], "families")?;
    let available = families
        .checked_mul(positive(
            &trace["selection"]["requests_per_family"],
            "repeats",
        )?)
        .ok_or("prompt count overflow")?;
    let minimum = positive(&trace["minimum_prompts"], "minimum prompts")?;
    let context = model
        .get("thoughtworks_context_size")
        .unwrap_or(&trace["context_size"]);
    let lanes = model
        .get("thoughtworks_active_lanes")
        .unwrap_or(&trace["active_lanes"]);
    for (index, concurrency) in config::LADDER.into_iter().enumerate() {
        let target = minimum.max(concurrency);
        let rounded = target
            .checked_add(concurrency - 1)
            .ok_or("prompt wave overflow")?
            / concurrency
            * concurrency;
        let count = rounded.min(available);
        let arms = if index % 2 == 0 {
            ["llama", "mesh"]
        } else {
            ["mesh", "llama"]
        };
        for arm in arms {
            cells.push(json!({"platform":platform,"model":model["key"],"workload":"thoughtworks","arm":arm,"output_tokens":trace["output_tokens"],"context_size":context,"active_lanes":lanes,"concurrency":concurrency,"prompt_count":count}));
        }
    }
    Ok(())
}
// Append to cumulative native plan owner before its cfg(test) module.
pub(super) fn admit_cell(document: &Value, bytes: &[u8], cell: &Value) -> DynResult<()> {
    config::admit(document)?;
    let platform = cell["platform"].as_str().ok_or("cell platform missing")?;
    let model = cell["model"].as_str().ok_or("cell model missing")?;
    let workload = cell["workload"].as_str().ok_or("cell workload missing")?;
    let platforms = choices(vec![platform], &[], &["cuda", "metal", "rocm"])?;
    let workloads = choices(vec![workload], &[], &["synthetic", "thoughtworks"])?;
    let arm = cell["arm"].as_str().ok_or("cell arm missing")?;
    let mut candidate = cell.clone();
    candidate["arm"] = match arm {
        "mesh-adaptive" => "mesh",
        "vllm" | "sglang" => "llama",
        "mesh" | "llama" => arm,
        _ => return Err("unknown competitive arm".into()),
    }
    .into();
    let plan = build(document, bytes, &platforms, &[model], &workloads)?;
    if !plan["cells"]
        .as_array()
        .ok_or("planned cells missing")?
        .contains(&candidate)
    {
        return Err("competitive cell differs from admitted native plan".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "competitive_plan_tests.rs"]
mod tests;

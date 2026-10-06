//! Report only correlated complete cells; parity gates labels, never synthetic performance proof.
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
pub(super) fn write(
    root: &Path,
    cells: &[Value],
    config: &str,
    deadline: Instant,
    partial: bool,
) -> DynResult<()> {
    let rows = load_rows(root, cells, config, partial)?;
    admit_outputs(root, &rows)?;
    let gates = gates(&rows);
    if Instant::now() >= deadline {
        return Err("report generation deadline reached".into());
    }
    let summary = root.join("summary");
    super::competitive_matrix::ensure_directory(root, &summary.join("charts"))?;
    for workload in ["synthetic", "thoughtworks"] {
        let selected: Vec<_> = rows
            .iter()
            .filter(|row| row["cell"]["workload"] == workload)
            .collect();
        super::competitive_report_output::replace(
            root,
            &summary.join(format!("{workload}.csv")),
            csv(&selected).as_bytes(),
        )?;
    }
    super::competitive_report_output::json(root, &summary.join("parity.json"), &gates)?;
    let mut parity_csv = String::from(
        "platform,model,arm,output_tokens,concurrency,passed
",
    );
    for gate in &gates {
        let cell = &gate["cell"];
        parity_csv.push_str(&format!(
            "{},{},{},{},{},{}
",
            escape(&cell["platform"]),
            escape(&cell["model"]),
            escape(&cell["arm"]),
            cell["output_tokens"],
            cell["concurrency"],
            gate["passed"]
        ));
    }
    super::competitive_report_output::replace(
        root,
        &summary.join("parity.csv"),
        parity_csv.as_bytes(),
    )?;
    let mut markdown = String::from(
        "# Skippy competitive benchmark

Rows below are measured local artifacts. A comparison is qualified only when exact deterministic continuation parity passes for the same model, platform and offered concurrency. Source pins and fixture results do not certify actual model performance.

| Platform | Model | Workload | Arm | Concurrency | Output tokens | Tokens/s | Parity |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
",
    );
    for row in &rows {
        let cell = &row["cell"];
        let gate = find_gate(&gates, cell);
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {} |
",
            escape(&cell["platform"]),
            escape(&cell["model"]),
            escape(&cell["workload"]),
            escape(&cell["arm"]),
            cell["concurrency"],
            cell["output_tokens"],
            row["throughput"].as_f64().unwrap_or(0.0),
            if cell["arm"] == "llama" {
                "REFERENCE"
            } else if gate {
                "PASS"
            } else {
                "PENDING / FAIL"
            }
        ));
    }
    markdown.push_str("
Capacity policy is retained per row in report.json. Optional engines with default paged capacity must be compared separately from aggregate KV-matched rows; no cross-platform/family throughput aggregation is performed.
");
    super::competitive_report_output::replace(
        root,
        &summary.join("REPORT.md"),
        markdown.as_bytes(),
    )?;
    super::competitive_report_output::json(
        root,
        &summary.join("report.json"),
        &json!({"schema_version":1,"config_sha256":config,"rows":rows,"parity":gates,"report_complete":rows.len()==cells.len(),"requested_cells":cells.len()}),
    )?;
    charts(root, &summary, &rows)?;
    inventory(root, deadline)
}
fn read(path: &Path) -> DynResult<Value> {
    serde_json::from_slice(&super::competitive_cell::read(path, 64 * 1024 * 1024)?)
        .map_err(Into::into)
}
fn escape(value: &Value) -> String {
    super::report_escape::cell(value.as_str().unwrap_or("invalid"))
}
fn csv(rows: &[&Value]) -> String {
    let mut output = String::from(
        "platform,model,workload,arm,output_tokens,concurrency,throughput,complete,launch_sha256
",
    );
    for row in rows {
        let cell = &row["cell"];
        let values = [
            cell["platform"].as_str().unwrap_or(""),
            cell["model"].as_str().unwrap_or(""),
            cell["workload"].as_str().unwrap_or(""),
            cell["arm"].as_str().unwrap_or(""),
        ];
        output.push_str(
            &values
                .iter()
                .map(|value| format!("\"{}\"", value.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push_str(&format!(
            ",{},{},{},true,{}
",
            cell["output_tokens"],
            cell["concurrency"],
            row["throughput"],
            row["launch_sha256"].as_str().unwrap_or("")
        ));
    }
    output
}
fn gates(rows: &[Value]) -> Vec<Value> {
    let mut gates = Vec::new();
    for row in rows
        .iter()
        .filter(|row| row["cell"]["workload"] == "synthetic" && row["cell"]["arm"] != "llama")
    {
        let cell = &row["cell"];
        let reference = rows.iter().find(|reference| {
            reference["cell"]["workload"] == "synthetic"
                && reference["cell"]["arm"] == "llama"
                && ["platform", "model", "concurrency", "output_tokens"]
                    .iter()
                    .all(|field| reference["cell"][field] == cell[field])
        });
        let passed =
            reference.is_some_and(|reference| parity_equal(&reference["parity"], &row["parity"]));
        gates.push(json!({"cell":cell,"passed":passed,"scope":"exact_structured_continuation_digest","proof":"paired_completed_local_probe"}));
    }
    gates
}
fn parity_equal(reference: &Value, candidate: &Value) -> bool {
    if reference["passed"] != true || candidate["passed"] != true {
        return false;
    }
    let Some(left) = reference["results"].as_array() else {
        return false;
    };
    let Some(right) = candidate["results"].as_array() else {
        return false;
    };
    !left.is_empty()
        && left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left["valid"] == true
                && right["valid"] == true
                && left["request_index"] == right["request_index"]
                && left["content_sha256"].as_str().is_some_and(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                && left["content_sha256"] == right["content_sha256"]
                && left["completion_tokens"] == 32
                && right["completion_tokens"] == 32
        })
}
fn find_gate(gates: &[Value], cell: &Value) -> bool {
    if cell["arm"] == "llama" {
        return false;
    }
    gates.iter().any(|gate| {
        gate["passed"] == true
            && [
                "platform",
                "model",
                "workload",
                "arm",
                "concurrency",
                "output_tokens",
            ]
            .iter()
            .all(|field| gate["cell"][field] == cell[field])
    })
}
fn charts(root: &Path, summary: &Path, rows: &[Value]) -> DynResult<()> {
    let mut groups = std::collections::BTreeMap::<String, Vec<&Value>>::new();
    for row in rows {
        let cell = &row["cell"];
        let key = chart_key(cell)?;
        groups.entry(key).or_default().push(row);
    }
    for (key, group) in groups {
        let peak = group
            .iter()
            .filter_map(|row| row["throughput"].as_f64())
            .fold(1.0_f64, f64::max);
        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"420\" viewBox=\"0 0 800 420\"><title>{}</title><rect width=\"800\" height=\"420\" fill=\"white\"/><text x=\"40\" y=\"30\">{}</text>",
            super::report_escape::xml(&key),
            super::report_escape::xml(&key)
        );
        for row in group {
            let cell = &row["cell"];
            let concurrency = cell["concurrency"].as_u64().ok_or("concurrency")?;
            let x = 40.0 + (concurrency as f64).log2() * 85.0;
            let y = 380.0 - row["throughput"].as_f64().ok_or("throughput")? / peak * 320.0;
            svg.push_str(&format!("<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"4\"><title>{} concurrency{} {:.3}tokens/s</title></circle>",super::report_escape::xml(cell["arm"].as_str().ok_or("arm")?),concurrency,row["throughput"].as_f64().unwrap_or(0.0)));
        }
        svg.push_str(
            "</svg>
",
        );
        super::competitive_report_output::replace(
            root,
            &summary.join("charts").join(format!("{key}-throughput.svg")),
            svg.as_bytes(),
        )?;
    }
    Ok(())
}
fn inventory(root: &Path, deadline: Instant) -> DynResult<()> {
    let mut pending = vec![(root.to_path_buf(), 0)];
    let mut files = Vec::<PathBuf>::new();
    let mut seen = 0;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 32 {
            return Err("competitive artifact depth exceeds32".into());
        }
        for entry in std::fs::read_dir(directory)? {
            if Instant::now() >= deadline {
                return Err("artifact inventory deadline reached".into());
            }
            seen += 1;
            if seen > 2_000_000 {
                return Err("competitive artifact entry bound exceeded".into());
            }
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                return Err("competitive output inventory refuses symlinks".into());
            }
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file()
                && entry.file_name() != "artifact-sha256.txt"
                && entry.file_name() != ".matrix-active"
            {
                files.push(entry.path());
            } else if !kind.is_file() {
                return Err("competitive inventory refuses special files".into());
            }
        }
    }
    files.sort();
    let mut output = String::new();
    for path in files {
        if Instant::now() >= deadline {
            return Err("artifact digest deadline reached".into());
        }
        let hash = crate::product::digest::file_sha256(&path).map_err(|error| error.error)?;
        output.push_str(&format!(
            "{hash}  {}
",
            path.strip_prefix(root)?
                .to_str()
                .ok_or("artifact Unicode")?
        ));
    }
    super::competitive_report_output::replace(
        root,
        &root.join("artifact-sha256.txt"),
        output.as_bytes(),
    )?;
    Ok(())
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-report --artifact PATH",
        values: &["--artifact"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!(
            "{}
",
            GRAMMAR.usage
        ))
        .emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let root = Path::new(parsed.last("--artifact").ok_or("artifact")?);
    if !root.is_absolute() || !std::fs::symlink_metadata(root)?.is_dir() {
        return Err("competitive report requires an absolute regular artifact directory".into());
    }
    let plan = read(&root.join("matrix-plan.json"))?;
    if plan["schema_version"] != 2 || plan["scope"] != "competitive_native_matrix" {
        return Err("competitive report requires native matrix plan".into());
    }
    let cells = plan["cells"].as_array().ok_or("planned cells")?;
    if cells.len() > 100000 {
        return Err("oversized report plan".into());
    }
    write(
        root,
        cells,
        plan["config_sha256"].as_str().ok_or("config SHA")?,
        Instant::now() + std::time::Duration::from_secs(600),
        true,
    )
}

fn load_rows(root: &Path, cells: &[Value], config: &str, partial: bool) -> DynResult<Vec<Value>> {
    let plan = read(&root.join("matrix-plan.json"))?;
    let config_bytes =
        super::competitive_cell::read(&root.join("benchmark-config.source.json"), 8 * 1024 * 1024)?;
    use sha2::Digest as _;
    if hex::encode(sha2::Sha256::digest(&config_bytes)) != config {
        return Err("report source snapshot hash differs".into());
    }
    let source: Value = serde_json::from_slice(&config_bytes)?;
    admit_archive(root, &plan, &source, &config_bytes, cells, config)?;
    let models: Vec<super::competitive_roster::Model> =
        serde_json::from_value(plan["prepared_inputs"].clone())?;
    let mut rows = Vec::new();
    for cell in cells {
        let directory = super::competitive_matrix::cell_directory(root, cell)?;
        super::competitive_matrix::existing_parents(root, &directory.join("worker"))?;
        if !directory.join("complete.json").exists() {
            if partial {
                continue;
            }
            return Err("requested report cell has no completion marker".into());
        }
        let launch = read(&directory.join("launch.json"))?;
        let model = models
            .iter()
            .find(|model| cell["model"] == model.key)
            .ok_or("report model not in prepared plan")?;
        let expected = super::competitive_resume::declared_provenance(&source, cell, model)?;
        if !super::competitive_resume::completed(&directory, cell, config, &expected)? {
            return Err("marked report cell is not correlated and complete".into());
        }
        let summary = read(&directory.join("worker/worker-summary.json"))?;
        let throughput = throughput(&directory, cell, &summary)?;
        let parity_path = directory.join("worker/parity.json");
        let parity = if parity_path.is_file() {
            Some(read(&parity_path)?)
        } else {
            None
        };
        rows.push(json!({"cell":cell,"throughput":throughput,"complete":true,"launch_sha256":crate::product::digest::file_sha256(&directory.join("launch.json")).map_err(|error|error.error)?,"capacity_policy":launch["capacity_policy"],"parity":parity}));
    }
    Ok(rows)
}
fn throughput(directory: &Path, cell: &Value, summary: &Value) -> DynResult<f64> {
    let throughput = if cell["workload"] == "synthetic" {
        read(&directory.join("worker/result.json"))?["benchmarks"][0]["tg_throughput"]["mean"]
            .as_f64()
    } else {
        let wall = summary["measured_wall_seconds"]
            .as_f64()
            .ok_or("measured wall")?;
        let tokens = summary["completion_tokens"]
            .as_u64()
            .ok_or("completion tokens")?;
        if wall > 0.0 {
            Some(tokens as f64 / wall)
        } else {
            None
        }
    };
    if !throughput.is_some_and(|value| value.is_finite() && value > 0.0) {
        return Err("complete report row lacks finite positive throughput".into());
    }
    throughput.ok_or_else(|| "missing throughput".into())
}

fn admit_archive(
    root: &Path,
    plan: &Value,
    source: &Value,
    bytes: &[u8],
    cells: &[Value],
    config: &str,
) -> DynResult<()> {
    if plan["schema_version"] != 2
        || plan["scope"] != "competitive_native_matrix"
        || plan["config_sha256"] != config
        || plan["cells"].as_array().map(Vec::as_slice) != Some(cells)
    {
        return Err("archive plan binding differs".into());
    }
    let selection = &plan["selection"];
    let input: super::competitive_matrix::Input = serde_json::from_value(
        json!({"config":root.join("benchmark-config.source.json"),"config_sha256":config,"source_context":plan["source_context"],"platform":plan["platform"],"models":plan["prepared_inputs"],"workloads":selection["workloads"],"optional_arms":selection["optional_arms"],"required_comparisons":selection["required_comparisons"],"adaptive":selection["adaptive"],"manifest":plan["manifest"],"benchy":plan["benchy"],"output":root,"timeout_seconds":600,"cell_timeout_seconds":20,"request_timeout_seconds":2,"resume":false,"force":false}),
    )?;
    let roster = super::competitive_roster::select_on(
        source,
        bytes,
        &input,
        plan["planner_linux"]
            .as_bool()
            .ok_or("archive planner platform")?,
    )?;
    if roster.cells != cells || roster.availability != plan["availability"] {
        return Err("archive cells differ from immutable configured/prepared roster".into());
    }
    for cell in cells {
        let path = super::competitive_matrix::cell_directory(root, cell)?;
        super::competitive_matrix::existing_parents(root, &path.join("worker"))?;
    }
    super::competitive_matrix::existing_parents(root, &root.join("summary/charts"))
}
fn chart_key(cell: &Value) -> DynResult<String> {
    Ok(format!(
        "{}-{}-{}-tg-{}",
        cell["platform"].as_str().ok_or("platform")?,
        cell["model"].as_str().ok_or("model")?,
        cell["workload"].as_str().ok_or("workload")?,
        cell["output_tokens"].as_u64().ok_or("output")?
    ))
}
fn admit_outputs(root: &Path, rows: &[Value]) -> DynResult<()> {
    for name in [
        "synthetic.csv",
        "thoughtworks.csv",
        "parity.csv",
        "parity.json",
        "report.json",
        "REPORT.md",
    ] {
        super::competitive_report_output::admit(root, &root.join("summary").join(name))?;
    }
    for row in rows {
        super::competitive_report_output::admit(
            root,
            &root
                .join("summary/charts")
                .join(format!("{}-throughput.svg", chart_key(&row["cell"])?)),
        )?;
    }
    super::competitive_report_output::admit(root, &root.join("artifact-sha256.txt"))
}

#[cfg(test)]
#[path = "competitive_report_tests.rs"]
mod tests;

//! Finite archived observations exercise the real report frontdoor; no engine/performance proof.
use super::super::{competitive_matrix, competitive_resume, competitive_roster};
use super::*;
use std::{collections::BTreeMap, fs};

struct Archive {
    root: PathBuf,
    source: Value,
    config: String,
    model: competitive_roster::Model,
    cells: Vec<Value>,
}
fn put(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn file_hash(path: &Path) -> String {
    crate::product::digest::file_sha256(path).unwrap_or_else(|failure| panic!("{}", failure.error))
}
fn backend(root: &Path) -> Value {
    let artifact = json!({"path":root.join("declared-inert-artifact"),"sha256":"a".repeat(64)});
    json!({"executable":artifact,"version_sha256":"b".repeat(64),"cwd":root,"runtime":artifact,"tokenizer":artifact,"hf_config":null,"comparison_model":null,"match_kv_capacity":false})
}
impl Archive {
    fn new(root: &Path) -> Self {
        let mut source: Value = serde_json::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../evals/skippy-competitive-benchmark.json"
        )))
        .unwrap();
        source["synthetic"]["output_tokens"] = json!([8]);
        for arm in ["vllm", "sglang"] {
            source["models"][0]["comparison_support"][arm] = json!({"available":true});
        }
        let bytes = serde_json::to_vec(&source).unwrap();
        fs::write(root.join("benchmark-config.source.json"), &bytes).unwrap();
        let config = file_hash(&root.join("benchmark-config.source.json"));
        let backends: BTreeMap<_, _> = ["llama", "mesh", "vllm", "sglang"]
            .into_iter()
            .map(|arm| (arm, backend(root)))
            .collect();
        let prepared = json!({"key":source["models"][0]["key"],"model":{"path":root.join("declared-model.gguf"),"sha256":source["models"][0]["sha256"]},"backends":backends});
        let input: competitive_matrix::Input = serde_json::from_value(json!({"config":root.join("benchmark-config.source.json"),"config_sha256":config,"platform":"cuda","models":[prepared],"workloads":["synthetic","thoughtworks"],"optional_arms":["vllm","sglang"],"required_comparisons":["vllm","sglang"],"adaptive":false,"manifest":null,"benchy":null,"output":root,"timeout_seconds":600,"cell_timeout_seconds":20,"request_timeout_seconds":2,"resume":false,"force":false})).unwrap();
        let roster = competitive_roster::select_on(&source, &bytes, &input, true).unwrap();
        put(
            &root.join("matrix-plan.json"),
            &json!({"schema_version":2,"scope":"competitive_native_matrix","config_sha256":config,"source_context":null,"platform":"cuda","planner_linux":true,"selection":{"workloads":input.workloads,"optional_arms":input.optional_arms,"required_comparisons":input.required_comparisons,"adaptive":false},"cells":roster.cells,"availability":roster.availability,"prepared_inputs":input.models,"manifest":null,"benchy":null}),
        );
        let model = serde_json::from_value(prepared).unwrap();
        Self {
            root: root.into(),
            source,
            config,
            model,
            cells: roster.cells,
        }
    }
    fn directory(&self, cell: &Value) -> PathBuf {
        competitive_matrix::cell_directory(&self.root, cell).unwrap()
    }
    fn cell(&self, cell: &Value, parity_digest: &str) {
        let directory = self.directory(cell);
        fs::create_dir_all(directory.join("worker")).unwrap();
        let provenance =
            competitive_resume::declared_provenance(&self.source, cell, &self.model).unwrap();
        let optional = matches!(cell["arm"].as_str(), Some("vllm" | "sglang"));
        put(
            &directory.join("launch.json"),
            &json!({"cell":cell,"config_sha256":self.config,"launch_provenance":provenance,"capacity_policy":{"mode":"declared-shared-context","comparison_kv_matched":!optional}}),
        );
        let mut summary = json!({"cell":cell,"config_sha256":self.config,"launch_provenance":provenance,"completed":true,"passed":true});
        if cell["workload"] == "synthetic" {
            self.synthetic(&directory, cell, parity_digest, &mut summary);
        } else {
            self.trace(&directory, cell, &mut summary);
        }
        put(&directory.join("worker/worker-summary.json"), &summary);
        put(
            &directory.join("lifecycle.json"),
            &json!({"infrastructure_clean":true,"worker_status":0,"scope":"finite archived correlation fixture, no executed server"}),
        );
        let marker = json!({"schema_version":2,"scope":"competitive_retained_cell","completed":true,"cell":cell,"config_sha256":self.config,"launch_sha256":file_hash(&directory.join("launch.json")),"worker_summary_sha256":file_hash(&directory.join("worker/worker-summary.json")),"lifecycle_sha256":file_hash(&directory.join("lifecycle.json"))});
        put(&directory.join("complete.json"), &marker);
        assert!(
            competitive_resume::completed(&directory, cell, &self.config, &provenance).unwrap()
        );
    }
    fn synthetic(&self, directory: &Path, cell: &Value, parity_digest: &str, summary: &mut Value) {
        let worker = directory.join("worker");
        put(
            &worker.join("result.json"),
            &json!({"benchmarks":[{"response_size":cell["output_tokens"],"tg_throughput":{"mean":rate(cell)}}]}),
        );
        fs::write(
            worker.join("progress.jsonl"),
            format!(
                "{}\n",
                json!({"type":"request_end","error":null,"total_tokens":cell["output_tokens"]})
            ),
        )
        .unwrap();
        put(
            &worker.join("parity.json"),
            &json!({"passed":true,"results":[{"request_index":0,"valid":true,"completion_tokens":32,"content_sha256":parity_digest}]}),
        );
        summary["completed_requests"] = json!(1);
        for (field, file) in [
            ("result_sha256", "result.json"),
            ("progress_sha256", "progress.jsonl"),
            ("parity_sha256", "parity.json"),
        ] {
            summary[field] = json!(file_hash(&worker.join(file)));
        }
    }
    fn trace(&self, directory: &Path, cell: &Value, summary: &mut Value) {
        let count = cell["prompt_count"].as_u64().unwrap();
        let tokens = cell["output_tokens"].as_u64().unwrap();
        let row = format!(
            "{}\n",
            json!({"phase":"measured","error":null,"completion_tokens":tokens})
        );
        let path = directory.join("worker/requests.jsonl");
        fs::write(&path, row.repeat(usize::try_from(count).unwrap())).unwrap();
        summary["requests_sha256"] = json!(file_hash(&path));
        summary["completion_tokens"] = json!(count * tokens);
        summary["measured_wall_seconds"] = json!((count * tokens) as f64 / rate(cell));
    }
    fn report(&self) {
        run(&["--artifact".into(), self.root.to_str().unwrap().into()]).unwrap();
    }
}
fn rate(cell: &Value) -> f64 {
    match (
        cell["workload"].as_str().unwrap(),
        cell["arm"].as_str().unwrap(),
    ) {
        ("synthetic", "llama") => 90.0,
        ("thoughtworks", "llama") => 80.0,
        (_, "mesh") => 100.0,
        ("synthetic", "vllm") => 110.0,
        ("thoughtworks", "vllm") => 115.0,
        ("synthetic", "sglang") => 120.0,
        ("thoughtworks", "sglang") => 125.0,
        _ => panic!("unreviewed finite fixture row"),
    }
}
fn csv_and_charts(archive: &Archive) {
    let model = &archive.model.key;
    for workload in ["synthetic", "thoughtworks"] {
        let csv = fs::read_to_string(archive.root.join(format!("summary/{workload}.csv"))).unwrap();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 37);
        for arm in ["llama", "mesh", "vllm", "sglang"] {
            let prefix = format!("\"cuda\",\"{model}\",\"{workload}\",\"{arm}\",");
            let rows: Vec<_> = lines
                .iter()
                .filter(|line| line.starts_with(&prefix))
                .collect();
            assert_eq!(rows.len(), 9);
            for row in rows {
                let fields: Vec<_> = row.split(',').collect();
                let actual: f64 = fields[6].parse().unwrap();
                assert!((actual - rate(&json!({"workload":workload,"arm":arm}))).abs() < 1e-9);
                assert_eq!(fields[7], "true");
                assert_eq!(fields[8].len(), 64);
            }
        }
        let tokens = if workload == "synthetic" {
            8
        } else {
            archive.source["thoughtworks"]["output_tokens"]
                .as_u64()
                .unwrap()
        };
        let svg = fs::read_to_string(archive.root.join(format!(
            "summary/charts/cuda-{model}-{workload}-tg-{tokens}-throughput.svg"
        )))
        .unwrap();
        for arm in ["llama", "mesh", "vllm", "sglang"] {
            assert!(svg.contains(&format!("{arm} concurrency1 ")));
            assert!(svg.contains(&format!("{arm} concurrency256 ")));
        }
    }
}
fn outputs(archive: &Archive) -> Value {
    let report = read(&archive.root.join("summary/report.json")).unwrap();
    let promotion = read(&archive.root.join("summary/promotion.json")).unwrap();
    assert_eq!(report["promotion"], promotion);
    let markdown = fs::read_to_string(archive.root.join("summary/REPORT.md")).unwrap();
    assert!(markdown.contains("Capacity policy is retained per row"));
    assert!(markdown.contains("no cross-platform/family throughput aggregation"));
    let inventory = fs::read_to_string(archive.root.join("artifact-sha256.txt")).unwrap();
    for file in [
        "summary/report.json",
        "summary/promotion.json",
        "summary/synthetic.csv",
        "summary/thoughtworks.csv",
    ] {
        assert!(
            inventory
                .lines()
                .any(|line| line == format!("{}  {file}", file_hash(&archive.root.join(file))))
        );
    }
    report
}
#[test]
fn competitive_archive_report_consumes_four_arm_multiworkload_rows_and_holds_unqualified_promotion()
{
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let archive = Archive::new(&root);
    assert_eq!(archive.cells.len(), 72);
    for cell in &archive.cells {
        archive.cell(cell, &"a".repeat(64));
    }
    archive.report();
    csv_and_charts(&archive);
    let report = outputs(&archive);
    assert_eq!(report["report_complete"], true);
    assert_eq!(report["rows"].as_array().unwrap().len(), 72);
    assert_eq!(report["promotion"][0]["winner"], "sglang");
    let markdown = fs::read_to_string(root.join("summary/REPORT.md")).unwrap();
    assert!(markdown.contains("+20.00 | +25.00 | true | true | PROMOTION CANDIDATE"));
    for arm in ["vllm", "sglang"] {
        assert!(markdown.contains(&format!(
            "| cuda | {} | synthetic | {arm} | 1 | 8 |",
            archive.model.key
        )));
        assert!(markdown.contains(&format!(
            "| cuda | {} | thoughtworks | {arm} | 1 |",
            archive.model.key
        )));
        let policy = report["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["cell"]["arm"] == arm)
            .unwrap();
        assert_eq!(policy["capacity_policy"]["comparison_kv_matched"], false);
    }
    let c1 = archive
        .cells
        .iter()
        .find(|cell| {
            cell["arm"] == "sglang" && cell["workload"] == "synthetic" && cell["concurrency"] == 1
        })
        .unwrap();
    archive.cell(c1, &"b".repeat(64));
    archive.report();
    let failed = outputs(&archive);
    assert_eq!(failed["promotion"][0]["winner"], "vllm");
    let candidate = failed["promotion"][0]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["arm"] == "sglang")
        .unwrap();
    assert_eq!(candidate["c1_parity"], false);
    assert_eq!(candidate["eligible"], false);
    let parity = read(&root.join("summary/parity.json")).unwrap();
    assert!(
        parity
            .as_array()
            .unwrap()
            .iter()
            .any(|gate| gate["cell"] == *c1 && gate["passed"] == false)
    );
    let trace = archive
        .cells
        .iter()
        .find(|cell| cell["arm"] == "vllm" && cell["workload"] == "thoughtworks")
        .unwrap();
    fs::remove_file(archive.directory(trace).join("complete.json")).unwrap();
    archive.report();
    let partial = outputs(&archive);
    assert_eq!(partial["report_complete"], false);
    assert_eq!(partial["rows"].as_array().unwrap().len(), 71);
    assert!(partial["promotion"][0]["winner"].is_null());
    assert!(
        fs::read_to_string(root.join("summary/REPORT.md"))
            .unwrap()
            .contains("hold")
    );
    temp.close().unwrap();
}

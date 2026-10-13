use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub struct Fixture {
    pub _directory: tempfile::TempDir,
    pub matrix: PathBuf,
    pub replay: PathBuf,
    pub hardware: PathBuf,
    pub runs: PathBuf,
    pub output: PathBuf,
    pub github: PathBuf,
    pub baseline: PathBuf,
}
pub fn write(path: &Path, value: &Value) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
pub fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
pub fn digest(value: &Value) -> String {
    hex::encode(Sha256::digest(serde_json::to_vec(value).unwrap()))
}
impl Fixture {
    pub fn new(recurrent: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let matrix = root.join("matrix.json");
        let replay = root.join("replay.json");
        let hardware = root.join("hardware.json");
        let runs = root.join("runs");
        let output = root.join("history.jsonl");
        let github = root.join("github-output");
        let baseline = root.join("baseline");
        let replay_value = json!({"mode":"all","passes":2,"warmup_turns":1,"max_output_tokens":128,"concurrency":[1],
            "sessions_per_concurrency":3,"minimum_worker_waves":1,"minimum_context_tokens":131072,
            "minimum_session_prompt_tokens":32768,"min_isl":32768,"max_isl":131072,"min_turns":2,
            "temperature":0,"seed":42,"selection_algorithm":"balanced-md5-v2","backend":"metal",
            "dataset":"fixture/dataset","dataset_revision":"d".repeat(40),"dataset_file":"sessions.parquet","dataset_sha256":"e".repeat(64)});
        let model = json!({"family":"fixture","class":if recurrent {"hybrid-recurrent"} else {"dense"},
            "repo":"fixture/model","revision":"b".repeat(40),"file":"model.gguf","quant":"Q4_K_M","sha256":"c".repeat(64),"native_context_tokens":131072});
        write(&matrix, &json!({"models":[model],"replay":replay_value}));
        write(&replay, &replay_value);
        write(
            &hardware,
            &json!({"machine_model":"fixture","chip":"fixture","gpu_cores":1,"unified_memory_bytes":274877906944_u64,"os_version":"fixture"}),
        );
        let fixture = Self {
            _directory: directory,
            matrix,
            replay,
            hardware,
            runs,
            output,
            github,
            baseline,
        };
        fixture.family("fixture", recurrent, "s");
        fixture
    }
    pub fn family(&self, name: &str, recurrent: bool, prefix: &str) {
        let root = self.runs.join(name);
        let trajectories: Vec<_> = ["swe-agent","mini-swe-agent","openhands"].iter().enumerate().map(|(index,framework)|json!({
            "session_id":format!("{prefix}{index}"),"source_dataset":"fixture","agent_framework":framework,"recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"},
                {"role":"user","content":"next"},{"role":"assistant","content":"final"}]
        })).collect();
        let mut warmup = trajectories[0].clone();
        warmup["session_id"] = "warmup".into();
        let manifest = json!({"cohorts":{"warmup":[warmup],"1":trajectories}});
        let manifest_path = root.join("inputs/manifest.json");
        write(&manifest_path, &manifest);
        let manifest_sha = hex::encode(Sha256::digest(std::fs::read(&manifest_path).unwrap()));
        let mut prompt_map = serde_json::Map::new();
        let mut records = Vec::new();
        let mut ids = Vec::new();
        let mut events = Vec::new();
        let mut recurrent_turns = Vec::new();
        for (index, trajectory) in trajectories.iter().enumerate() {
            let session = trajectory["session_id"].as_str().unwrap();
            for turn in 0..2 {
                let id = format!("{session}:{turn}");
                let prompt = if turn == 0 { 40000 } else { 41000 };
                prompt_map.insert(id.clone(), prompt.into());
                ids.push(id.clone());
                let started = (index * 2 + turn) as f64 * 2.0;
                records.push(json!({"session_id":session,"request_id":id,"assistant_turn":turn,"prompt_tokens":prompt,
                    "cached_tokens":if turn==0 {0} else {40000},"completion_tokens":2,"requested_output_tokens":8,
                    "generation_seconds":1.0,"elapsed_seconds":2.0,"ttft_seconds":1.0,"started":started,"completed":started+2.0,
                    "cache_pct":if turn==0 {0.0} else {100.0*40000.0/41000.0},"finish_reason":"stop","content_sha256":"f".repeat(64)}));
                let attributes = json!({"openai.prompt_cache_key":session,"skippy.kv.decision":if turn==0 {"miss"} else {"exact_hit"},
                    "skippy.exact_cache.payload_kind":if turn==0 {None} else {Some("kv-recurrent")},
                    "skippy.exact_cache.restored_tokens":if turn==0 {0} else {40000}});
                events.push(json!({"event":"stage.openai_kv_lookup_decision","start_time_unix_nanos":index*2+turn+1,"attributes":attributes}));
                recurrent_turns.push(json!({"request_id":id,"state_restored":turn>0,"restored_tokens":if turn==0 {0} else {40000},"lookup":attributes}));
            }
        }
        let cohort = digest(&Value::Array(trajectories));
        let cell = json!({"concurrency":1,"trajectories":3,"requests":6,"successful_requests":6,"failed_requests":0,"successful_request_ids":ids,
            "failed_request_ids":[],"completion_tokens":12.0,"prompt_tokens":243000.0,"cached_tokens":120000.0,
            "generation_seconds":6.0,"workload_window_seconds":12.0,"ttft_samples":[1.0,1.0,1.0,1.0,1.0,1.0],
            "cache_pct":100.0*120000.0/243000.0,"finish_reason_length_requests":0,"budget_exhausted_requests":0,
            "decode_tokens_per_second":2.0,"workload_output_tokens_per_second":1.0,
            "session_cohort_sha256":cohort,"completeness":{"passed":true,"problems":[],"expected_request_ids":ids,"expected_turns":6},
            "acceptance":{"passed":true,"problems":[]}});
        for pass in 1..=2 {
            let arm = root.join(format!("data/pass-{pass}/main"));
            let mut cell = cell.clone();
            if recurrent {
                cell["recurrent_state"] = json!({"passed":true,"problems":[],"restores":3,"minimum_restored_tokens":32768,"turns":recurrent_turns});
            }
            write(&arm.join("c-1.json"), &cell);
            let raw = records
                .iter()
                .map(|value| serde_json::to_string(value).unwrap())
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            std::fs::write(arm.join("c-1-requests.jsonl"), raw).unwrap();
            std::fs::write(
                arm.join("mesh.log"),
                events
                    .iter()
                    .map(|value| serde_json::to_string(value).unwrap())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
            std::fs::write(arm.join("mesh.stderr.log"), "").unwrap();
        }
        write(
            &root.join("context-preflight/main/runtime.json"),
            &json!({"models":[{"context_length":131072}]}),
        );
        write(
            &root.join("run.json"),
            &json!({"config":{"model":format!("fixture/model@{}/model.gguf","b".repeat(40))},
            "inputs":{"manifest":manifest_path,"manifest_sha256":manifest_sha},
            "builds":[{"label":"main","commit":"a".repeat(40),"binary_sha256":"9".repeat(64)}],
            "context_preflight":{"main":{"passed":true,"model":{"sha256":"c".repeat(64),"native_context_tokens":131072},
                "cohorts":{"1":{"passed":true,"context_tokens":131072}},"prompt_tokens_by_cohort":{"1":prompt_map}}},
            "gates":{"passed":true},"completed_at":"2026-10-02T00:00:00Z"}),
        );
    }
    pub fn run(&self, gate: bool, baseline: bool) -> Output {
        self.run_label("main", gate, baseline)
    }
    pub fn run_label(&self, label: &str, gate: bool, baseline: bool) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command
            .args(["automation", "replay-matrix", "history", "--matrix"])
            .arg(&self.matrix)
            .arg("--replay")
            .arg(&self.replay)
            .arg("--hardware")
            .arg(&self.hardware)
            .arg("--replay-dir")
            .arg(&self.runs)
            .args(["--label", label, "--source-sha"])
            .arg("a".repeat(40))
            .arg("--output")
            .arg(&self.output)
            .arg("--github-output")
            .arg(&self.github);
        if gate {
            command.arg("--gate");
        }
        if baseline {
            command.arg("--baseline").arg(&self.baseline);
        }
        command.output().unwrap()
    }
    pub fn rows(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    pub fn seed(&self, mut change: impl FnMut(&mut Value)) {
        assert!(self.run(false, false).status.success());
        let row = self.rows().remove(0);
        std::fs::create_dir_all(&self.baseline).unwrap();
        let mut prior = Vec::new();
        for index in 0..3 {
            let mut row = row.clone();
            row["source_sha"] = format!("{:040x}", index + 1).into();
            change(&mut row);
            prior.push(serde_json::to_string(&row).unwrap());
        }
        std::fs::write(self.baseline.join("prior.jsonl"), prior.join("\n") + "\n").unwrap();
    }
}

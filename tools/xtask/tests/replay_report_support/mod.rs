pub fn document(label: &str) -> serde_json::Value {
    let cohort = serde_json::json!({"trajectory_count":2,"assistant_turns":4,
        "framework_trajectories":{"harness":2},"framework_assistant_turns":{"harness":4}});
    serde_json::json!({"config":{"model":"model","concurrency":[1],"warmup_turns":1},
        "inputs":{"kind":"captured","manifest_sha256":"manifest","cohorts":{"warmup":cohort,"1":cohort}},
        "builds":[{"label":label,"version":"v`1`|snow 雪","commit":"abcdef0123456789"}],
        "results":[{"label":label,"ref":"refs/test","commit":"abcdef0123456789","cells":[{
            "concurrency":1,"trajectories":2,"requests":1,"successful_requests":1,
            "failed_request_ids":[],"successful_request_ids":["s:0"],"content_sha256_by_request":{"s:0":"same"},
            "prompt_tokens_min":40,"prompt_tokens_max":40,"completion_tokens":100,"prompt_tokens":40,"cached_tokens":30,
            "generation_seconds":10,"workload_window_seconds":20,"ttft_samples":[1,9],
            "decode_tokens_per_second":10,"agent_steps_per_second":0.05,"workload_output_tokens_per_second":5,
            "ttft_p50_seconds":1,"ttft_p95_seconds":9,"mean_in_flight":1,"budget_exhausted_requests":0}]}]})
}

pub fn run(artifact: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "report", "--artifact"])
        .arg(artifact)
        .output()
        .unwrap()
}

pub fn csv_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => record.push(std::mem::take(&mut field)),
            '\r' if !quoted => {}
            '\n' if !quoted => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            character => field.push(character),
        }
    }
    assert!(!quoted);
    records
}

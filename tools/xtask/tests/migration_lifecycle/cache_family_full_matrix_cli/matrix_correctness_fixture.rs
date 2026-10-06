//! Inert product report emitter invoked only in the isolated helper subprocess.
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};
pub(super) fn run() {
    let arguments: Vec<String> = std::env::args().collect();
    // The shell wrapper provides the actual product argv in an owned JSON file.
    let path = PathBuf::from(std::env::var_os("CACHE_MATRIX_ARGS").unwrap());
    let lines = std::fs::read_to_string(path).unwrap();
    let argv: Vec<_> = lines.lines().collect();
    assert_eq!(argv[0], "state-handoff");
    let mut flags = BTreeMap::new();
    let mut index = 1;
    while index < argv.len() {
        if argv[index] == "--n-gpu-layers=0" {
            index += 1;
            continue;
        }
        assert!(index + 1 < argv.len());
        assert!(flags.insert(argv[index], argv[index + 1]).is_none());
        index += 2;
    }
    let known = [
        "--model",
        "--model-id",
        "--stage-server-bin",
        "--layer-end",
        "--ctx-size",
        "--activation-width",
        "--stage-load-mode",
        "--state-layer-start",
        "--state-layer-end",
        "--state-stage-index",
        "--state-payload-kind",
        "--prefix-token-count",
        "--cache-hit-repeats",
        "--runtime-lane-count",
        "--source-bind-addr",
        "--restore-bind-addr",
        "--report-out",
        "--prompt",
    ];
    assert!(flags.keys().all(|k| known.contains(k)));
    assert!(arguments.iter().any(|a| a == "--ignored"));
    assert_eq!(flags["--model-id"], "Qwen/Qwen3-0.6B:Q8_0");
    assert_eq!(flags["--layer-end"], "28");
    assert_eq!(flags["--activation-width"], "1024");
    assert_eq!(flags["--ctx-size"], "512");
    assert_eq!(flags["--stage-load-mode"], "runtime-slice");
    assert_eq!(flags["--state-payload-kind"], "resident-kv");
    assert_eq!(flags["--prefix-token-count"], "64");
    assert_eq!(flags["--cache-hit-repeats"], "3");
    let start = flags["--state-layer-start"].parse::<u32>().unwrap();
    let end = flags["--state-layer-end"].parse::<u32>().unwrap();
    let stage = flags["--state-stage-index"].parse::<u32>().unwrap();
    assert!([(0, 28, 0), (0, 9, 0), (9, 18, 1), (18, 28, 2)].contains(&(start, end, stage)));
    let report = json!({"mode":"state-handoff","status":"pass","matches":true,"predicted_token_matches":true,"cache_hit_matches":true,"model_identity":{"model_id":"Qwen/Qwen3-0.6B:Q8_0"},"state_payload_kind":"resident-kv","stage_index":stage,"layer_start":start,"layer_end":end,"requested_prefix_token_count":64,"benchmark_prompt_token_count":65,"benchmark_prompt_text":"fixed prompt","activation_width":1024,"cache_hit_repeats":3,"cache_hit_import_ms":[1.0,2.0,3.0],"cache_hit_decode_ms":[3.0,4.0,5.0],"recompute_total_ms":8.0,"cache_hit_total_ms":6.0});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    std::fs::write(flags["--report-out"], &bytes).unwrap();
    println!("{}", String::from_utf8(bytes).unwrap());
}

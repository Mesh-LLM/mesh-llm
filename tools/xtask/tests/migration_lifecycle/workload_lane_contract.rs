//! Actual copied workload lane helpers, finite prerequisites, no model or server execution.
#[path = "workload_lane_contract/fixture.rs"]
mod fixture;
use fixture::{Fixture, function};
use serde_json::{Value, json};
#[test]
fn workload_lane_contract_summary_preserves_classes_and_never_promotes_preflight_evidence() {
    let fixture = Fixture::new();
    let classes = [
        "causal_generation",
        "embedding",
        "rerank",
        "encoder_decoder",
        "ocr",
        "speech_synthesis",
        "speech_recognition",
    ];
    let models: Vec<_> = classes
        .iter()
        .enumerate()
        .map(|(i, class)| json!({"family":format!("family-{i}"),"class":class}))
        .collect();
    let outcome = json!({"name":"model-preflight","status":"pass","outcome":"pass","exit_code":0});
    let mut rows: Vec<_> = models
        .iter()
        .map(|model| json!({"family":model["family"],"outcomes":[outcome.clone()]}))
        .collect();
    rows.push(json!({"family":"battery","outcomes":[{"name":"environment-preflight","status":"pass","outcome":"pass","exit_code":0}]}));
    rows.push(json!({"family":"family-0","split_layer":2,"outcomes":[{"name":"chain","status":"pass","outcome":"pass","exit_code":0}]}));
    rows.push(json!({"family":"explicit","workload_class":"embedding","outcomes":[{"name":"embedding-oracle","status":"pass","outcome":"pass","exit_code":0}]}));
    let input = rows
        .iter()
        .map(|row| serde_json::to_string(row).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fixture.write("results.jsonl", &input);
    fixture.write("plan.json", &json!({"selected_models":models}).to_string());
    let result = fixture.run(
        &(function("skippy-family-battery.sh", "write_lane_summary") + "\nwrite_lane_summary\n"),
        &[],
    );
    assert_eq!(result.code, 0, "{}", result.stderr);
    let summary = fixture.text("summary.tsv");
    let actual: Vec<_> = summary
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect();
    assert_eq!(
        actual[0],
        [
            "family",
            "class",
            "split_layer",
            "lane",
            "status",
            "outcome",
            "exit_code"
        ]
    );
    let expected: Vec<_> = classes
        .into_iter()
        .chain(["", "causal_generation", "embedding"])
        .collect();
    assert_eq!(
        actual[1..].iter().map(|row| row[1]).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(fixture.text("results.jsonl"), input);
}
#[test]
fn workload_lane_contract_dry_run_plans_each_oracle_lane_without_executing_or_provisioning() {
    let fixture = Fixture::new();
    let result = fixture.battery(true);
    assert_eq!(result.code, 0, "{}", result.stderr);
    for seconds in [600, 900] {
        assert!(
            result
                .stdout
                .contains(&format!("--startup-timeout-secs {seconds}"))
        );
    }
    assert_eq!(result.stdout.matches("--require-oracle").count(), 2);
    assert!(result.stdout.contains("counts=2,0"));
    assert!(!fixture.root.join("results.jsonl").exists());
}
#[test]
fn workload_lane_contract_missing_oracle_retains_both_family_failures_and_continues() {
    let fixture = Fixture::new();
    let result = fixture.battery(false);
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert!(result.stdout.contains("counts=2,2"));
    let results = fixture.text("results.jsonl");
    let rows: Vec<Value> = serde_json::Deserializer::from_str(&results)
        .into_iter::<Value>()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row["family"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    for row in rows {
        assert_eq!(row["exit_code"], 1);
        let outcomes = row["outcomes"].as_array().unwrap();
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|lane| lane["status"] == "fail"));
    }
}
#[test]
fn workload_lane_contract_each_readiness_budget_rejects_dead_children_and_admits_matching_model() {
    for label in ["OpenAI server", "monolithic oracle server"] {
        for mode in ["timeout", "dead", "ready", "wrong-model"] {
            let fixture = Fixture::new();
            fixture.write("server.log", "finite startup log\n");
            let script = match mode {
                "timeout" => {
                    "curl() { return 1; }; wait_for_workload_server $$ 1 \"$SERVER_LOG\" \"$LABEL\""
                }
                "dead" => {
                    "(exit 0) & dead=$!; wait \"$dead\"; wait_for_workload_server \"$dead\" 1 \"$SERVER_LOG\" \"$LABEL\""
                }
                "ready" => {
                    "curl() { printf '{\"data\":[{\"id\":\"fixture\"}]}'; }; wait_for_workload_server $$ 1 \"$SERVER_LOG\" \"$LABEL\""
                }
                "wrong-model" => {
                    "curl() { printf '{\"data\":[{\"id\":\"different-model\"}]}'; }; wait_for_workload_server $$ 1 \"$SERVER_LOG\" \"$LABEL\""
                }
                _ => unreachable!(),
            };
            let result = fixture.run(
                &(function("skippy-workload-certify.sh", "wait_for_workload_server")
                    + "\n"
                    + script),
                &[
                    ("STARTUP_TIMEOUT_SECS", "1".into()),
                    ("MODEL_CLASS", "embedding".into()),
                    ("MODEL_ID", "fixture".into()),
                    (
                        "SERVER_LOG",
                        fixture.root.join("server.log").display().to_string(),
                    ),
                    ("LABEL", label.into()),
                ],
            );
            assert_eq!(
                result.code,
                if mode == "ready" { 0 } else { 1 },
                "{mode}: {}",
                result.stderr
            );
            if mode != "ready" {
                assert!(result.stderr.contains("finite startup log"));
                assert!(result.stderr.contains(label));
                assert!(result.stderr.contains(if mode == "dead" {
                    "exited early"
                } else {
                    "was not ready within 1 seconds"
                }));
            }
        }
    }
}
#[test]
fn workload_lane_contract_address_in_use_detection_rejects_other_startup_failures() {
    for (message, code) in [
        ("Address already in use (os error 48)", 0),
        ("AddrInUse", 0),
        ("EADDRINUSE", 0),
        ("model initialization failed", 1),
    ] {
        let fixture = Fixture::new();
        fixture.write("server.log", message);
        let result = fixture.run(
            &(function("skippy-workload-certify.sh", "address_in_use_log")
                + "\naddress_in_use_log \"$LOG\"\n"),
            &[("LOG", fixture.root.join("server.log").display().to_string())],
        );
        assert_eq!(result.code, code, "{message}: {}", result.stderr);
    }
}

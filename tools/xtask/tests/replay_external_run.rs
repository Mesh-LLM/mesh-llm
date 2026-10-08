#![cfg(unix)]
#[path = "replay_external_run/support.rs"]
mod support;
use serde_json::json;
use sha2::{Digest, Sha256};
use support::*;

#[test]
fn all_external_engine_types_execute_captured_runs_through_the_cli() {
    for engine in ["llama.cpp", "vllm", "sglang"] {
        let fixture = Fixture::new(false);
        fixture.modify_config(|config| config["arms"][0]["engine"] = engine.into());
        assert_success(&fixture.run());
        assert_eq!(fixture.document()["builds"][0]["engine"], engine);
        assert_eq!(fixture.document()["results"][0]["engine"], engine);
        assert_eq!(fixture.launches(), 2);
    }
}

#[test]
fn server_failure_preserves_partial_evidence_without_claiming_a_complete_run() {
    let fixture = Fixture::new(false);
    fixture.fail_server();
    assert!(!fixture.run().status.success());
    let document = fixture.document();
    assert!(document["completed_at"].is_null());
    assert!(document["gates"].is_null());
    assert_eq!(document["results"].as_array().unwrap().len(), 1);
    assert_eq!(document["results"][0]["passed"], false);
    assert_eq!(document["results"][0]["acceptance_failed"], false);
    assert!(document["results"][0]["error"].is_string());
    assert!(document["results"][0]["command"].is_array());
    assert_eq!(fixture.launches(), 1);
    assert!(!fixture.output.join("summary/REPORT.md").exists());
}

#[test]
fn external_only_captured_run_retains_versions_commands_and_reports() {
    let fixture = Fixture::new(false);
    assert_success(&fixture.run());
    let document = fixture.document();
    assert_eq!(document["gates"]["passed"], true);
    assert_eq!(document["results"].as_array().unwrap().len(), 2);
    assert_eq!(
        document["builds"][0]["version_sha256"],
        hex::encode(Sha256::digest(b"fixture engine 1.2"))
    );
    assert_eq!(
        document["builds"][0]["provenance"]["resolved_executable"],
        serde_json::to_value(&fixture.executable).unwrap()
    );
    assert_eq!(
        document["context_preflight"]["external.fixture"]["status"],
        "not_requested"
    );
    assert!(document["context_preflight"]["external.fixture"]["passed"].is_null());
    for result in document["results"].as_array().unwrap() {
        assert_eq!(result["engine"], "llama.cpp");
        assert_eq!(
            result["version_sha256"],
            document["builds"][0]["version_sha256"]
        );
        assert_eq!(
            result["command"][0],
            document["builds"][0]["provenance"]["resolved_executable"]
        );
        assert!(result["artifact_sha256"]["server.command.json"].is_string());
        assert!(result["artifact_sha256"]["workload.json"].is_string());
    }
    let report = std::fs::read_to_string(fixture.output.join("summary/REPORT.md")).unwrap();
    assert!(report.contains("Context qualification: not requested"));
    assert!(report.contains("fixture engine 1.2"));
    assert!(fixture.output.join("summary/comparison.csv").exists());
    assert!(!fixture.output.join("context-preflight").exists());
    assert!(
        !fixture
            .output
            .join("data/pass-1/external.fixture/runtime.json")
            .exists()
    );
    assert_eq!(fixture.launches(), 2);
}

#[test]
fn mixed_mesh_external_runs_alternate_abba_and_reverse_even_pass_cells() {
    let fixture = Fixture::new(true);
    fixture.modify_input(|input| input["passes"] = 4.into());
    assert_success(&fixture.run());
    let document = fixture.document();
    let labels = document["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["label"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            "mesh",
            "external.fixture",
            "external.fixture",
            "mesh",
            "mesh",
            "external.fixture",
            "external.fixture",
            "mesh"
        ]
    );
    for result in document["results"].as_array().unwrap() {
        let levels = result["cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(|cell| cell["concurrency"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            levels,
            if result["pass"].as_u64().unwrap().is_multiple_of(2) {
                vec![2, 1]
            } else {
                vec![1, 2]
            }
        );
    }
    assert_eq!(document["builds"][0]["engine"], "mesh");
    assert_eq!(document["builds"][1]["engine"], "llama.cpp");
    assert!(document["builds"][1]["binary_sha256"].is_null());
    assert_eq!(document["gates"]["passed"], true);
}

#[test]
fn completed_mixed_resume_verifies_identity_without_relaunching_servers() {
    let fixture = Fixture::new(true);
    assert_success(&fixture.run());
    let prior = fixture.document();
    fixture.modify_input(|input| input["resume"] = true.into());
    assert_success(&fixture.run());
    assert_eq!(fixture.launches(), 2);
    assert_eq!(fixture.document()["results"], prior["results"]);
    assert_eq!(fixture.document()["order"], prior["order"]);
}

#[test]
fn resume_rejects_version_source_options_and_executable_drift_before_server_launch() {
    for change in ["version", "source", "options", "executable"] {
        let fixture = Fixture::new(false);
        assert_success(&fixture.run());
        let prior = std::fs::read(fixture.output.join("run.json")).unwrap();
        fixture.modify_input(|input| input["resume"] = true.into());
        match change {
            "version" => std::fs::write(&fixture.version, "fixture engine 2.0\n").unwrap(),
            "source" => {
                let mut bytes = std::fs::read(&fixture.config).unwrap();
                bytes.push(b'\n');
                std::fs::write(&fixture.config, bytes).unwrap();
            }
            "options" => {
                fixture.modify_config(|config| config["arms"][0]["prefix_cache"] = false.into())
            }
            "executable" => fixture
                .modify_config(|config| config["arms"][0]["executable"] = "./engine-server".into()),
            _ => unreachable!(),
        }
        assert!(!fixture.run().status.success(), "{change}");
        assert_eq!(fixture.launches(), 2, "{change}");
        assert_eq!(
            std::fs::read(fixture.output.join("run.json")).unwrap(),
            prior,
            "{change}"
        );
    }
}

#[test]
fn resume_rejects_retained_command_and_context_evidence_tampering() {
    for change in ["command", "context"] {
        let fixture = Fixture::new(false);
        assert_success(&fixture.run());
        fixture.modify_input(|input| input["resume"] = true.into());
        if change == "command" {
            write_json(
                &fixture
                    .output
                    .join("data/pass-1/external.fixture/server.command.json"),
                &json!(["foreign"]),
            );
        } else {
            modify(&fixture.output.join("run.json"), |run| {
                run["context_preflight"]["external.fixture"] = json!({"passed":true})
            });
        }
        assert!(!fixture.run().status.success());
        assert_eq!(fixture.launches(), 2);
    }
}

#[test]
fn configured_gate_failure_retains_complete_external_reports() {
    let fixture = Fixture::new(false);
    fixture.modify_input(|input| input["min_cache_pct"] = 80.into());
    assert!(!fixture.run().status.success());
    assert_eq!(fixture.document()["gates"]["passed"], false);
    assert_eq!(fixture.document()["results"].as_array().unwrap().len(), 2);
    assert!(fixture.output.join("summary/REPORT.md").exists());
    assert!(fixture.output.join("summary/charts/ttft-p50.svg").exists());
}

#[test]
fn external_long_context_or_label_overlap_rejects_before_launch() {
    for change in ["context", "overlap"] {
        let fixture = Fixture::new(true);
        if change == "context" {
            fixture.modify_input(|input| {
                input["model"] = "/nonexistent/provenance".into();
                input["context_qualification"] = "mesh".into();
            });
        } else {
            fixture.modify_config(|config| config["arms"][0]["label"] = "mesh".into());
        }
        assert!(!fixture.run().status.success());
        assert!(!fixture.state.path().join("launches.txt").exists());
        assert!(!fixture.output.exists());
    }
}

//! CLI documentation ownership, generation and normal CI contracts.
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn source(path: &str) -> String {
    fs::read_to_string(root().join(path)).expect("authored CLI website contract source")
}
fn json(path: &str) -> Value {
    serde_json::from_str(&source(path)).expect("valid authored JSON")
}
fn terms(text: &str, expected: &[&str]) {
    for term in expected {
        assert!(text.contains(term), "missing CLI website contract {term}");
    }
}

#[test]
fn cli_website_build_and_dev_generate_inventory_and_keep_browser_tests() {
    let package = json("mesh/website/package.json");
    let scripts = &package["scripts"];
    assert_eq!(
        scripts["generate:cli"],
        "node ./scripts/generate-cli-inventory.mjs"
    );
    assert_eq!(
        scripts["check:cli"],
        "node ./scripts/generate-cli-inventory.mjs --check"
    );
    for mode in ["build", "dev"] {
        terms(
            scripts[mode].as_str().expect("npm script"),
            &["npm run generate:cli"],
        );
    }
    assert_eq!(
        scripts["test:cli-explorer"],
        "node ./scripts/test-cli-explorer.mjs"
    );
}

#[test]
fn cli_website_d3_is_locked_local_and_script_json_is_escaped() {
    let package = json("mesh/website/package.json");
    let lock = json("mesh/website/package-lock.json");
    assert_eq!(package["dependencies"]["d3"], "^7.9.0");
    assert_eq!(lock["packages"][""]["dependencies"]["d3"], "^7.9.0");
    assert_eq!(lock["packages"]["node_modules/d3"]["version"], "7.9.0");
    let config = source("mesh/website/.eleventy.js");
    terms(
        &config,
        &[
            r#""node_modules/d3/dist/d3.min.js""#,
            r#""assets/d3.min.js""#,
            r#"addFilter("jsonScript""#,
            r#""<": "\\u003C""#,
            r#""\u2028": "\\u2028""#,
        ],
    );
    assert!(!config.to_lowercase().contains("jsdelivr"));
}

#[test]
fn cli_website_generated_inventory_is_ignored_and_owned_by_cleaner() {
    assert!(
        source("mesh/website/.gitignore")
            .lines()
            .any(|line| line == "src/_data/cliInventory.json")
    );
    terms(
        &source("mesh/website/scripts/clean-generated-site.mjs"),
        &["\"mesh/website/src/_data/cliInventory.json\""],
    );
}

#[test]
fn cli_website_generator_uses_locked_exporter_and_validates_schema_and_paths() {
    terms(
        &source("mesh/website/scripts/generate-cli-inventory.mjs"),
        &[
            "\"run\"",
            "\"--locked\"",
            "\"--quiet\"",
            "\"-p\"",
            "\"mesh-llm-cli\"",
            "\"--bin\"",
            "\"mesh-llm-cli-inventory\"",
            "\"--\"",
            "\"--check\"",
            "schemaVersion",
            "document.root",
            "duplicate node path",
        ],
    );
}

#[test]
fn cli_website_cli_crate_owns_cli_domain() {
    let catalog = json("ci/ownership.yml");
    assert!(
        catalog["crate_rules"]
            .as_array()
            .expect("crate rules")
            .iter()
            .any(|rule| {
                rule["domain"] == "cli"
                    && rule["crates"]
                        .as_array()
                        .expect("owned crates")
                        .iter()
                        .any(|name| name == "mesh-llm-cli")
            })
    );
}

fn affected_cli_documentation() -> Value {
    let scratch = tempfile::tempdir().expect("CLI selection output directory");
    let stdout_path = scratch.path().join("selection.json");
    let stderr_path = scratch.path().join("diagnostic.txt");
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args([
            "repository",
            "affected-crates",
            "mesh/crates/mesh-llm-cli/src/parser/commands.rs",
        ])
        .env("CARGO_NET_OFFLINE", "true")
        .stdin(Stdio::null())
        .stdout(fs::File::create(&stdout_path).expect("selection output"))
        .stderr(fs::File::create(&stderr_path).expect("diagnostic output"))
        .spawn()
        .expect("actual Cargo-built xtask executable");
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().expect("selection child status") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CLI documentation selection exceeded 30 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(
        status.success(),
        "CLI selection failed: {}",
        fs::read_to_string(stderr_path).expect("selection diagnostic")
    );
    serde_json::from_slice(&fs::read(stdout_path).expect("selection output"))
        .expect("typed affected-crates output")
}

#[test]
fn cli_website_nested_cli_inputs_select_cli_validation_without_website_build() {
    terms(
        &source(".github/actions/compute-changes/derive-outputs.sh"),
        &["^(mesh/|skippy/)?crates/mesh-llm-cli/"],
    );
    let payload = affected_cli_documentation();
    assert_eq!(payload["website_changed"], false);
    assert_eq!(
        payload["all_rust"], false,
        "metadata failure must not satisfy direct CLI ownership"
    );
    assert!(
        payload["affected"]
            .as_array()
            .expect("affected crates")
            .iter()
            .any(|name| name == "mesh-llm-cli")
    );
    assert!(
        payload["test_crates"]
            .as_array()
            .expect("direct crates")
            .iter()
            .any(|name| name == "mesh-llm-cli")
    );
}

#[test]
fn cli_website_ci_keeps_inventory_and_browser_contracts_in_normal_workflows() {
    let targets = json("ci/quality-rust-contract-targets.json");
    assert_eq!(
        targets
            .as_array()
            .expect("required Rust owners")
            .iter()
            .filter(|target| target.as_str() == Some("lane_l2_quality_contracts"))
            .count(),
        1,
        "the retired CLI module replacement must be a required normal CI owner"
    );
    terms(
        &source("just/ci.just"),
        &["--test lane_l2_quality_contracts"],
    );
    let quality = source(".github/workflows/ci-quality-slice.yml");
    assert_eq!(cli_inventory_timeout(&quality), 15);
    terms(
        &quality,
        &[
            "Verify generated CLI inventory is deterministic and current",
            "just cli-inventory-check",
        ],
    );
    assert!(!quality.contains("Require public website docs update"));
    for path in [
        ".github/workflows/ci-web-slice.yml",
        ".github/workflows/website-pages.yml",
    ] {
        terms(&source(path), &["npm run test:cli-explorer"]);
    }
}

fn cli_inventory_timeout(quality: &str) -> u64 {
    let job = quality
        .split_once("  cli_docs_sync:\n")
        .expect("CLI documentation job")
        .1
        .split("\n  #")
        .next()
        .expect("CLI documentation body");
    job.lines()
        .find_map(|line| line.trim().strip_prefix("timeout-minutes: "))
        .expect("bounded CLI inventory job")
        .parse::<u64>()
        .expect("numeric timeout")
}

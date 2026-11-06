use super::{check, invocations, policy::POLICY_PATH};
use crate::command::DynResult;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

const POLICY: &str = r#"{
  "schema_version": 1,
  "dependency_inputs": [{
    "manifest_path": "ci/llama-canary/jinja/pyproject.toml",
    "lock_path": "ci/llama-canary/jinja/uv.lock",
    "consumer_path": "scripts/llama-canary-agent-repair.sh",
    "consumer_assignment": "JINJA_PROJECT",
    "purpose": "locked upstream Jinja CTest dependencies",
    "project_name": "mesh-llama-canary-jinja-tests",
    "project_version": "0.1.0",
    "direct_dependencies": [{"name": "jinja2", "version": "3.1.6"}],
    "locked_packages": [
      {"name": "jinja2", "version": "3.1.6"},
      {"name": "markupsafe", "version": "3.0.4"}
    ]
  }],
  "external_source_descriptors": [{
    "descriptor_path": "ci/required-sdk-python/sdk-source.json",
    "repository": "Mesh-LLM/mesh-llm-python-sdk",
    "revision": "d2a881612186b0626a153ee237dfa3df00ab6087",
    "manifest_sha256": "2844313b33527d1745d7208b568bdbab7984a0459172f5cfa305eacb35e2b9c7",
    "purpose": "required external SDK compatibility and embedding clients"
  }],
  "external_script_pins": [{
    "path": "scripts/run-llama-benchy-openai.sh",
    "repository": "https://github.com/eugr/llama-benchy",
    "revision": "e9be344578cec17745066b220798b80a0d2686d3",
    "assignment": "LLAMA_BENCHY_FROM",
    "purpose": "default benchmark CLI source; caller overrides remain caller-owned"
  }],
  "allowed_invocations": []
}"#;

fn fixture() -> (TempDir, PathBuf) {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().to_owned();
    Command::new("git")
        .args(["init", "-q"])
        .current_dir(&root)
        .status()
        .unwrap();
    write(&root, POLICY_PATH, POLICY);
    write(
        &root,
        "ci/required-sdk-python/sdk-source.json",
        r#"{"repository":"Mesh-LLM/mesh-llm-python-sdk","revision":"d2a881612186b0626a153ee237dfa3df00ab6087","manifest_sha256":"2844313b33527d1745d7208b568bdbab7984a0459172f5cfa305eacb35e2b9c7"}"#,
    );
    write(
        &root,
        "scripts/run-llama-benchy-openai.sh",
        "LLAMA_BENCHY_FROM=\"${LLAMA_BENCHY_FROM:-git+https://github.com/eugr/llama-benchy@e9be344578cec17745066b220798b80a0d2686d3}\"\n",
    );
    write(
        &root,
        "scripts/llama-canary-agent-repair.sh",
        "JINJA_PROJECT=\"$TRUSTED_ROOT/ci/llama-canary/jinja\"\n",
    );
    write(
        &root,
        "ci/llama-canary/jinja/pyproject.toml",
        "[project]\nname = \"mesh-llama-canary-jinja-tests\"\nversion = \"0.1.0\"\ndependencies = [\"jinja2==3.1.6\"]\n",
    );
    write(
        &root,
        "ci/llama-canary/jinja/uv.lock",
        "version = 1\nrequires-python = \">=3.9\"\n\n[[package]]\nname = \"mesh-llama-canary-jinja-tests\"\nversion = \"0.1.0\"\nsource = { virtual = \".\" }\n\n[[package]]\nname = \"jinja2\"\nversion = \"3.1.6\"\nsource = { registry = \"https://pypi.org/simple\" }\nsdist = { hash = \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" }\nwheels = [{ hash = \"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" }]\n\n[[package]]\nname = \"markupsafe\"\nversion = \"3.0.4\"\nsource = { registry = \"https://pypi.org/simple\" }\nsdist = { hash = \"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\" }\nwheels = [{ hash = \"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\" }]\n",
    );
    (temporary, root)
}

fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = fs::File::create(path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
}

fn check_fixture(root: &Path) -> DynResult<()> {
    check(root)
}

#[test]
fn exact_external_project_and_pins_are_admitted() {
    let (_temporary, root) = fixture();
    check_fixture(&root).unwrap();
}

fn agentx_fixture() -> (TempDir, PathBuf) {
    let (temporary, root) = fixture();
    let mut policy: serde_json::Value = serde_json::from_str(POLICY).unwrap();
    let reviewed: serde_json::Value =
        serde_json::from_str(include_str!("../../../../../ci/python-tooling-policy.json")).unwrap();
    let agentx = reviewed["dependency_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|input| input["manifest_path"] == "skippy/evals/agentx/pyproject.toml")
        .unwrap();
    policy["dependency_inputs"]
        .as_array_mut()
        .unwrap()
        .push(agentx.clone());
    policy["allowed_invocations"] = serde_json::Value::Array(
        reviewed["allowed_invocations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|invocation| invocation["path"] == "just/skippy.just")
            .cloned()
            .collect(),
    );
    write(&root, POLICY_PATH, &serde_json::to_string(&policy).unwrap());
    write(
        &root,
        "skippy/evals/agentx/pyproject.toml",
        include_str!("../../../../../skippy/evals/agentx/pyproject.toml"),
    );
    write(
        &root,
        "skippy/evals/agentx/uv.lock",
        include_str!("../../../../../skippy/evals/agentx/uv.lock"),
    );
    write(
        &root,
        "just/skippy.just",
        include_str!("../../../../../just/skippy.just"),
    );
    (temporary, root)
}

#[test]
fn reviewed_agentx_wheel_lock_and_preinstalled_interpreter_setup_are_admitted() {
    let (_temporary, root) = agentx_fixture();
    check_fixture(&root).unwrap();
}

#[test]
fn agentx_manifest_source_override_must_match_its_reviewed_wheel() {
    for missing in [false, true] {
        let (_temporary, root) = agentx_fixture();
        let path = "skippy/evals/agentx/pyproject.toml";
        let mut manifest: toml::Value =
            toml::from_str(&fs::read_to_string(root.join(path)).unwrap()).unwrap();
        if missing {
            manifest["tool"]["uv"]
                .as_table_mut()
                .unwrap()
                .remove("sources");
        } else {
            manifest["tool"]["uv"]["sources"]["agentx-harness"]["url"] =
                toml::Value::String("https://example.org/other.whl".into());
        }
        write(&root, path, &toml::to_string(&manifest).unwrap());
        assert!(
            check_fixture(&root)
                .unwrap_err()
                .to_string()
                .contains("reviewed")
        );
    }
}

#[test]
fn agentx_source_url_wheel_url_and_valid_but_different_hash_are_rejected() {
    for field in [
        "source-url",
        "wheel-url",
        "wheel-hash",
        "extra-wheel",
        "sdist",
    ] {
        let (_temporary, root) = agentx_fixture();
        let path = "skippy/evals/agentx/uv.lock";
        let mut lock: toml::Value =
            toml::from_str(&fs::read_to_string(root.join(path)).unwrap()).unwrap();
        let agentx = lock["package"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|package| package["name"].as_str() == Some("agentx-harness"))
            .unwrap();
        match field {
            "source-url" => {
                agentx["source"]["url"] =
                    toml::Value::String("https://example.org/other.whl".into())
            }
            "wheel-url" => {
                agentx["wheels"][0]["url"] =
                    toml::Value::String("https://example.org/other.whl".into())
            }
            "wheel-hash" => {
                agentx["wheels"][0]["hash"] =
                    toml::Value::String(format!("sha256:{}", "e".repeat(64)))
            }
            "extra-wheel" => {
                let extra = agentx["wheels"][0].clone();
                agentx["wheels"].as_array_mut().unwrap().push(extra);
            }
            "sdist" => {
                let sdist = agentx["wheels"][0].clone();
                agentx.as_table_mut().unwrap().insert("sdist".into(), sdist);
            }
            _ => unreachable!(),
        }
        write(&root, path, &toml::to_string(&lock).unwrap());
        assert!(
            check_fixture(&root)
                .unwrap_err()
                .to_string()
                .contains("reviewed exact wheel URL/SHA-256"),
            "{field}"
        );
    }
}

#[test]
fn registry_admission_cannot_hide_a_second_package_source() {
    let (_temporary, root) = fixture();
    let path = "ci/llama-canary/jinja/uv.lock";
    let source = fs::read_to_string(root.join(path)).unwrap().replacen(
        "source = { registry = \"https://pypi.org/simple\" }",
        "source = { registry = \"https://pypi.org/simple\", url = \"https://example.org/other.whl\" }", 1);
    write(&root, path, &source);
    assert!(
        check_fixture(&root)
            .unwrap_err()
            .to_string()
            .contains("pinned PyPI registry")
    );
}

#[test]
fn agentx_lock_rejects_missing_extra_and_duplicate_transitive_packages() {
    for change in ["missing", "extra", "duplicate"] {
        let (_temporary, root) = agentx_fixture();
        let path = "skippy/evals/agentx/uv.lock";
        let mut lock: toml::Value =
            toml::from_str(&fs::read_to_string(root.join(path)).unwrap()).unwrap();
        let packages = lock["package"].as_array_mut().unwrap();
        let index = packages
            .iter()
            .position(|package| package["name"].as_str() == Some("aiofiles"))
            .unwrap();
        match change {
            "missing" => {
                packages.remove(index);
            }
            "extra" => {
                let mut extra = packages[index].clone();
                extra["name"] = toml::Value::String("unreviewed-package".into());
                packages.push(extra);
            }
            "duplicate" => {
                packages.push(packages[index].clone());
            }
            _ => unreachable!(),
        }
        write(&root, path, &toml::to_string(&lock).unwrap());
        let error = check_fixture(&root).unwrap_err().to_string();
        assert!(
            error.contains("locked package set differs")
                || error.contains("repeats locked package"),
            "{change}: {error}"
        );
    }
}

#[test]
fn agentx_url_admission_does_not_allow_other_transitive_sources_or_versions() {
    for source_override in [false, true] {
        let (_temporary, root) = agentx_fixture();
        let path = "skippy/evals/agentx/uv.lock";
        let mut lock: toml::Value =
            toml::from_str(&fs::read_to_string(root.join(path)).unwrap()).unwrap();
        let package = lock["package"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|package| package["name"].as_str() == Some("aiofiles"))
            .unwrap();
        if source_override {
            package["source"] = toml::Value::Table(toml::map::Map::from_iter([(
                "url".into(),
                toml::Value::String("https://example.org/aiofiles.whl".into()),
            )]));
        } else {
            package["version"] = toml::Value::String("99.0.0".into());
        }
        write(&root, path, &toml::to_string(&lock).unwrap());
        let error = check_fixture(&root).unwrap_err().to_string();
        assert!(
            if source_override {
                error.contains("pinned PyPI registry")
            } else {
                error.contains("locked package set differs")
            },
            "{error}"
        );
    }
}

#[test]
fn agentx_admission_keeps_python_source_and_command_occurrences_closed() {
    let (_temporary, root) = agentx_fixture();
    write(&root, "skippy/evals/agentx/local-harness.py", "pass\n");
    assert!(
        check_fixture(&root)
            .unwrap_err()
            .to_string()
            .contains("repository Python source is forbidden")
    );
    fs::remove_file(root.join("skippy/evals/agentx/local-harness.py")).unwrap();
    let setup = fs::read_to_string(root.join("just/skippy.just")).unwrap();
    write(
        &root,
        "just/skippy.just",
        &setup.replace("--no-python-downloads ", ""),
    );
    assert!(
        check_fixture(&root)
            .unwrap_err()
            .to_string()
            .contains("unreviewed Python tooling invocation")
    );
    let (_temporary, root) = agentx_fixture();
    let setup = fs::read_to_string(root.join("just/skippy.just")).unwrap();
    write(
        &root,
        "just/skippy.just",
        &format!(
            "{setup}\nextra-client-install:\n    uv sync --locked --no-managed-python --no-python-downloads --python 3.12 --project \"$AGENTX_PROJECT\"\n"
        ),
    );
    assert!(
        check_fixture(&root)
            .unwrap_err()
            .to_string()
            .contains("exceeds its reviewed occurrence count")
    );
}

#[test]
fn url_wheel_admission_must_pin_a_direct_dependency_and_a_valid_digest() {
    for field in ["name", "sha256", "url"] {
        let (_temporary, root) = agentx_fixture();
        let mut policy: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join(POLICY_PATH)).unwrap()).unwrap();
        policy["dependency_inputs"][1]["url_packages"][0][field] = serde_json::json!(match field {
            "name" => "aiofiles",
            "sha256" => "not-a-sha256",
            "url" => "http://example.org/unpinned.whl",
            _ => unreachable!(),
        });
        write(&root, POLICY_PATH, &serde_json::to_string(&policy).unwrap());
        assert!(
            check_fixture(&root)
                .unwrap_err()
                .to_string()
                .contains("exact direct wheel"),
            "{field}"
        );
    }
}

#[test]
fn locked_project_must_match_its_consumed_assignment() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "scripts/llama-canary-agent-repair.sh",
        "JINJA_PROJECT=\"$TRUSTED_ROOT/ci/llama-canary/other\"\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("must bind JINJA_PROJECT to the reviewed project directory"));
}

#[test]
fn hidden_python_source_is_rejected_outside_the_old_eval_directory() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".evals/agents/new_probe.py",
        "print('not admitted')\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains(".evals/agents/new_probe.py"));
}

#[test]
fn hidden_sensitive_name_python_source_is_captured_from_raw_git_roster() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".agents/private/access-token.py",
        "print('not admitted')\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains(".agents/private/access-token.py"));
}

#[test]
fn new_unreviewed_python_package_input_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".agents/new-tool/requirements-dev.txt",
        "example==1.2.3\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("unreviewed Python package input"));
}

#[test]
fn suffixless_python_shebang_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "skills/new-tool/run",
        "#!/usr/bin/env -S python3 -I\nprint('x')\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("Python shebang"));
}

#[test]
fn extensionless_maintained_script_invocation_is_rejected() {
    let (_temporary, root) = fixture();
    write(&root, "scripts/new-check", "#!/bin/sh\npython3 -m probe\n");
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("unreviewed Python tooling invocation"));
}

#[test]
fn python_inside_executable_shell_heredoc_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "scripts/new-check.sh",
        "bash <<'SCRIPT'\npython3 -m probe\nSCRIPT\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("scripts/new-check.sh"));
    assert!(error.contains("unreviewed Python tooling invocation"));
}

#[test]
fn quoted_or_arithmetic_shift_does_not_open_a_heredoc() {
    for opener in [
        "echo \"usage: tool <<EOF\"",
        "shift=$((1 << 2))",
        "(( mask = 1 << 3 ))",
    ] {
        let (_temporary, root) = fixture();
        write(
            &root,
            "scripts/new-check.sh",
            &format!("{opener}\npython3 -m probe\nEOF\n2\n3\n"),
        );
        let error = check_fixture(&root).unwrap_err().to_string();
        assert!(
            error.contains("unreviewed Python tooling invocation"),
            "{opener}: {error}"
        );
    }
}

#[test]
fn python_text_in_cat_heredoc_remains_data() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "scripts/show-sample.sh",
        "cat <<'DATA'\npython3 -m probe\nDATA\n",
    );
    check_fixture(&root).unwrap();
}

#[test]
fn inline_workflow_interpreter_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".github/workflows/new.yml",
        "jobs:\n  check:\n    steps:\n      - run: \"python3 -c 'print(1)'\"\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains(".github/workflows/new.yml"));
}

#[test]
fn flow_map_workflow_interpreter_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".github/workflows/new.yml",
        "jobs:\n  check:\n    steps: [{name: probe, run: \"python3 -c 'print(1)'\"}]\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains(".github/workflows/new.yml"));
    assert!(error.contains("flow-map workflow run fields are unsupported"));
}

#[test]
fn block_flow_map_workflow_steps_fail_closed() {
    for step in [
        "      - {run: \"python3 -c 'print(1)'\"}\n",
        "      - {name: probe, run: \"python3 -c 'print(1)'\"}\n",
    ] {
        let (_temporary, root) = fixture();
        write(
            &root,
            ".github/workflows/new.yml",
            &format!("jobs:\n  check:\n    steps:\n{step}"),
        );
        let error = check_fixture(&root).unwrap_err().to_string();
        assert!(error.contains("flow-map workflow run fields are unsupported"));
    }
}

#[test]
fn shell_braces_do_not_hide_a_later_workflow_run_block() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".github/workflows/new.yml",
        "jobs:\n  check:\n    defaults:\n      run:\n        shell: bash\n    steps:\n      - run: |\n          echo '${{ toJSON(matrix) }}'\n          echo '{ braces }'\n      - run: echo clean\n",
    );
    check_fixture(&root).unwrap();
}

#[test]
fn exact_command_exception_is_scoped_to_its_path_and_occurrence_count() {
    let (_temporary, root) = fixture();
    let policy = POLICY.replace(
        "\"allowed_invocations\": []",
        "\"allowed_invocations\": [{\"path\":\".github/workflows/allowed.yml\",\"line\":\"uv run --locked project\",\"occurrences\":1,\"purpose\":\"owned locked external environment\"}]",
    );
    write(&root, POLICY_PATH, &policy);
    write(
        &root,
        ".github/workflows/allowed.yml",
        "jobs:\n  check:\n    steps:\n      - run: uv run --locked project\n",
    );
    check_fixture(&root).unwrap();
    write(
        &root,
        ".github/workflows/allowed.yml",
        "jobs:\n  check:\n    steps:\n      - run: uv run --locked other-project\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("unreviewed Python tooling invocation"));
}

#[test]
fn production_rust_literal_interpreter_process_is_rejected() {
    let error = invocations::check_rust_process_invocations(
        "crates/example/src/lib.rs",
        b"fn run() { std::process::Command::new(\"python3\").arg(\"-m\").arg(\"module\"); }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("production Rust process invocation"));
}

#[test]
fn production_rust_shell_python_script_is_rejected() {
    let error = invocations::check_rust_process_invocations(
        "crates/example/src/lib.rs",
        b"fn run() { std::process::Command::new(\"sh\").arg(\"-c\").arg(\"python3 scripts/probe.py\"); }",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("embeds a Python command"));
}

#[test]
fn production_rust_env_python_launcher_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "crates/example/build.rs",
        "fn main() { std::process::Command::new(\"env\").arg(\"py\").arg(\"-3\"); }",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("production Rust shell or env process"));
}

#[test]
fn windows_python_executable_is_detected() {
    assert!(invocations::has_python_tool_invocation(
        "python3.exe -I -m probe"
    ));
}

#[test]
fn comments_and_test_data_do_not_count_as_invocations() {
    assert!(!invocations::has_python_tool_invocation(
        "# python3 -c 'print(1)'"
    ));
    assert!(!invocations::has_python_tool_invocation(
        "echo \"python3 -c print(1)\""
    ));
    invocations::check_rust_process_invocations(
        "crates/example/src/lib.rs",
        b"#[cfg(test)] mod tests { fn fixture() { std::process::Command::new(\"python3\"); } }",
    )
    .unwrap();
}

#[test]
fn javascript_action_process_invocation_is_detected() {
    assert!(
        !invocations::javascript_source_python_invocations(
            "spawnSync('python3', ['-c', 'print(1)']);"
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        invocations::javascript_source_python_invocations(
            "// spawnSync('python3', ['-c', 'print(1)']);"
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn javascript_literal_shell_commands_and_tool_launches_are_checked() {
    for source in [
        "execSync('python3 -c probe');",
        "spawnSync('uvx', ['probe']);",
    ] {
        assert!(
            !invocations::javascript_source_python_invocations(source)
                .unwrap()
                .is_empty()
        );
    }
    assert!(
        invocations::javascript_source_python_invocations("execSync('echo python3');")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn multiline_javascript_process_invocation_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        ".github/actions/new-check/index.js",
        "spawnSync(\n  'python3',\n  ['-c', 'print(1)'],\n);\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains(".github/actions/new-check/index.js"));
    assert!(error.contains("unreviewed Python tooling invocation"));
}

#[test]
fn javascript_string_and_comment_examples_remain_data() {
    let source = r#"
const sample = "spawnSync('python3')";
// spawnSync('python3')
/*
execFileSync('python3');
*/
"#;
    assert!(
        invocations::javascript_source_python_invocations(source)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn env_shebang_assignment_before_python_is_rejected() {
    let (_temporary, root) = fixture();
    write(
        &root,
        "skills/new-tool/run",
        "#!/usr/bin/env -S TOOL_MODE=check python3 -I\nprint('x')\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("Python shebang"));
}

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

fn pinned_external_fixture() -> (TempDir, PathBuf) {
    let (temporary, root) = fixture();
    let manifest = "bench/harness/pyproject.toml";
    let lock = "bench/harness/uv.lock";
    let consumer = "just/harness.just";
    let command = "uv sync --locked --project bench/harness";
    write(
        &root,
        manifest,
        "[project]\nname = \"upstream-harness\"\nversion = \"0.1.0\"\ndependencies = [\"example==1.2.3\"]\n",
    );
    write(
        &root,
        lock,
        "version = 1\n[[package]]\nname = \"example\"\nversion = \"1.2.3\"\nsource = { url = \"https://example.test/example.whl\" }\nwheels = [{ hash = \"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" }]\n",
    );
    write(&root, consumer, &format!("harness:\n    {command}\n"));
    let digest = |path: &str| {
        crate::product::digest::file_sha256(&root.join(path))
            .map_err(|failure| failure.error)
            .unwrap()
    };
    let mut policy: serde_json::Value = serde_json::from_str(POLICY).unwrap();
    policy["external_dependency_inputs"] = serde_json::json!([{
        "manifest_path": manifest,
        "lock_path": lock,
        "manifest_sha256": digest(manifest),
        "lock_sha256": digest(lock),
        "consumer_path": consumer,
        "consumer_line": command,
        "purpose": "locked upstream harness wheel and dependency closure"
    }]);
    policy["allowed_invocations"] = serde_json::json!([{
        "path": consumer,
        "line": command,
        "occurrences": 1,
        "purpose": "restore the reviewed external environment"
    }]);
    write(
        &root,
        POLICY_PATH,
        &serde_json::to_string_pretty(&policy).unwrap(),
    );
    (temporary, root)
}

#[test]
fn hash_bound_external_wheel_inputs_are_admitted_without_running_the_harness() {
    let (_temporary, root) = pinned_external_fixture();
    check_fixture(&root).unwrap();
}

#[test]
fn external_wheel_manifest_and_lock_drift_require_reviewed_hashes() {
    for path in ["bench/harness/pyproject.toml", "bench/harness/uv.lock"] {
        let (_temporary, root) = pinned_external_fixture();
        check_fixture(&root).unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(root.join(path))
            .unwrap()
            .write_all(b"\n# changed input\n")
            .unwrap();
        let error = check_fixture(&root).unwrap_err().to_string();
        assert!(error.contains(path), "{error}");
        assert!(error.contains("reviewed hash"), "{error}");
    }
}

#[test]
fn external_dependency_command_must_remain_exact_and_locked() {
    let (_temporary, root) = pinned_external_fixture();
    write(
        &root,
        "just/harness.just",
        "harness:\n    uv sync --project bench/harness\n",
    );
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(
        error.contains("reviewed external dependency command"),
        "{error}"
    );
}

#[test]
fn approved_external_inputs_do_not_admit_repository_python_implementations() {
    let (_temporary, root) = pinned_external_fixture();
    write(&root, "bench/harness/probe.py", "print('not admitted')\n");
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(
        error.contains("repository Python source is forbidden"),
        "{error}"
    );
}

#[test]
fn external_dependency_inputs_refuse_oversize_before_hashing() {
    let (_temporary, root) = pinned_external_fixture();
    fs::write(
        root.join("bench/harness/uv.lock"),
        vec![b'x'; 16 * 1024 * 1024 + 1],
    )
    .unwrap();
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("at most 16 MiB"), "{error}");
}

#[cfg(unix)]
#[test]
fn external_dependency_inputs_refuse_linked_approved_bytes() {
    let (_temporary, root) = pinned_external_fixture();
    let lock = root.join("bench/harness/uv.lock");
    let original = root.join("original-lock");
    fs::rename(&lock, &original).unwrap();
    std::os::unix::fs::symlink(&original, &lock).unwrap();
    let error = check_fixture(&root).unwrap_err().to_string();
    assert!(error.contains("symlink refused"), "{error}");
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

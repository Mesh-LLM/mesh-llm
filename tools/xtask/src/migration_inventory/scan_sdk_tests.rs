//! Census reachability for native-admitted upstream SDK interpreter variables.
use super::*;
const MCP: &str = "skippy/crates/skippy-bench/src/evals/adapters/templates/mcp_atlas_run.sh";
const SWE: &str = "skippy/crates/skippy-bench/src/evals/adapters/templates/swe_bench_pro_run.sh";

#[test]
fn actual_module_and_stdin_sdk_leaves_are_executable_candidates() {
    for (path, source, expected) in [
        (
            MCP,
            include_str!(
                "../../../../skippy/crates/skippy-bench/src/evals/adapters/templates/mcp_atlas_run.sh"
            ),
            vec!["\"$SDK_PYTHON\" -I -B -m mcp_completion.main"],
        ),
        (
            SWE,
            include_str!(
                "../../../../skippy/crates/skippy-bench/src/evals/adapters/templates/swe_bench_pro_run.sh"
            ),
            vec![
                "\"$PREPARED_PYTHON\" -I -B - \"$INSTANCES\" \"$EXPERT_INSTANCES\" \"$DOCKER_PLATFORM\" <<'PY'",
                "\"$PREPARED_PYTHON\" -I -B -m sweagent.run.run run-batch \\",
            ],
        ),
    ] {
        let rows = scan_source_located(path, source);
        for expected in expected {
            let matching = rows
                .iter()
                .filter(|row| row.candidate.source_block == expected)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), 1, "{path}/{expected}");
            assert!(matching[0].candidate.executable, "{path}/{expected}");
            assert!(matching[0].line > 0);
        }
    }
}

#[test]
fn configured_sdk_tokens_do_not_approve_arbitrary_paths_or_metadata() {
    for path in [MCP, SWE, "scripts/foreign-sdk.sh"] {
        let source = "# \"$SDK_PYTHON\" -I -m mcp_completion.main\nSDK_PYTHON=/tmp/interpreter\necho \"$SDK_PYTHON\"\n\"$UNKNOWN_SDK\" -I -m foreign.module\n";
        assert!(scan_source(path, source).is_empty(), "{path}");
    }
    assert!(
        scan_source(
            "scripts/foreign-sdk.sh",
            "\"$SDK_PYTHON\" -I -m mcp_completion.main"
        )
        .is_empty()
    );
    assert!(scan_source(MCP, "\"$PREPARED_PYTHON\" -I -m sweagent.run.run").is_empty());
    assert!(scan_source(SWE, "\"$SDK_PYTHON\" -I -m mcp_completion.main").is_empty());
}

#[test]
fn configured_sdk_launch_stays_visible_when_interface_changes() {
    let original = scan_source(MCP, "\"$SDK_PYTHON\" -I -B -m mcp_completion.main");
    let changed = scan_source(MCP, "\"$SDK_PYTHON\" -I -B -m foreign.module");
    assert_eq!(original.len(), 1);
    assert_eq!(changed.len(), 1);
    assert!(changed[0].executable);
    assert_ne!(original[0].id, changed[0].id);
}

#[test]
fn existing_lowercase_interpreters_and_script_arguments_remain_visible() {
    let path = ".github/actions/check/action.yml";
    let source = "run: |\n  \"$sdk_python\" -c 'from openai import OpenAI'\n  \"$sdk_python\" -c 'from langchain_openai import ChatOpenAI'\n  \"$sdk_python\" -c 'from litellm import completion'\n  python3 scripts/ci-openai-embeddings-smoke.py\n";
    let rows = scan_source(path, source);
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|row| row.executable));
    assert!(rows.iter().all(|row| row.path == path));
    assert_eq!(
        rows.iter()
            .map(|row| &row.id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4
    );
}

#[test]
fn five_existing_sdk_leaves_keep_identities_and_truthfully_execute() {
    for (path, source, digests) in [
        (
            MCP,
            include_str!(
                "../../../../skippy/crates/skippy-bench/src/evals/adapters/templates/mcp_atlas_run.sh"
            ),
            vec!["b07ff933837b06e4", "c75e957750de6a6b"],
        ),
        (
            SWE,
            include_str!(
                "../../../../skippy/crates/skippy-bench/src/evals/adapters/templates/swe_bench_pro_run.sh"
            ),
            vec!["731c1b7c3c507458", "83c93f8f59564ce7", "54fe76481bb1d4b3"],
        ),
    ] {
        let rows = scan_source(path, source);
        for digest in digests {
            let id = format!("{path}#candidate:{digest}:1");
            let row = rows
                .iter()
                .find(|row| row.id == id)
                .expect("existing SDK candidate");
            assert!(row.executable, "{path}/{digest}");
            assert_eq!(
                rows.iter()
                    .filter(|row| row.id == format!("{path}#candidate:{digest}:1"))
                    .count(),
                1,
                "{path}/{digest}"
            );
        }
    }
}

#[test]
fn configured_interpreter_mentions_and_arguments_do_not_execute() {
    for path in [MCP, SWE] {
        let source = "echo \"$SDK_PYTHON module.py\"\nprintf '%s' 'PREPARED_PYTHON module.py'\n# SDK_PYTHON module.py describes a retained client\nSDK_PYTHON=module.py\nPREPARED_PYTHON=module.py\n";
        let rows = scan_source(path, source);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| !row.executable), "{path}");
    }
}

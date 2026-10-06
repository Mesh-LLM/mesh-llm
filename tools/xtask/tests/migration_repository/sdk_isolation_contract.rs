//! Exact existing client/isolation sources; not Python execution or full shell reachability.
use std::{collections::BTreeSet, fs, path::Path};
fn read(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap()
}
#[test]
fn sdk_four_clients_keep_only_ecosystem_imports_and_isolated_actual_invocations() {
    for (file, expected) in [
        (
            "openai-python",
            &["__future__", "argparse", "typing", "openai"][..],
        ),
        (
            "langchain-openai",
            &["__future__", "argparse", "typing", "langchain_openai"][..],
        ),
        (
            "litellm",
            &["__future__", "argparse", "typing", "litellm"][..],
        ),
        (
            "openai-embeddings",
            &[
                "__future__",
                "argparse",
                "base64",
                "math",
                "struct",
                "openai",
            ][..],
        ),
    ] {
        let source = read(&format!("scripts/ci-{file}-smoke.py"));
        // Closed import declarations in these exact four sources, not a general Python parser.
        let imports: BTreeSet<_> = source
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                line.strip_prefix("from ")
                    .or_else(|| line.strip_prefix("import "))
                    .map(|tail| tail.split_whitespace().next().unwrap())
            })
            .collect();
        assert_eq!(imports, expected.iter().copied().collect(), "{file}");
    }
    let compat = read("scripts/ci-compat-smoke.sh");
    for line in [
        "\"$SDK_PYTHON\" -I scripts/ci-openai-python-smoke.py --base-url \"$BASE_URL\"",
        "\"$SDK_PYTHON\" -I scripts/ci-litellm-smoke.py --base-url \"$BASE_URL\" --model \"$MODEL_ID\"",
        "\"$SDK_PYTHON\" -I scripts/ci-langchain-openai-smoke.py --base-url \"$BASE_URL\" --model \"$MODEL_ID\"",
    ] {
        assert_eq!(compat.lines().filter(|actual| *actual == line).count(), 1);
    }
    let embeddings = read("scripts/skippy-workload-certify.sh");
    assert_eq!(
        embeddings
            .matches("\"$SDK_PYTHON\" -I \"$ROOT/scripts/ci-openai-embeddings-smoke.py\"")
            .count(),
        1
    );
    assert!(embeddings.contains("if [[ \"$MODEL_CLASS\" == embedding ]]; then"));
    assert!(embeddings.contains("SDK_PYTHON=\"${SKIPPY_WORKLOAD_SDK_PYTHON:-}\""));
    assert!(compat.contains("SDK_PYTHON=\"${MESH_REQUIRED_SDK_PYTHON:-}\""));
}
#[test]
fn sdk_setup_keeps_closed_class_project_locks_and_preserved_required_cadence() {
    let setup = read(".github/actions/setup-canary-python/action.yml");
    for flag in [
        "uv sync --locked --project \"$PWD/ci/canary-python\" --python \"$host_python\"",
        "UV_PYTHON_DOWNLOADS=never",
        "-I -m venv",
        "-I -m pip --isolated install",
        "--require-hashes --no-deps --only-binary=:all:",
        "--index-url https://pypi.org/simple",
        "-r \"$PWD/ci/required-sdk-python/requirements.lock\"",
        "-I -m pip --isolated check",
    ] {
        assert!(setup.contains(flag), "{flag}");
    }
    let smoke = read(".github/workflows/smoke.yml");
    assert_eq!(smoke.matches("sdk-kind: compatibility").count(), 2);
    for name in [
        "Dense OpenAI client compatibility smoke",
        "Recurrent OpenAI client compatibility smoke",
    ] {
        assert_eq!(smoke.matches(name).count(), 2);
    }
    for project in ["ci/required-sdk-python", "ci/canary-python"] {
        let declaration = read(&format!("{project}/pyproject.toml"));
        assert!(declaration.contains("openai>=1.0,<3"));
        assert!(!read(&format!("{project}/uv.lock")).is_empty());
    }
    let lock = read("ci/required-sdk-python/requirements.lock");
    for dependency in ["openai==", "langchain-openai==", "litellm=="] {
        assert!(lock.lines().any(|line| line.starts_with(dependency)));
    }
    assert!(lock.contains("--hash=sha256:"));
}

fn corpus_isolated_declarations(script: &str, just: &str) -> bool {
    let body = |name: &str| {
        script
            .split_once(&format!("def {name}("))
            .map(|(_, tail)| tail.split("\ndef ").next().unwrap_or(tail))
    };
    let Some(probe) = body("python_has_duckdb") else {
        return false;
    };
    let Some(query) = body("run_duckdb_json") else {
        return false;
    };
    let Some(required) = body("require_duckdb") else {
        return false;
    };
    let Some(caller) = just
        .lines()
        .find(|line| line.contains("scripts/generate-bench-corpus.py"))
    else {
        return false;
    };
    probe.contains("[python, \"-I\", \"-c\", \"import duckdb\"]")
        && query.contains("command = [sys.executable, \"-I\", \"-c\", code]")
        && query.contains("require_duckdb()")
        && !query.contains("[\"uv\"")
        && required.contains("if python_has_duckdb(sys.executable):")
        && !required.contains("command_exists")
        && required.contains("locked DuckDB environment")
        && [
            "uv run",
            "--offline",
            "--locked",
            "--no-sync",
            "--no-python-downloads",
            "ci/agentic-replay-nightly",
            "python -I",
            "{{ tier }}",
            "{{ ARGS }}",
        ]
        .iter()
        .all(|flag| caller.contains(flag))
        && !caller.contains("--with")
}

#[test]
fn retained_corpus_locked_caller_and_children_refuse_unisolated_or_unpinned_declarations() {
    let script = read("scripts/generate-bench-corpus.py");
    let just = read("just/skippy.just");
    assert!(corpus_isolated_declarations(&script, &just));
    assert!(!corpus_isolated_declarations(
        &script.replace("\"-I\", ", ""),
        &just
    ));
    let fallback = script.replace(
        "command = [sys.executable, \"-I\", \"-c\", code]",
        "command = [\"uv\", \"run\", \"--with\", \"duckdb\", \"python\", \"-c\", code]",
    );
    assert!(!corpus_isolated_declarations(&fallback, &just));
    for flag in [
        "--offline",
        "--locked",
        "--no-sync",
        "--no-python-downloads",
        "python -I",
    ] {
        assert!(
            !corpus_isolated_declarations(&script, &just.replace(flag, "")),
            "{flag}"
        );
    }
    let project = read("ci/agentic-replay-nightly/pyproject.toml");
    let lock = read("ci/agentic-replay-nightly/uv.lock");
    assert!(project.contains("dependencies = [\"duckdb==1.4.5\"]"));
    assert!(lock.contains("name = \"duckdb\"\nversion = \"1.4.5\""));
    assert!(lock.contains("hash = \"sha256:"));
    // Source declaration, not installed-package or runtime/data qualification.
    let main = script
        .split_once("def main()")
        .map(|(_, body)| body)
        .unwrap()
        .split("\ndef ")
        .next()
        .unwrap();
    assert!(main.find("require_duckdb()").unwrap() < main.find("download_source(").unwrap());
}

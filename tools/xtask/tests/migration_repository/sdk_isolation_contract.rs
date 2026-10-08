//! Exact existing client/isolation sources; not Python execution or full shell reachability.
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};
fn sdk_root() -> &'static Path {
    static SOURCE: OnceLock<PathBuf> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mesh = super::support::repository_root();
        let result = super::support::Invocation {
            cwd: &mesh,
            args: &[
                "automation",
                "smoke-observation",
                "sdk-source",
                "--kind",
                "root",
            ],
            stdin: None,
            env: &[],
        }
        .run()
        .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let root = PathBuf::from(String::from_utf8(result.stdout).unwrap().trim());
        assert!(root.is_absolute() && root.is_dir());
        root
    })
}
fn read(path: &str) -> String {
    if (path.ends_with(".py") && path.starts_with("scripts/ci-"))
        || matches!(
            path,
            "ci/required-sdk-python/pyproject.toml"
                | "ci/required-sdk-python/uv.lock"
                | "ci/required-sdk-python/requirements.lock"
                | "ci/canary-python/pyproject.toml"
                | "ci/canary-python/uv.lock"
        )
    {
        return fs::read_to_string(sdk_root().join(path)).unwrap();
    }
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
    for client in ["openai", "litellm", "langchain"] {
        let invocation =
            format!("automation smoke-observation sdk-client --client {client} --python");
        assert_eq!(compat.matches(&invocation).count(), 1);
    }
    assert!(compat.contains("automation smoke-observation sdk-ready --base-url"));
    assert!(compat.contains("--timeout-secs \"$MAX_WAIT\""));
    assert!(!compat.contains("MODELS_JSON="));
    assert!(compat.contains("--max-time 5 --connect-timeout 5"));
    let embeddings = read("skippy/scripts/skippy-workload-certify.sh");
    assert_eq!(
        embeddings
            .matches("--client embeddings --python \"$SDK_PYTHON\"")
            .count(),
        1
    );
    assert!(embeddings.contains("--receipt \"$WORK_DIR/embedding-sdk.json\""));
    assert!(embeddings.contains("if [[ \"$MODEL_CLASS\" == embedding ]]; then"));
    assert!(embeddings.contains("SDK_PYTHON=\"${SKIPPY_WORKLOAD_SDK_PYTHON:-}\""));
    assert!(compat.contains("SDK_PYTHON=\"${MESH_REQUIRED_SDK_PYTHON:-}\""));
}
#[test]
fn sdk_setup_keeps_closed_class_project_locks_and_preserved_required_cadence() {
    let setup = read(".github/actions/setup-canary-python/action.yml");
    for flag in [
        "uv sync --locked --project \"$sdk_project\" --python \"$host_python\"",
        "UV_PYTHON_DOWNLOADS=never",
        "-I -m venv",
        "-I -m pip --isolated install",
        "--require-hashes --no-deps --only-binary=:all:",
        "--index-url https://pypi.org/simple",
        "-r \"$sdk_project/requirements.lock\"",
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

#[test]
fn native_corpus_recipe_uses_prepared_rust_dataset_owner() {
    let just = read("just/skippy.just");
    let recipe = just
        .split_once("bench-corpus tier=")
        .expect("maintained corpus recipe")
        .1
        .split("\n\n")
        .next()
        .unwrap();
    assert!(recipe.contains("just with-lld cargo run --locked -p trajectory-reader --features corpus-input --bin trajectory-reader -- corpus"));
    assert!(recipe.contains("{{ tier }}") && recipe.contains("{{ ARGS }}"));
    assert!(!recipe.contains("python") && !recipe.contains("duckdb") && !recipe.contains("uv run"));
    // The owning corpus library and supervised CLI tests execute selection,
    // immutable artifact admission, quota refusal and coherent publication.
}

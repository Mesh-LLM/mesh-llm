use super::tests::{matrix, policy};
use super::text_tests::scalar_argv;
use super::*;
use crate::automation::codepoint_json::parser;
use crate::automation::replay_matrix::input;

#[test]
fn builds_recurrent_argv_with_ordered_refs_parameters_and_root() {
    let replay = parser::parse(
        br#"{
        "mode":"all", "sessions_per_concurrency":30, "minimum_worker_waves":3,
        "minimum_context_tokens":262144, "minimum_session_prompt_tokens":65536,
        "min_isl":40000, "max_isl":196608, "min_turns":7, "passes":5,
        "warmup_turns":6, "max_output_tokens":4096, "concurrency":[8,1,4,2],
        "temperature":0, "seed":42, "backend":"metal",
        "selection_algorithm":"balanced-md5-v2"
    }"#,
    )
    .expect("distinct fixture");
    let policy = input::validate(&replay).unwrap_or_else(|_| panic!("valid distinct policy"));
    let models = matrix(
        r#"[{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"digest","class":"hybrid-recurrent","native_context_tokens":262144}]"#,
    );
    let selected = select(models.get("models"), &policy, &"dense".into()).expect("selected family");
    let refs = ["main=HEAD".into(), "base=old".into()];
    let invocation = ReplayInvocation {
        python: OsStr::new("/usr/bin/python3"),
        script: Path::new("/checkout/evals/agentic-replay.py"),
        dataset: Path::new("dataset.parquet"),
        output: Path::new("report"),
        worktree_root: Some(OsStr::new("/scratch/worktrees")),
        refs: &refs,
    };
    let actual = scalar_argv(argv(&selected, &policy, &invocation));
    let expected: Vec<OsString> = [
        "/usr/bin/python3",
        "/checkout/evals/agentic-replay.py",
        "run",
        "--model",
        "org/model@pin/one.gguf",
        "--backend",
        "metal",
        "--replay-mode",
        "all",
        "--expected-model-sha256",
        "digest",
        "--dataset-file",
        "dataset.parquet",
        "--output",
        "report",
        "--worktree-root",
        "/scratch/worktrees",
        "--ref",
        "main=HEAD",
        "--ref",
        "base=old",
        "--sessions-per-concurrency",
        "30",
        "--minimum-worker-waves",
        "3",
        "--minimum-context-tokens",
        "262144",
        "--minimum-session-prompt-tokens",
        "65536",
        "--min-isl",
        "40000",
        "--max-isl",
        "196608",
        "--min-turns",
        "7",
        "--passes",
        "5",
        "--warmup-turns",
        "6",
        "--max-output-tokens",
        "4096",
        "--concurrency",
        "8",
        "--concurrency",
        "1",
        "--concurrency",
        "4",
        "--concurrency",
        "2",
        "--require-recurrent-restores",
    ]
    .into_iter()
    .map(OsString::from)
    .collect();
    assert_eq!(actual, expected);
}

#[test]
fn omits_recurrent_and_empty_root_but_retains_duplicate_refs() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"hash","class":"moe","native_context_tokens":131072}]"#,
    );
    let selected = select(models.get("models"), &policy, &"dense".into()).expect("selected family");
    let refs = ["main=HEAD".into(), "main=HEAD".into()];
    let invocation = ReplayInvocation {
        python: OsStr::new("python3"),
        script: Path::new("/checkout/evals/agentic-replay.py"),
        dataset: Path::new("dataset"),
        output: Path::new("out"),
        worktree_root: Some(OsStr::new("")),
        refs: &refs,
    };
    let result = scalar_argv(argv(&selected, &policy, &invocation));
    assert!(!result.contains(&OsString::from("--worktree-root")));
    assert!(!result.contains(&OsString::from("--require-recurrent-restores")));
    assert_eq!(
        result
            .windows(2)
            .filter(|pair| pair[0] == OsStr::new("--ref"))
            .map(|pair| pair[1].clone())
            .collect::<Vec<_>>(),
        [OsString::from("main=HEAD"), OsString::from("main=HEAD")]
    );
}

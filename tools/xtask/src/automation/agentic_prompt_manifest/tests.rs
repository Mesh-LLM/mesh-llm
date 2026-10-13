use serde_json::json;

use super::document::{build_manifest, flatten_messages};
use super::selection::{Selection, Selector, Trajectory};

fn row(session: &str, isl: u64, tokens: u64, text: &str) -> Trajectory {
    Trajectory {
        session_id: session.into(),
        source_dataset: "source".into(),
        messages_json: json!([{"role":"user", "content":text}]).to_string(),
        n_turns: 20,
        max_isl: isl,
        total_tokens: tokens,
    }
}

fn selection(families: usize) -> Selection {
    Selection {
        sources: vec!["source".into()],
        families,
        min_isl: 8192,
        max_isl_exclusive: 12000,
        min_turns: 20,
    }
}

#[test]
fn flattens_roles_content_and_tool_calls() {
    let input = json!([
        {"role":"system","content":"rules","tool_calls_json":null},
        {"role":"assistant","content":"inspect","tool_calls_json":"[{\"id\": 1}]"}
    ])
    .to_string();
    assert_eq!(
        flatten_messages(&input).unwrap(),
        "<system>\nrules\n\n<assistant>\ninspect\n<tool_calls>[{\"id\": 1}]</tool_calls>"
    );
}

#[test]
fn interleaves_repeated_families_and_preserves_provenance() {
    let rows = [
        row("session-0", 9000, 9100, "prefix-0"),
        row("session-1", 9000, 9100, "prefix-1"),
    ];
    let policy = selection(2);
    let document = build_manifest(&rows, 2, "abc", &policy).unwrap();
    assert_eq!(
        document
            .prompts
            .iter()
            .map(|p| p.family.as_str())
            .collect::<Vec<_>>(),
        [
            "trajectory-0",
            "trajectory-1",
            "trajectory-0",
            "trajectory-1"
        ]
    );
    assert_eq!(
        document.prompts[2].prompt,
        "<user>\nprefix-0\n\n<user>\nBenchmark branch 1: summarize the latest repository state in one sentence."
    );
    assert_eq!(document.metadata.dataset_revision, "abc");
    assert_eq!(document.metadata.rows[1].session_id, "session-1");
    let metadata = serde_json::to_value(&document.metadata).unwrap();
    assert!(metadata["rows"][0].get("messages_json").is_none());
}

#[test]
fn rejects_malformed_message_shapes_and_zero_requests() {
    for input in [
        "[]",
        "{}",
        "[null]",
        "[{\"role\":7}]",
        "[{\"tool_calls_json\":{}}]",
    ] {
        assert!(flatten_messages(input).is_err(), "{input}");
    }
    assert!(build_manifest(&[row("a", 9000, 9100, "text")], 0, "abc", &selection(1)).is_err());
    assert_eq!(flatten_messages("[{}]").unwrap(), "<unknown>\n");
    assert_eq!(
        flatten_messages("[{\"content\":{\"z\":1,\"a\":\"é\"}}]").unwrap(),
        "<unknown>\n{\"a\": \"é\", \"z\": 1}"
    );
}

#[test]
fn selection_filters_boundaries_and_deduplicates_before_family_limit() {
    let policy = selection(1);
    let mut selector = Selector::new(&policy).unwrap();
    selector.observe(row("a", 8191, 99999, "below"));
    selector.observe(row("a", 12000, 99999, "above"));
    let mut excluded = row("a", 11999, 99999, "wrong source");
    excluded.source_dataset = "other".into();
    selector.observe(excluded);
    selector.observe(row("a", 8192, 20000, "eligible"));
    selector.observe(row("a", 9000, 10000, "higher ISL"));
    selector.observe(row("a", 9000, 11000, "higher tokens"));
    assert_eq!(
        selector.finish().unwrap(),
        [row("a", 9000, 11000, "higher tokens")]
    );
}

#[test]
fn bounded_stream_selection_is_independent_of_row_order() {
    let policy = selection(2);
    let rows = [
        row("a", 9000, 9100, "a"),
        row("b", 9000, 9100, "b"),
        row("c", 9000, 9100, "c"),
        row("a", 9100, 9200, "updated"),
    ];
    let select = |rows: Vec<Trajectory>| {
        let mut selector = Selector::new(&policy).unwrap();
        for row in rows {
            selector.observe(row);
        }
        selector.finish().unwrap()
    };
    let forward = select(rows.to_vec());
    let reversed = select(rows.into_iter().rev().collect());
    assert_eq!(forward, reversed);
    assert_eq!(
        forward
            .iter()
            .map(|r| r.session_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "c"]
    );
}

#[test]
fn insufficient_eligible_sessions_and_invalid_policy_fail() {
    assert!(Selector::new(&selection(0)).is_err());
    let policy = selection(2);
    let mut selector = Selector::new(&policy).unwrap();
    selector.observe(row("a", 9000, 9100, "a"));
    let mut too_short = row("b", 9000, 9100, "b");
    too_short.n_turns = 19;
    selector.observe(too_short);
    assert!(
        selector
            .finish()
            .unwrap_err()
            .to_string()
            .contains("selected 1 trajectories, expected 2")
    );
}

#[test]
fn prompt_manifest_serialization_preserves_hash_consumed_field_order() {
    let rows = [row("a", 9000, 9100, "é")];
    let policy = selection(1);
    let document = build_manifest(&rows, 1, "abc", &policy).unwrap();
    let mut actual = serde_json::to_string_pretty(&document).unwrap();
    actual.push('\n');
    let expected = r#"{
  "metadata": {
    "dataset": "thoughtworks/agentic-coding-trajectories",
    "dataset_revision": "abc",
    "selection": {
      "sources": [
        "source"
      ],
      "families": 1,
      "min_isl": 8192,
      "max_isl_exclusive": 12000,
      "min_turns": 20,
      "order": "md5(session_id)"
    },
    "requests_per_family": 1,
    "rows": [
      {
        "session_id": "a",
        "source_dataset": "source",
        "n_turns": 20,
        "max_isl": 9000,
        "total_tokens": 9100
      }
    ]
  },
  "prompts": [
    {
      "family": "trajectory-0",
      "prompt": "<user>\né\n\n<user>\nBenchmark branch 0: summarize the latest repository state in one sentence."
    }
  ]
}
"#;
    assert_eq!(actual.as_bytes(), expected.as_bytes());
}

#[test]
fn prompt_manifest_options_reject_duplicates_and_inconsistent_selection() {
    let valid = [
        "--dataset-file",
        "input.parquet",
        "--dataset-revision",
        "abc",
        "--output",
        "manifest.json",
        "--source-dataset",
        "source",
    ]
    .map(String::from)
    .to_vec();
    let parsed = super::options::Options::parse(&valid).unwrap();
    assert_eq!(parsed.selection.families, 8);
    assert_eq!(parsed.requests_per_family, 2);
    for extra in [
        vec!["--families", "0"],
        vec!["--output", "other.json"],
        vec!["--max-isl", "8000"],
        vec!["--requests-per-family", "0"],
        vec!["--unknown", "value"],
        vec!["--families"],
    ] {
        let mut args = valid.clone();
        args.extend(extra.into_iter().map(String::from));
        assert!(super::options::Options::parse(&args).is_err());
    }
}

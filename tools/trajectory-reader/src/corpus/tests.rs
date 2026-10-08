use super::{
    acquisition::{self, Format},
    config::Source,
    document, edit_loop, projections, sampling,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs};

fn source(adapter: &str) -> Source {
    Source {
        name: "fixture".into(),
        dataset: "owner/data".into(),
        config: "default".into(),
        split: "train".into(),
        revision: "a".repeat(40),
        family: "coding_edit".into(),
        adapter: adapter.into(),
        routing_hint: Some("ngram".into()),
        quota: BTreeMap::from([("smoke".into(), 2)]),
    }
}
#[test]
fn all_configured_and_dormant_projections_keep_outputs_and_filters() {
    let cases = [
        (
            "commitpack_edit",
            json!({"old_contents":"old","new_contents":"new","old_file":"x","lang":"rust"}),
            json!("new"),
        ),
        (
            "code_refinement",
            json!({"buggy":"old","fixed":"new"}),
            json!("new"),
        ),
        (
            "swe_bench_issue",
            json!({"problem_statement":"bug","repo":"org/r","patch":"diff"}),
            json!("diff"),
        ),
        (
            "apps_codegen",
            json!({"question":"code","solutions":"[solution]"}),
            json!("[solution]"),
        ),
        (
            "codesearchnet_explain",
            json!({"code":"code","comment":"explanation"}),
            json!("explanation"),
        ),
        (
            "xlam_tool_call",
            json!({"query":"request","tools":[{"name":"tool"}],"answers":["answer"]}),
            json!(["answer"]),
        ),
        (
            "spider_sql",
            json!({"db_schema":"schema","question":"ask","query":"SELECT 1","db_id":"db"}),
            json!("SELECT 1"),
        ),
        (
            "oasst_prompt",
            json!({"role":"prompter","lang":"en","text":"ask","message_id":"id"}),
            Value::Null,
        ),
        (
            "dolly_instruction",
            json!({"instruction":"ask","context":"context","response":"answer","category":"cat"}),
            json!("answer"),
        ),
        (
            "gsm8k_reasoning",
            json!({"question":"math","answer":"42"}),
            json!("42"),
        ),
        (
            "xsum_summarize",
            json!({"document":"article","summary":"short"}),
            json!("short"),
        ),
    ];
    for (adapter, row, expected) in cases {
        let output = document::normalize(&source(adapter), "smoke", 0, &row, 6000, None).unwrap();
        assert_eq!(output.len(), 1, "{adapter}");
        assert_eq!(output[0]["expected_output"], expected);
        assert_eq!(output[0]["source_revision"], "a".repeat(40));
        assert!(!output[0]["prompt"].as_str().unwrap().is_empty());
        assert!(projections::project(adapter, &json!({})).is_none());
    }
    assert!(
        projections::project(
            "oasst_prompt",
            &json!({"role":"assistant","lang":"en","text":"text"})
        )
        .is_none()
    );
    assert!(
        projections::project(
            "oasst_prompt",
            &json!({"role":"prompter","lang":"fr","text":"text"})
        )
        .is_none()
    );
}
#[test]
fn repeated_edit_sessions_keep_eight_turns_twelve_history_and_first_expected_only() {
    let mut messages = vec![json!({"role":"system","content":"private system"})];
    for index in 0..12 {
        messages.push(json!({"role":"user","content":format!("request {index}")}));
        messages.push(json!({"role":"assistant","content":format!("edit {index}")}));
    }
    let row = json!({"messages":messages,"patch":"diff","instance_id":"case","model":"model","resolved":true});
    let projected = edit_loop::project(&row).unwrap();
    assert_eq!(projected.prompts.len(), 8);
    assert!(!projected.prompts[0].contains("private system"));
    assert!(!projected.prompts[7].contains("request 0"));
    let rows = document::normalize(
        &source("swe_smith_trajectory_loop"),
        "coding-loop",
        3,
        &row,
        300,
        None,
    )
    .unwrap();
    assert_eq!(rows.len(), 8);
    assert_eq!(rows[0]["expected_output"], "diff");
    assert!(rows[1]["expected_output"].is_null());
    assert_eq!(rows[7]["metadata"]["loop_turns"], 8);
    assert_eq!(rows[0]["session_group"], "swe-smith:case");
    assert!(
        rows.iter()
            .all(|r| r["prompt"].as_str().unwrap().chars().count() <= 300)
    );
    assert!(
        edit_loop::project(
            &json!({"messages":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}]})
        )
        .is_none()
    );
    assert!(edit_loop::project(&json!({"messages":[{"role":"alien","content":"u"}]})).is_none());
}
#[test]
fn unicode_budgets_are_finite_and_expansion_is_explicit_stress() {
    let output = super::prompt_budget(&"étape\r\n".repeat(500), 300, None).unwrap();
    assert!(output.chars().count() <= 300);
    assert!(output.contains("truncated"));
    assert!(!output.contains('\r'));
    let expanded = super::prompt_budget("short", 500, Some(400)).unwrap();
    assert!(expanded.contains("not quality scoring"));
    assert!(expanded.chars().count() >= 400);
    assert!(super::prompt_budget("text", 20, None).is_err());
    assert!(super::prompt_budget("text", 500, Some(501)).is_err());
}
#[test]
fn raw_jsonl_sampling_is_seeded_deterministic_and_detects_replacement() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("train.jsonl");
    let rows = (0..100)
        .map(|n| json!({"buggy":n.to_string(),"fixed":"ok"}).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&path, rows).unwrap();
    let artifact = acquisition::inspect(&path, Format::Jsonl).unwrap();
    let source = source("code_refinement");
    let first = sampling::sample(std::slice::from_ref(&artifact), &source, 2458, 10).unwrap();
    assert_eq!(
        first,
        sampling::sample(std::slice::from_ref(&artifact), &source, 2458, 10).unwrap()
    );
    assert_ne!(
        first,
        sampling::sample(std::slice::from_ref(&artifact), &source, 2459, 10).unwrap()
    );
    fs::write(&path, "{}\n").unwrap();
    assert!(sampling::sample(&[artifact], &source, 2458, 10).is_err());
}
#[test]
fn pinned_repository_selection_has_exact_raw_layout_and_split_scope() {
    let mut source = source("commitpack_edit");
    source.dataset = "bigcode/commitpackft".into();
    source.config = "rust".into();
    assert_eq!(
        acquisition::requested_files(&source, &["data/rust/data.jsonl".into()]).unwrap()[0].0,
        "data/rust/data.jsonl"
    );
    assert!(acquisition::requested_files(&source, &["data/python/data.jsonl".into()]).is_err());
    source.dataset = "codeparrot/apps".into();
    source.config = "all".into();
    source.split = "test".into();
    assert_eq!(
        acquisition::requested_files(&source, &["test.jsonl".into()]).unwrap()[0].0,
        "test.jsonl"
    );
    source.dataset = "owner/parquet".into();
    source.config = "small".into();
    assert_eq!(
        acquisition::requested_files(
            &source,
            &[
                "small/test-000.parquet".into(),
                "medium/test-000.parquet".into(),
                "small/train-000.parquet".into()
            ]
        )
        .unwrap()
        .len(),
        1
    );
}
fn local_command(root: &std::path::Path, quota: usize) -> Vec<String> {
    let path = root.join("input.jsonl");
    fs::write(
        &path,
        "{\"buggy\":\"bad\",\"fixed\":\"good\"}\n{\"buggy\":\"bad2\",\"fixed\":\"good2\"}\n",
    )
    .unwrap();
    let artifact = acquisition::inspect(&path, Format::Jsonl).unwrap();
    let mut source = source("code_refinement");
    source.quota.insert("smoke".into(), quota);
    fs::write(
        root.join("config.json"),
        json!({"schema_version":1,"seed":2458,"tiers":{"smoke":{}},"sources":[source]}).to_string(),
    )
    .unwrap();
    fs::write(root.join("artifacts.json"),json!({"schema_version":1,"sources":[{"dataset":"owner/data","revision":"a".repeat(40),"config":"default","split":"train","conversion_provenance":{"tool":"fixture","source_revision":"a".repeat(40)},"artifacts":[{"path":"input.jsonl","format":"jsonl","sha256":artifact.sha256,"bytes":artifact.bytes}]}]}).to_string()).unwrap();
    [
        "smoke".into(),
        "--config".into(),
        root.join("config.json").display().to_string(),
        "--artifact-manifest".into(),
        root.join("artifacts.json").display().to_string(),
        "--out-root".into(),
        root.join("output").display().to_string(),
    ]
    .to_vec()
}
#[test]
fn local_command_publishes_bound_provenance_and_refuses_quota_or_corruption() {
    let root = tempfile::tempdir().unwrap();
    let args = local_command(root.path(), 2);
    super::cli::run(&args).unwrap();
    let output = root.path().join("output/smoke");
    let bytes = fs::read(output.join("corpus.jsonl")).unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["row_count"], 2);
    assert_eq!(manifest["corpus_sha256"], super::digest(&bytes));
    assert_eq!(manifest["sampling_algorithm"], sampling::ALGORITHM);
    assert!(super::cli::run(&args).is_err());
    assert_eq!(fs::read(output.join("corpus.jsonl")).unwrap(), bytes);
    fs::write(root.path().join("input.jsonl"), "{}\n").unwrap();
    assert!(super::cli::run(&args).is_err());
    assert_eq!(fs::read(output.join("corpus.jsonl")).unwrap(), bytes);
    let root = tempfile::tempdir().unwrap();
    let args = local_command(root.path(), 3);
    assert!(super::cli::run(&args).is_err());
    assert!(!root.path().join("output").exists());
}
#[test]
fn generated_compressed_parquet_rows_follow_same_native_projection() {
    use parquet::{
        basic::Compression,
        data_type::ByteArray,
        file::{properties::WriterProperties, writer::SerializedFileWriter},
        schema::parser::parse_message_type,
    };
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("rows.parquet");
    let schema = std::sync::Arc::new(
        parse_message_type(
            "message corpus { REQUIRED BINARY buggy (UTF8); REQUIRED BINARY fixed (UTF8); }",
        )
        .unwrap(),
    );
    let props = std::sync::Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build(),
    );
    let mut writer =
        SerializedFileWriter::new(fs::File::create(&path).unwrap(), schema, props).unwrap();
    let mut group = writer.next_row_group().unwrap();
    for values in [["bad", "bad2"], ["good", "good2"]] {
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<parquet::data_type::ByteArrayType>()
            .write_batch(&values.map(ByteArray::from), None, None)
            .unwrap();
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
    let artifact = acquisition::inspect(&path, Format::Parquet).unwrap();
    let source = source("code_refinement");
    let rows = sampling::sample(&[artifact], &source, 2458, 2).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| {
        document::normalize(&source, "smoke", 0, row, 6000, None)
            .unwrap()
            .len()
            == 1
    }));
}
#[test]
fn conversion_manifest_refuses_foreign_revision_unsafe_paths_and_missing_provenance() {
    let root = tempfile::tempdir().unwrap();
    local_command(root.path(), 2);
    let path = root.path().join("artifacts.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for change in [
        json!({"revision":"b".repeat(40)}),
        json!({"conversion_provenance":{}}),
        json!({"artifacts":[{"path":"../input.jsonl","bytes":0,"sha256":"bad","format":"jsonl"}]}),
    ] {
        let mut manifest = original.clone();
        manifest["sources"][0]
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        fs::write(&path, manifest.to_string()).unwrap();
        assert!(
            acquisition::acquire(&source("code_refinement"), root.path(), Some(&path)).is_err()
        );
    }
}

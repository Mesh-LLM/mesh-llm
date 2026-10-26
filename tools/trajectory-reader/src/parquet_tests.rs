use std::{fs::File, sync::Arc};

use parquet::{
    basic::Compression,
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};

#[path = "../tests/parquet_fixture.rs"]
mod parquet_fixture;
use parquet_fixture::write_parquet;

use super::{
    parquet_input,
    selection::{Selection, Trajectory},
};

fn policy(families: usize) -> Selection {
    Selection {
        sources: vec!["source".into()],
        families,
        min_isl: 8192,
        max_isl_exclusive: 12000,
        min_turns: 20,
    }
}

fn row(id: &str, isl: u64, content: &str) -> Trajectory {
    Trajectory {
        session_id: id.into(),
        source_dataset: "source".into(),
        messages_json: serde_json::json!([{"role":"user", "content":content}]).to_string(),
        n_turns: 20,
        max_isl: isl,
        total_tokens: 13000,
    }
}

#[test]
fn native_parquet_reads_compressed_row_groups_and_selects_best_duplicate() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trajectories.parquet");
    let groups = [
        vec![
            row("c", 9000, "c"),
            row("a", 8192, "initial"),
            row("b", 12000, "excluded"),
        ],
        vec![row("a", 9500, "selected"), row("d", 8191, "excluded")],
    ];
    for compression in [
        Compression::UNCOMPRESSED,
        Compression::SNAPPY,
        Compression::ZSTD(Default::default()),
    ] {
        write_parquet(&path, &groups, compression);
        let selected = parquet_input::select(&path, &policy(2)).unwrap();
        assert_eq!(selected, [row("a", 9500, "selected"), row("c", 9000, "c")]);
    }
}

#[test]
fn malformed_parquet_and_missing_selection_columns_fail() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.parquet");
    std::fs::write(&path, b"not a parquet file").unwrap();
    assert!(parquet_input::select(&path, &policy(1)).is_err());
    let schema =
        Arc::new(parse_message_type("message missing { REQUIRED INT64 max_isl; }").unwrap());
    let writer = SerializedFileWriter::new(
        File::create(&path).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    writer.close().unwrap();
    assert!(
        parquet_input::select(&path, &policy(1))
            .unwrap_err()
            .to_string()
            .contains("all six selection columns")
    );
}

#[test]
fn excluded_or_nullable_filter_rows_do_not_require_prompt_body_admission() {
    use parquet::record::{Field, Row};
    for (source, turns, isl) in [
        ("other", Field::Long(20), Field::Long(9000)),
        ("source", Field::Null, Field::Long(9000)),
        ("source", Field::Long(20), Field::Null),
        ("source", Field::Long(-1), Field::Long(9000)),
        ("source", Field::Long(20), Field::Long(12000)),
    ] {
        let row = Row::new(vec![
            ("source_dataset".into(), Field::Str(source.into())),
            ("n_turns".into(), turns),
            ("max_isl".into(), isl),
            ("messages_json".into(), Field::Null),
        ]);
        assert!(
            parquet_input::trajectory(row, &policy(1))
                .unwrap()
                .is_none()
        );
    }
    let malformed_eligible = Row::new(vec![
        ("source_dataset".into(), Field::Str("source".into())),
        ("n_turns".into(), Field::Long(20)),
        ("max_isl".into(), Field::Long(9000)),
        ("messages_json".into(), Field::Null),
    ]);
    assert!(parquet_input::trajectory(malformed_eligible, &policy(1)).is_err());
}

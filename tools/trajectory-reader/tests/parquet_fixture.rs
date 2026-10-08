use crate::selection::Trajectory;
use parquet::{
    basic::Compression,
    data_type::{ByteArray, ByteArrayType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use std::{fs::File, path::Path, sync::Arc};

pub fn write_parquet(path: &Path, groups: &[Vec<Trajectory>], compression: Compression) {
    let schema = parse_message_type(
        "message trajectories {
        REQUIRED BINARY session_id (UTF8);
        REQUIRED BINARY source_dataset (UTF8);
        REQUIRED BINARY messages_json (UTF8);
        REQUIRED INT64 n_turns;
        REQUIRED INT64 max_isl;
        REQUIRED INT64 total_tokens;
        REQUIRED BINARY unrelated_column (UTF8);
    }",
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut writer = SerializedFileWriter::new(
        File::create(path).unwrap(),
        Arc::new(schema),
        Arc::new(properties),
    )
    .unwrap();
    for rows in groups {
        let mut group = writer.next_row_group().unwrap();
        let texts = [
            rows.iter()
                .map(|r| r.session_id.as_str())
                .collect::<Vec<_>>(),
            rows.iter().map(|r| r.source_dataset.as_str()).collect(),
            rows.iter().map(|r| r.messages_json.as_str()).collect(),
        ];
        for values in texts {
            let values = values.into_iter().map(ByteArray::from).collect::<Vec<_>>();
            let mut column = group.next_column().unwrap().unwrap();
            column
                .typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .unwrap();
            column.close().unwrap();
        }
        for values in [
            rows.iter().map(|r| r.n_turns as i64).collect::<Vec<_>>(),
            rows.iter().map(|r| r.max_isl as i64).collect(),
            rows.iter().map(|r| r.total_tokens as i64).collect(),
        ] {
            let mut column = group.next_column().unwrap().unwrap();
            column
                .typed::<Int64Type>()
                .write_batch(&values, None, None)
                .unwrap();
            column.close().unwrap();
        }
        let mut ignored = group.next_column().unwrap().unwrap();
        ignored
            .typed::<ByteArrayType>()
            .write_batch(&vec![ByteArray::from("unrelated"); rows.len()], None, None)
            .unwrap();
        ignored.close().unwrap();
        assert!(group.next_column().unwrap().is_none());
        group.close().unwrap();
    }
    writer.close().unwrap();
}

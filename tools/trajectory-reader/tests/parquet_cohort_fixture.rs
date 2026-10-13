use parquet::{
    basic::Compression,
    data_type::{ByteArray, ByteArrayType, Int64Type},
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    schema::parser::parse_message_type,
};
use std::{fs::File, path::Path, sync::Arc};
#[derive(Clone)]
pub struct Row {
    pub id: String,
    pub framework: String,
    pub body: String,
    pub isl: i64,
    pub turns: i64,
}
pub fn row(id: &str, framework: &str) -> Row {
    Row {id:id.into(),framework:framework.into(),body:serde_json::json!([
        {"role":"user","content":"é task"},{"role":"assistant","content":null,"tool_calls_json":"[{\"id\":\"call\",\"function\":{\"name\":\"read\",\"arguments\":\"{}\"}}]"},
        {"role":"tool","content":"result","tool_call_id":"call"},{"role":"assistant","content":"answer"}]).to_string(),isl:9000,turns:2}
}
pub fn write(path: &Path, rows: &[Row], compression: Compression) {
    let schema=parse_message_type("message trajectories { REQUIRED BINARY session_id (UTF8); REQUIRED BINARY source_dataset (UTF8); REQUIRED BINARY agent_framework (UTF8); REQUIRED BINARY recorded_model (UTF8); REQUIRED BINARY messages_json (UTF8); REQUIRED INT64 n_turns; REQUIRED INT64 max_isl; REQUIRED INT64 total_tokens; }").unwrap();
    let props = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let mut file = SerializedFileWriter::new(
        File::create(path).unwrap(),
        Arc::new(schema),
        Arc::new(props),
    )
    .unwrap();
    for rows in rows.chunks(3) {
        let mut group = file.next_row_group().unwrap();
        for texts in [
            rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            vec!["source"; rows.len()],
            rows.iter().map(|r| r.framework.as_str()).collect(),
            vec!["model"; rows.len()],
            rows.iter().map(|r| r.body.as_str()).collect(),
        ] {
            let values = texts.into_iter().map(ByteArray::from).collect::<Vec<_>>();
            let mut col = group.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>()
                .write_batch(&values, None, None)
                .unwrap();
            col.close().unwrap();
        }
        for values in [
            rows.iter().map(|r| r.turns).collect::<Vec<_>>(),
            rows.iter().map(|r| r.isl).collect(),
            vec![12000; rows.len()],
        ] {
            let mut col = group.next_column().unwrap().unwrap();
            col.typed::<Int64Type>()
                .write_batch(&values, None, None)
                .unwrap();
            col.close().unwrap();
        }
        group.close().unwrap();
    }
    file.close().unwrap();
}

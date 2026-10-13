use super::report_row::{COLUMNS, Row};
use crate::command::DynResult;
use std::io::Write;

pub(super) fn write(output: &mut impl Write, rows: &[Row]) -> DynResult<()> {
    if rows.is_empty() {
        output.write_all(b"\r\n")?;
        return Ok(());
    }
    record(
        output,
        &COLUMNS
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
    )?;
    for row in rows {
        let serialized = serde_json::to_value(row)?;
        let fields = COLUMNS
            .iter()
            .map(|name| match &serialized[*name] {
                serde_json::Value::Null => String::new(),
                serde_json::Value::String(text) => text.clone(),
                value => value.to_string(),
            })
            .collect::<Vec<_>>();
        record(output, &fields)?;
    }
    Ok(())
}

fn record(output: &mut impl Write, fields: &[String]) -> DynResult<()> {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            output.write_all(b",")?;
        }
        if field.contains([',', '"', '\r', '\n']) {
            write!(output, "\"{}\"", field.replace('"', "\"\""))?;
        } else {
            output.write_all(field.as_bytes())?;
        }
    }
    output.write_all(b"\r\n")?;
    Ok(())
}

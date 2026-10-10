//! Nine-field runner GPU observation. Identity fields enter only an irreversible digest.
use crate::command::DynResult;
use serde_json::{Value, json};
pub(super) fn fingerprint(bytes: &[u8]) -> DynResult<(Value, Value)> {
    let text = std::str::from_utf8(bytes)?;
    let fields = row(text)?;
    if fields.len() != 9 {
        return Err("runner GPU fingerprint must contain one9-field row".into());
    }
    let stable = [
        "name",
        "uuid",
        "compute_capability",
        "driver_version",
        "pci_bus_id",
    ];
    for (index, name) in stable.iter().enumerate() {
        if fields[index].is_empty() {
            return Err(format!("runner GPU fingerprint missing stable field:{name}").into());
        }
    }
    let identity = crate::automation::cohort_identity::digest(
        &json!({"uuid":fields[1],"pci_bus_id":fields[4]}),
    )?;
    Ok((
        json!({"name":fields[0],"compute_capability":fields[2],"driver_version":fields[3],"gpu_identity_sha256":identity}),
        json!({"pstate":fields[5],"temperature_c":fields[6],"sm_clock_mhz":fields[7],"memory_clock_mhz":fields[8]}),
    ))
}
fn row(text: &str) -> DynResult<Vec<String>> {
    let text = text.trim_end_matches(['\n', '\r']);
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut closed = false;
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                    closed = true;
                }
            }
            '"' if field.is_empty() && !closed => quoted = true,
            ',' if !quoted => {
                fields.push(field.trim().to_owned());
                field.clear();
                closed = false;
            }
            '\n' | '\r' if !quoted => return Err("runner GPU fingerprint requires one row".into()),
            _ if closed && !c.is_whitespace() => return Err("invalid GPU CSV quoting".into()),
            _ => field.push(c),
        }
    }
    if quoted {
        return Err("unterminated GPU CSV field".into());
    }
    fields.push(field.trim().to_owned());
    Ok(fields)
}

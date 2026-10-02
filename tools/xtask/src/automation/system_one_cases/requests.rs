use serde_json::{Value, json};
pub(super) const DEFAULT_STATE: &str = "I was charged twice this month.";
pub(super) fn noul(instructions: &str) -> Value {
    json!({"type":"noul","instructions":instructions})
}
pub(super) fn choice(count: usize, instructions: &str) -> Value {
    let criteria: serde_json::Map<String, Value> = (0..count)
        .map(|i| (format!("team_{i}"), json!(format!("responsibility {i}"))))
        .collect();
    json!({"type":"choice","instructions":instructions,"criteria":criteria})
}
pub(super) fn score(count: usize) -> Value {
    json!({"type":"score","instructions":"How urgent is this?","criteria":(0..count).map(|i|format!("level {i}")).collect::<Vec<_>>()})
}
pub(super) fn request(model: &str, questions: Value, state: &str) -> Value {
    json!({"model":model,"questions":questions,"state":state})
}
pub(super) fn billing() -> Value {
    json!({"billing":noul("Is this a billing issue?")})
}

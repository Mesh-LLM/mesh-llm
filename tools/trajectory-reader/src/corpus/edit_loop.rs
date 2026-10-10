use super::{digest, field, text};
use serde_json::{Value, json};

pub(super) struct EditLoop {
    pub prompts: Vec<String>,
    pub expected: Value,
    pub metadata: Value,
    pub group: String,
}
pub(super) fn project(row: &Value) -> Option<EditLoop> {
    let messages = row["messages"].as_array()?;
    let mut transcript = Vec::new();
    let mut prompts = Vec::new();
    for message in messages {
        let role = message.get("role")?.as_str()?;
        if !["system", "user", "assistant", "tool", "function"].contains(&role) {
            return None;
        }
        let content = text(&message["content"]);
        if role == "system" || content.is_empty() {
            continue;
        }
        if role == "assistant" && !transcript.is_empty() {
            prompts.push(render(&transcript));
            if prompts.len() == 8 {
                break;
            }
        }
        transcript.push((role.to_owned(), content));
    }
    if prompts.len() < 2 {
        return None;
    }
    let key = ["instance_id", "traj_id"]
        .iter()
        .map(|key| field(row, key))
        .find(|s| !s.is_empty())
        .unwrap_or_else(|| digest(row.to_string().as_bytes())[..12].to_owned());
    Some(EditLoop {
        prompts,
        expected: row["patch"].clone(),
        metadata: json!({"instance_id":row["instance_id"],"traj_id":row["traj_id"],"model":row["model"],"resolved":row["resolved"]}),
        group: format!("swe-smith:{key}"),
    })
}
fn render(transcript: &[(String, String)]) -> String {
    let history = transcript
        .iter()
        .skip(transcript.len().saturating_sub(12))
        .map(|(role, content)| format!("{}:\n{content}", role.to_uppercase()))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "Continue this software engineering agent session.\n\nTranscript so far:\n{history}\n\nRespond with the next assistant action or code edit."
    )
}

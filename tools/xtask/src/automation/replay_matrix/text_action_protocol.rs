//! Explicit text-action adaptation for replay; never executes model commands.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub(super) enum Protocol {
    #[serde(rename = "shell-fence-v1")]
    ShellFenceV1,
}

const INSTRUCTIONS: &str = "Text-action interface (shell-fence-v1):\n\
You interact with the computer by submitting a shell command for the runner to execute. \
Each response must contain exactly one nonempty fenced shell command block, labelled bash or left unlabelled. \
You may explain your next action before the block. End your response immediately after its closing fence. \
Wait for the next user observation before taking another action. Never invent command results or write an observation yourself. \
If the task specifies a submission command, use it in the same command-block format when finished. \
Use a longer backtick fence if the shell command itself contains backtick fences.\n\
Required response format:\n\
Brief explanation of the next action.\n\
```bash\n\
<your shell command>\n\
```\n\
The backtick fences are literal Markdown syntax. Do not replace them with XML tags such as <shell> or <code_source>.";

impl Protocol {
    pub(super) fn apply(self, body: &mut Value) -> DynResult<()> {
        let messages = body["messages"].as_array_mut().ok_or("missing messages")?;
        if let Some(first) = messages.first_mut().filter(|m| m["role"] == "system") {
            let text = first["content"]
                .as_str()
                .ok_or("system content must be text")?;
            first["content"] = format!("{text}\n\n{INSTRUCTIONS}").into();
        } else {
            messages.insert(0, json!({"role":"system","content":INSTRUCTIONS}));
        }
        Ok(())
    }

    pub(super) fn content_check(self) -> fn(&str) -> Option<String> {
        shell_problem
    }
}

fn shell_problem(content: &str) -> Option<String> {
    validate_shell(content)
        .err()
        .map(|reason| format!("shell-fence-v1: {reason}"))
}

/// Structural action validation, not shell syntax or task-success evaluation.
fn validate_shell(content: &str) -> Result<(), &'static str> {
    let mut lines = content.lines();
    let opening = lines
        .by_ref()
        .find(|line| line.trim_start().starts_with("```"))
        .ok_or("missing shell command block")?
        .trim();
    let width = opening.bytes().take_while(|b| *b == b'`').count();
    if !matches!(opening[width..].trim(), "" | "bash" | "sh" | "shell") {
        return Err("command block must contain shell commands");
    }
    let mut command = false;
    let mut closed = false;
    for line in lines.by_ref() {
        let line = line.trim();
        if line.len() >= width && line.bytes().all(|b| b == b'`') {
            closed = true;
            break;
        }
        command |= !line.is_empty() && !line.starts_with('#');
    }
    if !closed {
        return Err("unterminated command block");
    }
    if !command {
        return Err("empty command block");
    }
    if lines.any(|line| !line.trim().is_empty()) {
        return Err("response continued after command; expected end of turn");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptation_preserves_history_and_is_explicit_in_turn_evidence() {
        use super::super::recorded_requests::{Selection, Trajectory, build};
        for system in [false, true] {
            let mut messages = vec![
                json!({"role":"user","content":"inspect /testbed"}),
                json!({"role":"assistant","content":"```\nls\n```"}),
                json!({"role":"user","content":"Observation: files"}),
                json!({"role":"assistant","content":"```\nsubmit\n```"}),
            ];
            if system {
                messages.insert(0, json!({"role":"system","content":"original system"}));
            }
            let original = json!({"session_id":"s","source_dataset":"capture","agent_framework":"swe-agent",
                "text_action_protocol":"shell-fence-v1","messages":messages});
            let trajectory: Trajectory = serde_json::from_value(original.clone()).unwrap();
            let turns = build(
                &trajectory,
                &Selection {
                    model: "m",
                    maximum_output_tokens: None,
                    turn_limit: None,
                    qualification_probe: false,
                },
            )
            .unwrap();
            assert_eq!(trajectory.original, original);
            for (i, turn) in turns.iter().enumerate() {
                assert_eq!(
                    serde_json::to_value(turn).unwrap()["text_action_protocol"],
                    "shell-fence-v1"
                );
                assert!(turn.body.get("text_action_protocol").is_none());
                assert!(turn.body.get("max_tokens").is_none());
                let body = turn.body["messages"].as_array().unwrap();
                assert_eq!(
                    body[0]["content"]
                        .as_str()
                        .unwrap()
                        .matches(INSTRUCTIONS)
                        .count(),
                    1
                );
                assert_eq!(
                    &body[1..],
                    &messages[usize::from(system)..(usize::from(system) + 1 + i * 2)]
                );
            }
            let mut conflicting = original;
            conflicting["tools"] = json!([{"type":"function","function":{"name":"bash"}}]);
            let conflicting: Trajectory = serde_json::from_value(conflicting).unwrap();
            assert!(
                build(
                    &conflicting,
                    &Selection {
                        model: "m",
                        maximum_output_tokens: None,
                        turn_limit: None,
                        qualification_probe: false
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn accepts_one_action_including_submission_and_embedded_fences() {
        for content in [
            "Inspect first.\n```bash\nls /testbed\n```",
            "```\nsubmit\n```\n",
            "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            "````bash\ncat > x <<'EOF'\n```python\nprint(1)\n```\nEOF\n````",
        ] {
            assert_eq!(shell_problem(content), None, "{content}");
        }
    }

    #[test]
    fn rejects_incident_shapes_truncation_and_simulated_observations() {
        for content in [
            "I'll inspect the repository.",
            "<code_source><result></result></code_source>",
            "```\n\n```",
            "```bash\n# inspect\n```",
            "```python\nprint(1)\n```",
            "```bash\nls",
            "```bash\nls\n```\nObservation: files",
            "```bash\nls\n```\n```bash\ncat x\n```",
        ] {
            assert!(shell_problem(content).is_some(), "{content}");
        }
    }
}

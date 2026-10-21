//! The system prompt: short on purpose. Small mesh models follow a few lines
//! better than goose's ~6.7 KB default (see block/buzz-app #941, where Gemma
//! wrote tool calls as fenced text under the long prompt).

use std::path::Path;

use crate::skills::{self, Skill};

const AGENTS_MD_LIMIT: usize = 8_000;

pub fn system_prompt(cwd: &Path, skills: &[Skill]) -> String {
    let mut prompt = format!(
        "You are a command-line agent on {os}. Working directory: {cwd}.\n\
         Use the shell tool to inspect and change things; do not guess file contents.\n\
         Make one tool call at a time and check its result before the next.\n\
         When the task is done, reply with a short plain-text answer and no tool call.",
        os = std::env::consts::OS,
        cwd = cwd.display(),
    );
    if let Some(index) = skills::index(skills) {
        prompt.push_str("\n\n");
        prompt.push_str(&index);
    }
    if let Ok(hints) = std::fs::read_to_string(cwd.join("AGENTS.md")) {
        let mut cut = hints.len().min(AGENTS_MD_LIMIT);
        while !hints.is_char_boundary(cut) {
            cut -= 1;
        }
        prompt.push_str("\n\n# Project notes (AGENTS.md)\n\n");
        prompt.push_str(&hints[..cut]);
    }
    prompt
}

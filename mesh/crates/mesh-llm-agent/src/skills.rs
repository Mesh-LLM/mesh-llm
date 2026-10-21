//! Skill discovery and the `load_skill` tool.
//!
//! Mirrors goose's skills: the same search roots (project then global, first
//! name wins), `SKILL.md` frontmatter `name`/`description`, and the same tool
//! name, schema and wording, so a model sees the same tool under mesh as under
//! goose.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use rmcp::model::{CallToolResult, ContentBlock as Content, Tool};

pub const NAME: &str = "load_skill";

#[derive(Clone, Debug)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

fn roots(cwd: &Path, home: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = vec![
        cwd.join(".agents/skills"),
        cwd.join(".goose/skills"),
        cwd.join(".claude/skills"),
    ];
    if let Some(home) = home {
        roots.push(home.join(".agents/skills"));
        roots.push(home.join(".config/goose/skills"));
        roots.push(home.join(".claude/skills"));
        roots.push(home.join(".config/agents/skills"));
    }
    roots
}

pub fn discover(cwd: &Path, home: Option<&Path>) -> Vec<Skill> {
    let mut seen = HashSet::new();
    let mut skills = Vec::new();
    for root in roots(cwd, home) {
        let Ok(entries) = fs::read_dir(&root) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for dir in dirs {
            let Ok(text) = fs::read_to_string(dir.join("SKILL.md")) else {
                continue;
            };
            let (name, description) = frontmatter(&text);
            let name = name.unwrap_or_else(|| {
                dir.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            if !name.is_empty() && seen.insert(name.clone()) {
                skills.push(Skill {
                    name,
                    description: description.unwrap_or_default(),
                    dir,
                });
            }
        }
    }
    skills
}

/// `name` and `description` from YAML frontmatter, single-line values only.
fn frontmatter(text: &str) -> (Option<String>, Option<String>) {
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return (None, None);
    }
    let (mut name, mut description) = (None, None);
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            let value = value.trim().trim_matches(['"', '\'']).to_string();
            match key.trim() {
                "name" => name = Some(value),
                "description" => description = Some(value),
                _ => {}
            }
        }
    }
    (name, description)
}

fn body(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---") else {
        return text;
    };
    rest.find("\n---")
        .map(|end| rest[end + 4..].trim_start_matches(['-', '\n', '\r']))
        .unwrap_or(text)
}

/// The prompt section listing available skills, in goose's wording.
pub fn index(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from(
        "# Skills\n\nYou have these skills at your disposal. Load one with load_skill when it can help with the task:\n",
    );
    for skill in skills {
        out.push_str(&format!("- {}: {}\n", skill.name, skill.description));
    }
    Some(out)
}

pub fn tool() -> Tool {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": { "type": "string", "description": "Skill name, or skill-name/path for a supporting file." }
        }
    });
    Tool::new(
        NAME,
        "Load a skill's full content into your context so you can follow its instructions.",
        schema.as_object().cloned().unwrap_or_default(),
    )
}

pub fn load(skills: &[Skill], name: &str) -> CallToolResult {
    let (skill_name, file) = match name.split_once('/') {
        Some((skill, file)) => (skill, Some(file)),
        None => (name, None),
    };
    let Some(skill) = skills.iter().find(|s| s.name == skill_name) else {
        let known: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        return CallToolResult::error(vec![Content::text(format!(
            "Skill \"{skill_name}\" not found. Available: {}",
            known.join(", ")
        ))]);
    };
    match file {
        None => match fs::read_to_string(skill.dir.join("SKILL.md")) {
            Ok(text) => CallToolResult::success(vec![Content::text(format!(
                "# Skill: {}\nSupporting files are relative to {}\n\n{}",
                skill.name,
                skill.dir.display(),
                body(&text)
            ))]),
            Err(error) => CallToolResult::error(vec![Content::text(error.to_string())]),
        },
        Some(file) => supporting_file(skill, file),
    }
}

fn supporting_file(skill: &Skill, file: &str) -> CallToolResult {
    let path = skill.dir.join(file);
    let inside = match (path.canonicalize(), skill.dir.canonicalize()) {
        (Ok(path), Ok(root)) => path.starts_with(root),
        _ => false,
    };
    if !inside {
        return CallToolResult::error(vec![Content::text(format!(
            "\"{file}\" is not a file inside skill \"{}\"",
            skill.name
        ))]);
    }
    match fs::read_to_string(&path) {
        Ok(text) => CallToolResult::success(vec![Content::text(text)]),
        Err(error) => CallToolResult::error(vec![Content::text(error.to_string())]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, dir: &str, name: &str) {
        let dir = root.join(dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: 'does {name}'\n---\n\nBody of {name}\n"),
        )
        .unwrap();
        fs::write(dir.join("ref.md"), "reference").unwrap();
    }

    #[test]
    fn project_skills_win_and_load_with_supporting_files() {
        let cwd = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        write_skill(&cwd.path().join(".agents/skills"), "a", "alpha");
        write_skill(&home.path().join(".agents/skills"), "a2", "alpha");
        write_skill(&home.path().join(".config/goose/skills"), "b", "beta");
        let skills = discover(cwd.path(), Some(home.path()));
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta"]);
        assert!(skills[0].dir.starts_with(cwd.path()));
        assert!(index(&skills).unwrap().contains("- beta: does beta"));

        let loaded = load(&skills, "alpha");
        let text = &loaded.content[0].as_text().unwrap().text;
        assert!(text.contains("Body of alpha") && !text.contains("description:"));
        let file = load(&skills, "alpha/ref.md");
        assert_eq!(file.content[0].as_text().unwrap().text, "reference");
    }

    #[test]
    fn rejects_paths_outside_the_skill() {
        let cwd = tempfile::tempdir().unwrap();
        write_skill(&cwd.path().join(".agents/skills"), "a", "alpha");
        fs::write(cwd.path().join("secret"), "x").unwrap();
        let skills = discover(cwd.path(), None);
        assert_eq!(load(&skills, "alpha/../../../secret").is_error, Some(true));
        assert_eq!(load(&skills, "missing").is_error, Some(true));
    }
}

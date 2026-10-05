use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const MARKER_FILE: &str = ".mesh-llm-skill.json";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkillAgent {
    Global,
    Goose,
    Pi,
    Codex,
    Opencode,
    Claude,
}

impl SkillAgent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Goose => "goose",
            Self::Pi => "pi",
            Self::Codex => "codex",
            Self::Opencode => "opencode",
            Self::Claude => "claude",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkillTarget {
    pub agent: SkillAgent,
    pub root: PathBuf,
    pub detected: bool,
    pub detection_reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkillPackage {
    pub provider_name: String,
    pub provider_version: String,
    pub name: String,
    pub source_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct SkillInstallOptions {
    pub home_dir: PathBuf,
    pub agents: Vec<SkillAgent>,
    pub detected_only: bool,
    pub dry_run: bool,
    pub force: bool,
}

impl SkillInstallOptions {
    pub fn from_env() -> Result<Self> {
        let home_dir = dirs::home_dir().context("Cannot determine home directory")?;
        Ok(Self {
            home_dir,
            agents: Vec::new(),
            detected_only: true,
            dry_run: false,
            force: false,
        })
    }

    pub fn for_agent(agent: SkillAgent) -> Result<Self> {
        let mut options = Self::from_env()?;
        options.agents = vec![agent];
        Ok(options)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkillInstallReport {
    pub available_skills: usize,
    pub targets: Vec<SkillTarget>,
    pub actions: Vec<SkillInstallAction>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SkillInstallAction {
    pub agent: SkillAgent,
    pub skill_name: String,
    pub provider_name: String,
    pub source_dir: PathBuf,
    pub destination_dir: PathBuf,
    pub status: SkillInstallStatus,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SkillInstallStatus {
    Installed,
    Updated,
    Unchanged,
    WouldInstall,
    WouldUpdate,
    WouldSkipConflict,
    SkippedConflict,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ManagedSkillMarker {
    source_provider: String,
    source_skill: String,
    provider_version: String,
}

pub fn install_skills(
    skills: &[SkillPackage],
    options: &SkillInstallOptions,
) -> Result<SkillInstallReport> {
    let targets = resolve_targets(options);
    let mut actions = Vec::new();

    for target in &targets {
        for skill in skills {
            actions.push(install_skill_to_target(skill, target, options)?);
        }
    }

    Ok(SkillInstallReport {
        available_skills: skills.len(),
        targets,
        actions,
    })
}

pub fn resolve_targets(options: &SkillInstallOptions) -> Vec<SkillTarget> {
    let agents = if options.agents.is_empty() {
        vec![
            SkillAgent::Global,
            SkillAgent::Goose,
            SkillAgent::Pi,
            SkillAgent::Codex,
            SkillAgent::Opencode,
            SkillAgent::Claude,
        ]
    } else {
        options.agents.clone()
    };

    let mut targets = agents
        .into_iter()
        .map(|agent| skill_target(agent, &options.home_dir))
        .filter(|target| !options.detected_only || target.detected)
        .collect::<Vec<_>>();
    deduplicate_targets_by_root(&mut targets);
    targets.sort_by(|left, right| {
        skill_agent_sort_rank(left.agent)
            .cmp(&skill_agent_sort_rank(right.agent))
            .then(left.agent.as_str().cmp(right.agent.as_str()))
    });
    targets
}

fn deduplicate_targets_by_root(targets: &mut Vec<SkillTarget>) {
    let mut seen = HashSet::new();
    targets.retain(|target| seen.insert(target.root.clone()));
}

fn skill_agent_sort_rank(agent: SkillAgent) -> usize {
    match agent {
        SkillAgent::Global => 0,
        SkillAgent::Claude => 1,
        SkillAgent::Codex => 2,
        SkillAgent::Goose => 3,
        SkillAgent::Opencode => 4,
        SkillAgent::Pi => 5,
    }
}

pub fn is_valid_skill_name(value: &str) -> bool {
    let mut previous_hyphen = false;
    if value.is_empty() || value.len() > 64 || value.starts_with('-') || value.ends_with('-') {
        return false;
    }
    for ch in value.chars() {
        let valid = ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-';
        if !valid || (ch == '-' && previous_hyphen) {
            return false;
        }
        previous_hyphen = ch == '-';
    }
    true
}

fn install_skill_to_target(
    skill: &SkillPackage,
    target: &SkillTarget,
    options: &SkillInstallOptions,
) -> Result<SkillInstallAction> {
    let destination_dir = target.root.join(&skill.name);
    let marker = ManagedSkillMarker {
        source_provider: skill.provider_name.clone(),
        source_skill: skill.name.clone(),
        provider_version: skill.provider_version.clone(),
    };
    let existing_marker = read_marker(&destination_dir)?;
    let status = classify_install(&destination_dir, existing_marker.as_ref(), &marker, options);

    if !options.dry_run {
        match status {
            SkillInstallStatus::Installed | SkillInstallStatus::Updated => {
                replace_skill_dir(&skill.source_dir, &destination_dir, &marker)?;
            }
            SkillInstallStatus::SkippedConflict | SkillInstallStatus::Unchanged => {}
            SkillInstallStatus::WouldInstall
            | SkillInstallStatus::WouldUpdate
            | SkillInstallStatus::WouldSkipConflict => unreachable!("dry-run status in live run"),
        }
    }

    Ok(SkillInstallAction {
        agent: target.agent,
        skill_name: skill.name.clone(),
        provider_name: skill.provider_name.clone(),
        source_dir: skill.source_dir.clone(),
        destination_dir,
        status,
    })
}

fn classify_install(
    destination_dir: &Path,
    existing_marker: Option<&ManagedSkillMarker>,
    marker: &ManagedSkillMarker,
    options: &SkillInstallOptions,
) -> SkillInstallStatus {
    if !destination_dir.exists() {
        return if options.dry_run {
            SkillInstallStatus::WouldInstall
        } else {
            SkillInstallStatus::Installed
        };
    }
    if existing_marker == Some(marker) && !options.force {
        return SkillInstallStatus::Unchanged;
    }
    if existing_marker
        .map(|existing| {
            existing.source_provider == marker.source_provider
                && existing.source_skill == marker.source_skill
        })
        .unwrap_or(options.force)
    {
        return if options.dry_run {
            SkillInstallStatus::WouldUpdate
        } else {
            SkillInstallStatus::Updated
        };
    }
    if options.dry_run {
        SkillInstallStatus::WouldSkipConflict
    } else {
        SkillInstallStatus::SkippedConflict
    }
}

fn replace_skill_dir(
    source_dir: &Path,
    destination_dir: &Path,
    marker: &ManagedSkillMarker,
) -> Result<()> {
    let parent = destination_dir.parent().with_context(|| {
        format!(
            "skill destination has no parent: {}",
            destination_dir.display()
        )
    })?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create skills directory {}", parent.display()))?;
    let temp_dir = parent.join(format!(
        ".mesh-llm-skill-{}-{}",
        std::process::id(),
        marker.source_skill
    ));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)
            .with_context(|| format!("remove stale temporary skill dir {}", temp_dir.display()))?;
    }
    copy_dir(source_dir, &temp_dir)?;
    write_marker(&temp_dir, marker)?;
    if destination_dir.exists() {
        fs::remove_dir_all(destination_dir)
            .with_context(|| format!("remove previous skill {}", destination_dir.display()))?;
    }
    fs::rename(&temp_dir, destination_dir).with_context(|| {
        format!(
            "install skill {} to {}",
            marker.source_skill,
            destination_dir.display()
        )
    })?;
    Ok(())
}

fn copy_dir(source_dir: &Path, destination_dir: &Path) -> Result<()> {
    fs::create_dir_all(destination_dir)
        .with_context(|| format!("create directory {}", destination_dir.display()))?;
    for entry in fs::read_dir(source_dir)
        .with_context(|| format!("read source directory {}", source_dir.display()))?
    {
        let entry = entry.with_context(|| format!("read source entry {}", source_dir.display()))?;
        let source = entry.path();
        let destination = destination_dir.join(entry.file_name());
        let file_type = entry
            .file_type()
            .with_context(|| format!("read file type for {}", source.display()))?;
        if file_type.is_dir() {
            copy_dir(&source, &destination)?;
        } else if file_type.is_file() {
            fs::copy(&source, &destination).with_context(|| {
                format!(
                    "copy skill file {} to {}",
                    source.display(),
                    destination.display()
                )
            })?;
        }
    }
    Ok(())
}

fn read_marker(skill_dir: &Path) -> Result<Option<ManagedSkillMarker>> {
    let marker_path = skill_dir.join(MARKER_FILE);
    if !marker_path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&marker_path)
        .with_context(|| format!("read skill marker {}", marker_path.display()))?;
    Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
        format!("parse skill marker {}", marker_path.display())
    })?))
}

fn write_marker(skill_dir: &Path, marker: &ManagedSkillMarker) -> Result<()> {
    let marker_path = skill_dir.join(MARKER_FILE);
    fs::write(&marker_path, serde_json::to_vec_pretty(marker)?)
        .with_context(|| format!("write skill marker {}", marker_path.display()))?;
    Ok(())
}

fn skill_target(agent: SkillAgent, home_dir: &Path) -> SkillTarget {
    let (root, config_dir) = match agent {
        SkillAgent::Global => (home_dir.join(".agents").join("skills"), None),
        SkillAgent::Goose => (
            home_dir.join(".agents").join("skills"),
            Some(home_dir.join(".config").join("goose")),
        ),
        SkillAgent::Pi => (
            home_dir.join(".pi").join("agent").join("skills"),
            Some(home_dir.join(".pi").join("agent")),
        ),
        SkillAgent::Codex => (
            home_dir.join(".agents").join("skills"),
            Some(home_dir.join(".codex")),
        ),
        SkillAgent::Opencode => (
            home_dir.join(".config").join("opencode").join("skills"),
            Some(home_dir.join(".config").join("opencode")),
        ),
        SkillAgent::Claude => (
            home_dir.join(".claude").join("skills"),
            Some(home_dir.join(".claude")),
        ),
    };
    let config_detected = config_dir.as_ref().is_some_and(|dir| dir.exists());
    let root_detected = root.exists();
    let detected = config_detected || root_detected;
    let detection_reason = if config_detected {
        config_dir.map(|dir| format!("found {}", dir.display()))
    } else if root_detected {
        Some(format!("found {}", root.display()))
    } else {
        None
    };

    SkillTarget {
        agent,
        root,
        detected,
        detection_reason,
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    fn skill(name: &str, source_dir: PathBuf) -> SkillPackage {
        SkillPackage {
            provider_name: "demo".to_string(),
            provider_version: "v1.0.0".to_string(),
            name: name.to_string(),
            source_dir,
        }
    }

    fn write_skill(root: &Path, name: &str) -> PathBuf {
        let skill_dir = root.join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Demo skill\n---\n"),
        )
        .unwrap();
        skill_dir
    }

    #[test]
    fn validates_agent_skill_names() {
        assert!(is_valid_skill_name("demo-skill-1"));
        assert!(!is_valid_skill_name("Demo"));
        assert!(!is_valid_skill_name("-demo"));
        assert!(!is_valid_skill_name("demo--skill"));
    }

    #[test]
    fn installs_skills_to_requested_agent_target() {
        let temp = TempDir::new().unwrap();
        let source_dir = write_skill(&temp.path().join("source"), "demo-skill");

        let options = SkillInstallOptions {
            home_dir: temp.path().join("home"),
            agents: vec![SkillAgent::Pi],
            detected_only: false,
            dry_run: false,
            force: false,
        };
        let report = install_skills(&[skill("demo-skill", source_dir)], &options).unwrap();

        assert_eq!(report.available_skills, 1);
        assert_eq!(report.actions[0].status, SkillInstallStatus::Installed);
        assert!(
            options
                .home_dir
                .join(".pi/agent/skills/demo-skill/SKILL.md")
                .exists()
        );
    }

    #[test]
    fn defaults_to_global_open_skill_target_once() {
        let temp = TempDir::new().unwrap();
        let home_dir = temp.path().join("home");
        fs::create_dir_all(home_dir.join(".agents/skills")).unwrap();
        fs::create_dir_all(home_dir.join(".codex")).unwrap();

        let options = SkillInstallOptions {
            home_dir: home_dir.clone(),
            agents: Vec::new(),
            detected_only: true,
            dry_run: true,
            force: false,
        };
        let targets = resolve_targets(&options);

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].agent, SkillAgent::Global);
        assert_eq!(targets[0].root, home_dir.join(".agents/skills"));
    }

    #[test]
    fn launch_time_agent_options_do_not_create_missing_skill_roots() {
        let temp = TempDir::new().unwrap();
        let source_dir = write_skill(&temp.path().join("source"), "demo-skill");
        let mut options = SkillInstallOptions::for_agent(SkillAgent::Goose).unwrap();
        options.home_dir = temp.path().join("home");

        let report = install_skills(&[skill("demo-skill", source_dir)], &options).unwrap();

        assert!(report.targets.is_empty());
        assert!(report.actions.is_empty());
        assert!(!options.home_dir.join(".agents").exists());
    }

    #[test]
    fn skips_user_owned_conflicts_without_force() {
        let temp = TempDir::new().unwrap();
        let source_dir = write_skill(&temp.path().join("source"), "demo-skill");

        let home_dir = temp.path().join("home");
        let existing = home_dir.join(".agents/skills/demo-skill");
        fs::create_dir_all(&existing).unwrap();
        fs::write(existing.join("SKILL.md"), "---\ndescription: mine\n---\n").unwrap();

        let options = SkillInstallOptions {
            home_dir,
            agents: vec![SkillAgent::Codex],
            detected_only: false,
            dry_run: false,
            force: false,
        };
        let report = install_skills(&[skill("demo-skill", source_dir)], &options).unwrap();

        assert_eq!(
            report.actions[0].status,
            SkillInstallStatus::SkippedConflict
        );
    }
}

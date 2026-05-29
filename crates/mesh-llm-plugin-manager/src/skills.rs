use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use mesh_llm_skills::{
    SkillAgent, SkillInstallOptions, SkillInstallReport, SkillPackage, install_skills,
    is_valid_skill_name,
};

use crate::store::{InstalledPluginMetadata, PluginStore, default_store_root};

#[derive(Clone, Debug)]
pub struct PluginSkillInstallOptions {
    pub store_root: PathBuf,
    pub skill_options: SkillInstallOptions,
}

impl PluginSkillInstallOptions {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            store_root: default_store_root()?,
            skill_options: SkillInstallOptions::from_env()?,
        })
    }

    pub fn for_agent(agent: SkillAgent) -> Result<Self> {
        Ok(Self {
            store_root: default_store_root()?,
            skill_options: SkillInstallOptions::for_agent(agent)?,
        })
    }
}

pub fn install_available_skills(options: &PluginSkillInstallOptions) -> Result<SkillInstallReport> {
    let store = PluginStore::new(&options.store_root);
    let skills = discover_plugin_skills(&store)?;
    install_skills(&skills, &options.skill_options)
}

pub fn discover_plugin_skills(store: &PluginStore) -> Result<Vec<SkillPackage>> {
    let mut skills = Vec::new();
    for plugin in store.list()? {
        if !plugin.enabled {
            continue;
        }
        append_plugin_root_skill(&mut skills, &plugin);
        append_plugin_skills_dir(&mut skills, &plugin)?;
    }
    skills.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.provider_name.cmp(&right.provider_name))
    });
    Ok(skills)
}

fn append_plugin_root_skill(skills: &mut Vec<SkillPackage>, plugin: &InstalledPluginMetadata) {
    let source_dir = plugin.install_path.clone();
    if !source_dir.join("SKILL.md").exists() {
        return;
    }
    skills.push(SkillPackage {
        provider_name: plugin.name.clone(),
        provider_version: plugin.installed_version.clone(),
        name: plugin.name.clone(),
        source_dir,
    });
}

fn append_plugin_skills_dir(
    skills: &mut Vec<SkillPackage>,
    plugin: &InstalledPluginMetadata,
) -> Result<()> {
    let skills_dir = plugin.install_path.join("skills");
    if !skills_dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&skills_dir)
        .with_context(|| format!("read plugin skills directory {}", skills_dir.display()))?
    {
        let entry =
            entry.with_context(|| format!("read plugin skill entry {}", skills_dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("read file type for {}", entry.path().display()))?;
        if !file_type.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !is_valid_skill_name(&name) {
            bail!(
                "plugin '{}' exposes invalid skill directory '{}'",
                plugin.name,
                name
            );
        }
        let source_dir = entry.path();
        if source_dir.join("SKILL.md").exists() {
            skills.push(SkillPackage {
                provider_name: plugin.name.clone(),
                provider_version: plugin.installed_version.clone(),
                name,
                source_dir,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::TempDir;

    use super::*;

    fn metadata(name: &str, install_path: PathBuf) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: format!("https://github.com/mesh-llm/{name}"),
            installed_version: "v1.0.0".to_string(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            downloaded_asset_name: format!("{name}-x86_64-unknown-linux-gnu.tar.gz"),
            install_path,
            enabled: true,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }

    fn write_skill(root: &Path, name: &str) {
        let skill_dir = root.join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: Demo skill\n---\n"),
        )
        .unwrap();
    }

    #[test]
    fn discovers_enabled_plugin_skills() {
        let temp = TempDir::new().unwrap();
        let install_path = temp.path().join("installed").join("demo");
        write_skill(&install_path, "demo-skill");

        let store = PluginStore::new(temp.path().join("store"));
        store.save(&metadata("demo", install_path)).unwrap();

        let skills = discover_plugin_skills(&store).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].provider_name, "demo");
        assert_eq!(skills[0].name, "demo-skill");
    }
}

use std::path::PathBuf;

use anyhow::{Context, Result};
use mesh_llm_skills::{
    SkillAgent, SkillInstallOptions, SkillInstallReport, SkillPackage, SkippedSkill,
    install_skills, is_valid_skill_name,
};

use crate::store::{InstalledPluginMetadata, PluginStore, default_store_root};

/// What `is_valid_skill_name` accepts, for skip reasons.
const SKILL_NAME_RULES: &str = "1 to 64 lowercase letters, digits, and hyphens, with no \
                                hyphen at the start or end and no two in a row";

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
    let discovered = collect_plugin_skills(&store)?;
    let mut report = install_skills(&discovered.skills, &options.skill_options)?;
    report.skipped = discovered.skipped;
    Ok(report)
}

pub fn discover_plugin_skills(store: &PluginStore) -> Result<Vec<SkillPackage>> {
    Ok(collect_plugin_skills(store)?.skills)
}

struct DiscoveredSkills {
    skills: Vec<SkillPackage>,
    skipped: Vec<SkippedSkill>,
}

fn collect_plugin_skills(store: &PluginStore) -> Result<DiscoveredSkills> {
    let mut discovered = DiscoveredSkills {
        skills: Vec::new(),
        skipped: Vec::new(),
    };
    for plugin in store.list()? {
        if !plugin.enabled {
            continue;
        }
        append_plugin_root_skill(&mut discovered, &plugin);
        append_plugin_skills_dir(&mut discovered, &plugin)?;
    }
    discovered.skills.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.provider_name.cmp(&right.provider_name))
    });
    Ok(discovered)
}

/// A root `SKILL.md` is installed under the plugin's name, which becomes a
/// directory in each agent's skills root, so the name has to pass the same
/// check as a `skills/` directory name. A plugin name can be valid without
/// being a valid skill name (`My_Tool`), so that skill is left out and
/// reported, and the other plugins' skills still install.
fn append_plugin_root_skill(discovered: &mut DiscoveredSkills, plugin: &InstalledPluginMetadata) {
    let source_dir = plugin.install_path.clone();
    if !source_dir.join("SKILL.md").exists() {
        return;
    }
    if !is_valid_skill_name(&plugin.name) {
        discovered.skipped.push(SkippedSkill {
            provider_name: plugin.name.clone(),
            skill_name: plugin.name.clone(),
            reason: format!(
                "the root SKILL.md is named after the plugin, and the plugin name is not a \
                 valid skill name ({SKILL_NAME_RULES})"
            ),
        });
        return;
    }
    discovered.skills.push(SkillPackage {
        provider_name: plugin.name.clone(),
        provider_version: plugin.installed_version.clone(),
        name: plugin.name.clone(),
        source_dir,
    });
}

/// Each `skills/<name>/SKILL.md` is installed as skill `<name>`. A directory
/// without a `SKILL.md` is not a skill and is ignored; one whose name is not a
/// valid skill name is left out and reported, like an invalid root skill, so
/// one plugin's bad skill does not stop the others from installing.
fn append_plugin_skills_dir(
    discovered: &mut DiscoveredSkills,
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
        let source_dir = entry.path();
        if !file_type.is_dir() || !source_dir.join("SKILL.md").exists() {
            continue;
        }
        // A name that is not UTF-8 is reported in its escaped form, so two
        // such names stay distinguishable.
        let file_name = entry.file_name();
        let name = file_name
            .to_str()
            .map_or_else(|| format!("{file_name:?}"), str::to_string);
        if !is_valid_skill_name(&name) {
            discovered.skipped.push(SkippedSkill {
                provider_name: plugin.name.clone(),
                skill_name: name,
                reason: format!(
                    "the skills/ directory name is not a valid skill name ({SKILL_NAME_RULES})"
                ),
            });
            continue;
        }
        discovered.skills.push(SkillPackage {
            provider_name: plugin.name.clone(),
            provider_version: plugin.installed_version.clone(),
            name,
            source_dir,
        });
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
            manifest: None,

            last_protocol_version: None,
            last_status: None,
            last_error: None,
            verified_sha256: None,
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

    #[test]
    fn root_skill_is_named_after_a_valid_plugin_name() {
        let temp = TempDir::new().unwrap();
        let install_path = temp.path().join("installed").join("demo");
        fs::create_dir_all(&install_path).unwrap();
        fs::write(install_path.join("SKILL.md"), "---\nname: demo\n---\n").unwrap();

        let store = PluginStore::new(temp.path().join("store"));
        store.save(&metadata("demo", install_path)).unwrap();

        let skills = discover_plugin_skills(&store).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "demo");
    }

    #[test]
    fn invalid_skill_names_are_skipped_and_reported_and_others_still_install() {
        let temp = TempDir::new().unwrap();
        // A root skill named after a plugin whose name is not a skill name,
        // a `skills/` directory with such a name, and a directory that holds
        // no skill at all.
        let odd_path = temp.path().join("installed").join("Demo_Plugin");
        fs::create_dir_all(odd_path.join("skills/.cache")).unwrap();
        fs::write(odd_path.join("SKILL.md"), "---\nname: demo\n---\n").unwrap();
        write_skill(&odd_path, "Bad_Name");
        let other_path = temp.path().join("installed").join("other");
        write_skill(&other_path, "other-skill");

        let store_root = temp.path().join("store");
        let store = PluginStore::new(&store_root);
        store.save(&metadata("Demo_Plugin", odd_path)).unwrap();
        store.save(&metadata("other", other_path)).unwrap();

        let report = install_available_skills(&PluginSkillInstallOptions {
            store_root,
            skill_options: SkillInstallOptions {
                home_dir: temp.path().join("home"),
                agents: vec![SkillAgent::Global],
                detected_only: false,
                dry_run: true,
                force: false,
            },
        })
        .unwrap();

        let installed: Vec<_> = report
            .actions
            .iter()
            .map(|action| action.skill_name.as_str())
            .collect();
        assert_eq!(installed, ["other-skill"]);
        let mut skipped: Vec<_> = report
            .skipped
            .iter()
            .map(|skipped| {
                assert!(skipped.reason.contains("not a valid skill name"));
                (skipped.provider_name.as_str(), skipped.skill_name.as_str())
            })
            .collect();
        skipped.sort();
        assert_eq!(
            skipped,
            [("Demo_Plugin", "Bad_Name"), ("Demo_Plugin", "Demo_Plugin")]
        );
    }

    #[test]
    fn root_skill_with_a_dot_name_written_into_metadata_is_skipped() {
        let temp = TempDir::new().unwrap();
        let install_path = temp.path().join("installed").join("demo");
        fs::create_dir_all(&install_path).unwrap();
        fs::write(install_path.join("SKILL.md"), "---\nname: demo\n---\n").unwrap();

        // `save` refuses `..`, but the store reads whatever name its metadata
        // file holds, so write that file by hand.
        let store_root = temp.path().join("store");
        let store = PluginStore::new(&store_root);
        store.save(&metadata("demo", install_path.clone())).unwrap();
        let metadata_file = store_root.join("demo").join("plugin-install.json");
        assert!(metadata_file.exists());
        fs::write(
            &metadata_file,
            serde_json::to_vec(&metadata("..", install_path)).unwrap(),
        )
        .unwrap();

        let discovered = collect_plugin_skills(&store).unwrap();
        assert!(discovered.skills.is_empty());
        assert_eq!(discovered.skipped.len(), 1);
        assert_eq!(discovered.skipped[0].skill_name, "..");
    }

    // APFS refuses file names that are not UTF-8, so macOS cannot create the
    // directory this test needs.
    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn a_skill_directory_name_that_is_not_utf8_is_reported_escaped() {
        use std::os::unix::ffi::OsStrExt;
        let temp = TempDir::new().unwrap();
        let install_path = temp.path().join("installed").join("demo");
        let name = std::ffi::OsStr::from_bytes(b"bad\xff");
        let skill_dir = install_path.join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "---\n---\n").unwrap();

        let store = PluginStore::new(temp.path().join("store"));
        store.save(&metadata("demo", install_path)).unwrap();

        let discovered = collect_plugin_skills(&store).unwrap();
        assert_eq!(discovered.skipped.len(), 1);
        assert_eq!(discovered.skipped[0].skill_name, r#""bad\xFF""#);
    }
}

use anyhow::Result;
use mesh_llm_plugin_manager::{
    PluginSkillInstallOptions, SkillAgent, SkillInstallReport, SkillInstallStatus,
    install_available_skills,
};

use mesh_llm_cli::{SkillAgentArg, SkillCommand};
use mesh_llm_tui::json_mode_enabled;

pub fn run_skills_command(command: &SkillCommand) -> Result<()> {
    match command {
        SkillCommand::Install {
            agent,
            all,
            dry_run,
            force,
        } => install(agent, *all, *dry_run, *force),
    }
}

pub fn install_skills_for_agent(agent: SkillAgent) {
    match PluginSkillInstallOptions::for_agent(agent).and_then(|options| {
        let report = install_available_skills(&options)?;
        Ok(report)
    }) {
        Ok(report) => print_agent_install_summary(agent, &report),
        Err(error) if !json_mode_enabled() => {
            eprintln!(
                "Could not install mesh plugin skills for {}: {error}",
                agent.as_str()
            );
        }
        Err(_) => {}
    }
}

fn install(agents: &[SkillAgentArg], all: bool, dry_run: bool, force: bool) -> Result<()> {
    let mut options = PluginSkillInstallOptions::from_env()?;
    options.skill_options.dry_run = dry_run;
    options.skill_options.force = force;
    if all {
        options.skill_options.detected_only = false;
    }
    if !agents.is_empty() {
        options.skill_options.agents = agents
            .iter()
            .copied()
            .map(skill_agent_arg_to_manager)
            .collect();
        options.skill_options.detected_only = false;
    }
    let report = install_available_skills(&options)?;
    print_install_report(&report, dry_run)?;
    Ok(())
}

fn skill_agent_arg_to_manager(agent: SkillAgentArg) -> mesh_llm_plugin_manager::SkillAgent {
    match agent {
        SkillAgentArg::Global => mesh_llm_plugin_manager::SkillAgent::Global,
        SkillAgentArg::Goose => mesh_llm_plugin_manager::SkillAgent::Goose,
        SkillAgentArg::Pi => mesh_llm_plugin_manager::SkillAgent::Pi,
        SkillAgentArg::Codex => mesh_llm_plugin_manager::SkillAgent::Codex,
        SkillAgentArg::Opencode => mesh_llm_plugin_manager::SkillAgent::Opencode,
        SkillAgentArg::Claude => mesh_llm_plugin_manager::SkillAgent::Claude,
    }
}

fn print_agent_install_summary(agent: SkillAgent, report: &SkillInstallReport) {
    if json_mode_enabled() {
        return;
    }
    let changed = report
        .actions
        .iter()
        .filter(|action| {
            matches!(
                action.status,
                SkillInstallStatus::Installed | SkillInstallStatus::Updated
            )
        })
        .count();
    if changed > 0 {
        eprintln!(
            "✅ Installed {changed} mesh plugin skill(s) for {}",
            agent.as_str()
        );
    }
}

fn print_install_report(report: &SkillInstallReport, dry_run: bool) -> Result<()> {
    if json_mode_enabled() {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    let heading = if dry_run {
        "🧪 Mesh plugin skill install preview"
    } else {
        "🧠 Installing mesh plugin skills"
    };
    eprintln!("{heading}");

    if report.available_skills == 0 {
        eprintln!("🔎 No plugin skills found in installed plugins.");
        eprintln!("📦 Plugins can expose skills with skills/<name>/SKILL.md.");
        return Ok(());
    }

    eprintln!(
        "📦 Found {}",
        plural_count(report.available_skills, "plugin skill")
    );

    if report.targets.is_empty() {
        eprintln!("🔎 No supported agent skill targets detected.");
        eprintln!("💡 Use --agent <agent> or --all to install anyway.");
        return Ok(());
    }

    eprintln!(
        "🎯 Targeting {}:",
        plural_count(report.targets.len(), "agent")
    );
    for target in &report.targets {
        let reason = target
            .detection_reason
            .as_deref()
            .unwrap_or("explicit target");
        eprintln!(
            "   • {:<8} {} ({reason})",
            target.agent.as_str(),
            target.root.display()
        );
    }

    eprintln!("🛠️  Applying skills:");
    for action in &report.actions {
        let Some(label) = action_status_label(&action.status, dry_run) else {
            continue;
        };
        eprintln!(
            "   {label:<17} {:<28} -> {:<8} {}",
            skill_display_name(action),
            action.agent.as_str(),
            action.destination_dir.display()
        );
    }

    print_install_summary(report, dry_run);
    Ok(())
}

fn print_install_summary(report: &SkillInstallReport, dry_run: bool) {
    let mut installed = 0;
    let mut updated = 0;
    let mut unchanged = 0;
    let mut conflicts = 0;
    for action in &report.actions {
        match action.status {
            SkillInstallStatus::Installed | SkillInstallStatus::WouldInstall => installed += 1,
            SkillInstallStatus::Updated | SkillInstallStatus::WouldUpdate => updated += 1,
            SkillInstallStatus::Unchanged => unchanged += 1,
            SkillInstallStatus::SkippedConflict | SkillInstallStatus::WouldSkipConflict => {
                conflicts += 1;
            }
        }
    }

    let verb = if dry_run { "planned" } else { "complete" };
    let mut parts = vec![
        count_label(installed, "installed", "installed"),
        count_label(updated, "updated", "updated"),
    ];
    if unchanged > 0 {
        parts.push(count_label(unchanged, "unchanged", "unchanged"));
    }
    if conflicts > 0 {
        parts.push(count_label(conflicts, "conflict", "conflicts"));
    }
    eprintln!("✅ Skill install {verb}: {}", parts.join(", "));
}

fn action_status_label(status: &SkillInstallStatus, dry_run: bool) -> Option<&'static str> {
    match status {
        SkillInstallStatus::Installed => Some("✅ installed"),
        SkillInstallStatus::Updated => Some("♻️  updated"),
        SkillInstallStatus::Unchanged if !dry_run => None,
        SkillInstallStatus::Unchanged => Some("⏭️  unchanged"),
        SkillInstallStatus::WouldInstall => Some("📝 would install"),
        SkillInstallStatus::WouldUpdate => Some("📝 would update"),
        SkillInstallStatus::WouldSkipConflict => Some("⚠️  would skip"),
        SkillInstallStatus::SkippedConflict => Some("⚠️  skipped"),
    }
}

fn skill_display_name(action: &mesh_llm_plugin_manager::SkillInstallAction) -> String {
    format!("{}/{}", action.provider_name, action.skill_name)
}

fn plural_count(count: usize, noun: &str) -> String {
    count_label(count, noun, &format!("{noun}s"))
}

fn count_label(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("{count} {singular}")
    } else {
        format!("{count} {plural}")
    }
}

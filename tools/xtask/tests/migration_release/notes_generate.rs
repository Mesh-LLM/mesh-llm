use crate::failure_diagnostics::failure_snapshot;
use crate::support::{Stage, TestResult, repo_root};
use std::fs;
use std::process::Command;
#[cfg(not(unix))]
use std::time::Duration;

#[cfg(unix)]
#[path = "generator_process.rs"]
mod generator_process;
#[cfg(unix)]
use generator_process::{GENERATOR_DEADLINE, run_generator};

#[cfg(not(unix))]
const GENERATOR_DEADLINE: Duration = Duration::from_secs(90);

#[cfg(not(unix))]
/// Original finite fixtures only; no deadline or process-tree guarantee.
fn run_generator(
    command: &mut Command,
    _stage: &Stage,
    _deadline: Duration,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    Ok(command.output()?)
}

#[test]
fn migration_release_generate_keeps_deterministic_notes_when_agent_plan_is_invalid() -> TestResult {
    let stage = Stage::new("generator-agent-fallback")?;
    let root = repo_root();
    let body = "## What's Changed\n* fix: repair cache by @alice in https://github.com/o/r/pull/1\n\n## New Contributors\n* @alice made their first contribution\n\n**Full Changelog**: compare\n";
    stage.write("published.md", body.as_bytes())?;
    stage.executable("bin/gh", "#!/bin/sh\ncase \"$1 $2\" in\n  'release view') cat \"$RELEASE_FIXTURE_ROOT/published.md\" ;;\n  *) exit 90 ;;\nesac\n")?;
    stage.executable("bin/git", "#!/bin/sh\ncase \"$1 $2 $3\" in\n  'rev-parse --show-toplevel '*) printf '%s\\n' \"$RELEASE_SOURCE_ROOT\" ;;\n  'log --reverse --format=%H%x1f%s%x1f%b%x1e') printf 'abc\\037fix: repair cache (#1)\\037\\036' ;;\n  'log --format=%s%x1f%b%x1e '*) printf 'fix: repair cache (#1)\\037\\036' ;;\n  *) exit 91 ;;\nesac\n")?;
    stage.executable("bin/opencode", "#!/bin/sh\ncase \"$1 $2\" in\n  'run --auto') case \"$*\" in\n    *'Reply with the single word: ready'*) printf 'ready\\n' ;;\n    *) printf '%s\\n' '{\"sections\":[{\"title\":\"Added\",\"prs\":[1,1]}]}' > \"$RELEASE_NOTES_WORKDIR/plan.agent.json\" ;;\n  esac ;;\n  *) exit 92 ;;\nesac\n")?;
    let path = format!(
        "{}:{}",
        stage.path().join("bin").display(),
        std::env::var("PATH")?
    );
    let output = run_generator(
        Command::new("bash")
            .arg(root.join("scripts/release-notes-generate.sh"))
            .current_dir(&root)
            .env("PATH", path)
            .env("RELEASE_FIXTURE_ROOT", stage.path())
            .env("RELEASE_SOURCE_ROOT", &root)
            .env("RELEASE_NOTES_WORKDIR", stage.path().join("work"))
            .env("RELEASE_TAG", "v1.0.1")
            .env("RELEASE_NOTES_BASE", "v1.0.0")
            .env("GITHUB_REPOSITORY", "o/r")
            .env("AGENT_MODEL", "fixture/model")
            .env("OPENCODE_API_KEY", "fixture")
            .env("DRY_RUN", "true"),
        &stage,
        GENERATOR_DEADLINE,
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains("agent plan failed validation; keeping deterministic notes"),
        "{stdout}"
    );
    assert!(stdout.contains("DRY_RUN=true; not editing the release"));
    let notes = fs::read_to_string(stage.path().join("work/notes.deterministic.md"))?;
    assert!(
        notes.contains("### Fixed\n\n* Repair cache by @alice in https://github.com/o/r/pull/1")
    );
    assert!(notes.ends_with("## New Contributors\n* @alice made their first contribution\n\n**Full Changelog**: compare\n"));
    assert!(!stage.path().join("work/notes.agent.md").exists());
    Ok(())
}

#[test]
fn migration_release_generate_ignores_stale_agent_plan_after_failed_review() -> TestResult {
    let stage = Stage::new("generator-stale-plan")?;
    let root = repo_root();
    stage.write("published.md", b"* fix: repair cache by @alice in https://github.com/o/r/pull/1\n**Full Changelog**: compare\n")?;
    stage.write(
        "work/plan.agent.json",
        b"{\"sections\":[{\"title\":\"Added\",\"prs\":[1}]}",
    )?;
    stage.executable("bin/gh", "#!/bin/sh\ncase \"$1 $2\" in\n  'release view') cat \"$RELEASE_FIXTURE_ROOT/published.md\" ;;\n  *) exit 90 ;;\nesac\n")?;
    stage.executable("bin/git", "#!/bin/sh\ncase \"$1 $2 $3\" in\n  'rev-parse --show-toplevel '*) printf '%s\\n' \"$RELEASE_SOURCE_ROOT\" ;;\n  'log --reverse --format=%H%x1f%s%x1f%b%x1e') printf 'abc\\037fix: repair cache (#1)\\037\\036' ;;\n  'log --format=%s%x1f%b%x1e '*) printf 'fix: repair cache (#1)\\037\\036' ;;\n  *) exit 91 ;;\nesac\n")?;
    stage.executable("bin/opencode", "#!/bin/sh\ncase \"$*\" in\n  *'Reply with the single word: ready'*) printf 'ready\\n' ;;\n  *) exit 99 ;;\nesac\n")?;
    let path = format!(
        "{}:{}",
        stage.path().join("bin").display(),
        std::env::var("PATH")?
    );
    let output = run_generator(
        Command::new("bash")
            .arg(root.join("scripts/release-notes-generate.sh"))
            .current_dir(&root)
            .env("PATH", path)
            .env("RELEASE_FIXTURE_ROOT", stage.path())
            .env("RELEASE_SOURCE_ROOT", &root)
            .env("RELEASE_NOTES_WORKDIR", stage.path().join("work"))
            .env("RELEASE_TAG", "v1.0.1")
            .env("RELEASE_NOTES_BASE", "v1.0.0")
            .env("GITHUB_REPOSITORY", "o/r")
            .env("AGENT_MODEL", "fixture/model")
            .env("OPENCODE_API_KEY", "fixture")
            .env("DRY_RUN", "true"),
        &stage,
        GENERATOR_DEADLINE,
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    assert!(
        stdout.contains(
            "agent review turn failed or exceeded its budget; keeping deterministic notes"
        )
    );
    assert!(stdout.contains("publishing deterministic notes"));
    assert!(!stage.path().join("work/plan.agent.json").exists());
    assert!(!stage.path().join("work/notes.agent.md").exists());
    Ok(())
}

#[test]
fn migration_release_generate_rejects_unapproved_publication_after_render() -> TestResult {
    let stage = Stage::new("generator-unapproved")?;
    let root = repo_root();
    stage.write("published.md", b"* fix: repair cache by @alice in https://github.com/o/r/pull/1\n**Full Changelog**: compare\n")?;
    stage.executable("bin/gh", "#!/bin/sh\ncase \"$1 $2\" in\n  'release view') cat \"$RELEASE_FIXTURE_ROOT/published.md\" ;;\n  *) exit 90 ;;\nesac\n")?;
    stage.executable("bin/git", "#!/bin/sh\ncase \"$1 $2 $3\" in\n  'rev-parse --show-toplevel '*) printf '%s\\n' \"$RELEASE_SOURCE_ROOT\" ;;\n  'log --reverse --format=%H%x1f%s%x1f%b%x1e') printf 'abc\\037fix: repair cache (#1)\\037\\036' ;;\n  'log --format=%s%x1f%b%x1e '*) printf 'fix: repair cache (#1)\\037\\036' ;;\n  *) exit 91 ;;\nesac\n")?;
    let path = format!(
        "{}:{}",
        stage.path().join("bin").display(),
        std::env::var("PATH")?
    );
    let output = run_generator(
        Command::new("bash")
            .arg(root.join("scripts/release-notes-generate.sh"))
            .current_dir(&root)
            .env("PATH", path)
            .env("RELEASE_FIXTURE_ROOT", stage.path())
            .env("RELEASE_SOURCE_ROOT", &root)
            .env("RELEASE_NOTES_WORKDIR", stage.path().join("work"))
            .env("RELEASE_TAG", "v1.0.1")
            .env("RELEASE_NOTES_BASE", "v1.0.0")
            .env("GITHUB_REPOSITORY", "o/r")
            .env_remove("AGENT_MODEL")
            .env_remove("RELEASE_NOTES_APPROVED")
            .env_remove("DRY_RUN"),
        &stage,
        GENERATOR_DEADLINE,
    )?;
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}",
        failure_snapshot(
            stage.path(),
            &output,
            &[
                "work/body.github.md",
                "work/body.md",
                "work/notes.deterministic.md"
            ]
        )
    );
    assert!(
        String::from_utf8(output.stderr)?
            .contains("refusing to edit a published release without approval")
    );
    let notes = fs::read_to_string(stage.path().join("work/notes.deterministic.md"))?;
    assert!(notes.contains("https://github.com/o/r/pull/1"));
    Ok(())
}

#[test]
fn migration_release_generate_times_out_agent_review_and_keeps_deterministic_notes() -> TestResult {
    let stage = Stage::new("generator-timeout")?;
    let root = repo_root();
    stage.write("published.md", b"* fix: repair cache by @alice in https://github.com/o/r/pull/1\n**Full Changelog**: compare\n")?;
    stage.executable("bin/gh", "#!/bin/sh\ncase \"$1 $2\" in\n  'release view') cat \"$RELEASE_FIXTURE_ROOT/published.md\" ;;\n  *) exit 90 ;;\nesac\n")?;
    stage.executable("bin/git", "#!/bin/sh\ncase \"$1 $2 $3\" in\n  'rev-parse --show-toplevel '*) printf '%s\\n' \"$RELEASE_SOURCE_ROOT\" ;;\n  'log --reverse --format=%H%x1f%s%x1f%b%x1e') printf 'abc\\037fix: repair cache (#1)\\037\\036' ;;\n  'log --format=%s%x1f%b%x1e '*) printf 'fix: repair cache (#1)\\037\\036' ;;\n  *) exit 91 ;;\nesac\n")?;
    stage.executable("bin/opencode", "#!/bin/sh\ncase \"$*\" in\n  *'Reply with the single word: ready'*) printf 'ready\\n' ;;\n  *) sleep 4 ;;\nesac\n")?;
    let path = format!(
        "{}:{}",
        stage.path().join("bin").display(),
        std::env::var("PATH")?
    );
    let output = run_generator(
        Command::new("bash")
            .arg(root.join("scripts/release-notes-generate.sh"))
            .current_dir(&root)
            .env("PATH", path)
            .env("RELEASE_FIXTURE_ROOT", stage.path())
            .env("RELEASE_SOURCE_ROOT", &root)
            .env("RELEASE_NOTES_WORKDIR", stage.path().join("work"))
            .env("RELEASE_TAG", "v1.0.1")
            .env("RELEASE_NOTES_BASE", "v1.0.0")
            .env("GITHUB_REPOSITORY", "o/r")
            .env("AGENT_MODEL", "fixture/model")
            .env("OPENCODE_API_KEY", "fixture")
            .env("AGENT_REVIEW_BUDGET_SECONDS", "1")
            .env("DRY_RUN", "true"),
        &stage,
        GENERATOR_DEADLINE,
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8(output.stdout)?.contains(
            "agent review turn failed or exceeded its budget; keeping deterministic notes"
        )
    );
    assert!(
        fs::read_to_string(stage.path().join("work/notes.deterministic.md"))?
            .contains("https://github.com/o/r/pull/1")
    );
    assert!(!stage.path().join("work/notes.agent.md").exists());
    Ok(())
}

//! Non-argv sources for a private-mesh invite token.
//!
//! `--join <token>` used to be the only way to supply an invite token. That
//! makes it impossible for the per-user service `mesh-llm setup` installs to
//! join a private mesh: the generated unit runs a bare `serve`, every extra
//! flag has to be hand-edited into the unit file, and argv is readable by any
//! process on the host. On a `[mesh_requirements]` mesh the token is also a
//! short-lived credential, so baking it into a persistent unit trades a leaked
//! secret for a silent expiry loop.
//!
//! The service already carries non-argv configuration: `setup` writes
//! `~/.config/mesh-llm/service.env`, systemd loads it through
//! `EnvironmentFile=-`, and the generated launchd runner sources it before
//! `exec serve`. This module gives the invite token a name to live under in
//! that file — `MESH_LLM_JOIN_FILE` (or `MESH_LLM_JOIN` for an inline
//! token) — plus `--join-file <PATH>` for foreground runs.
//!
//! File-backed tokens are re-read for every rejoin attempt, so rotating a
//! token is "write the new token to the file" and nothing else: no unit edit,
//! no restart, and the token never appears in argv.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::RuntimeOptions;

/// Inline invite token; equivalent to a single `--join <TOKEN>`.
pub const MESH_LLM_JOIN_ENV: &str = "MESH_LLM_JOIN";
/// Path to a file holding the invite token; equivalent to `--join-file <PATH>`.
pub const MESH_LLM_JOIN_FILE_ENV: &str = "MESH_LLM_JOIN_FILE";

/// Read a file-backed invite token.
///
/// The trimmed contents are the token and nothing more: invite tokens are
/// opaque signed values, so a `#` inside one is not a comment and a newline
/// is not a list separator.
fn read_join_token_file(path: &Path) -> std::result::Result<String, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            return Err(format!(
                "cannot read join token file {}: {error}; write the invite token to that file, \
                 point MESH_LLM_JOIN_FILE / --join-file somewhere else, or unset it",
                path.display()
            ));
        }
    };
    let token = raw.trim();
    if token.is_empty() {
        return Err(format!(
            "join token file {} is empty; write the invite token to that file",
            path.display()
        ));
    }
    Ok(token.to_owned())
}

fn push_unique(tokens: &mut Vec<String>, token: &str) {
    let token = token.trim();
    if token.is_empty() || tokens.iter().any(|existing| existing == token) {
        return;
    }
    tokens.push(token.to_owned());
}

/// Token files named by the process environment, in order.
fn environment_join_token_files() -> Vec<PathBuf> {
    std::env::var_os(MESH_LLM_JOIN_FILE_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .into_iter()
        .collect()
}

/// Resolve every invite token the runtime should try, plus one message per
/// unusable file-backed source.
///
/// Priority is argv, then `MESH_LLM_JOIN`, then `--join-file`, then
/// `MESH_LLM_JOIN_FILE`; duplicates are dropped. Returning messages instead
/// of an error keeps both callers honest: startup escalates them to a hard
/// failure, while the 60s rejoin loop reports each one once and keeps
/// retrying, so a long-running service cannot be stranded silently by a token
/// file it cannot read.
pub(crate) fn resolve_invite_token_sources(
    literals: &[String],
    inline_env: Option<&str>,
    join_files: &[PathBuf],
    environment_files: &[PathBuf],
) -> (Vec<String>, Vec<String>) {
    let mut tokens = Vec::new();
    for literal in literals {
        push_unique(&mut tokens, literal);
    }
    if let Some(inline) = inline_env {
        push_unique(&mut tokens, inline);
    }

    let mut errors = Vec::new();
    let mut files: Vec<PathBuf> = join_files.to_vec();
    for path in environment_files {
        if !files.contains(path) {
            files.push(path.clone());
        }
    }
    for path in files {
        match read_join_token_file(&path) {
            Ok(token) => push_unique(&mut tokens, &token),
            Err(message) => errors.push(message),
        }
    }

    (tokens, errors)
}

/// `resolve_invite_token_sources` against the live process environment.
pub(crate) fn resolve_invite_tokens(
    literals: &[String],
    join_files: &[PathBuf],
) -> (Vec<String>, Vec<String>) {
    let inline_env = std::env::var(MESH_LLM_JOIN_ENV).ok();
    let environment_files = environment_join_token_files();
    resolve_invite_token_sources(
        literals,
        inline_env.as_deref(),
        join_files,
        &environment_files,
    )
}

/// Fold every non-argv invite token source into `options.join`.
///
/// Startup is the place to be strict: a serve that cannot read its configured
/// token must say so rather than quietly running standalone, and the error
/// lands in the service log where an operator will see it.
pub(crate) fn apply_join_token_sources(options: &mut RuntimeOptions) -> Result<()> {
    let (tokens, errors) = resolve_invite_tokens(&options.join, &options.join_files);
    if let Some(first) = errors.first() {
        bail!("{first}");
    }
    options.join = tokens;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, contents).expect("token file should write");
        path
    }

    #[test]
    fn read_join_token_file_trims_surrounding_whitespace() {
        let temp = tempfile::tempdir().unwrap();
        let path = token_file(temp.path(), "invite.token", "  signed-token\n");

        assert_eq!(
            read_join_token_file(&path).expect("token should read"),
            "signed-token"
        );
    }

    #[test]
    fn read_join_token_file_rejects_empty_files() {
        let temp = tempfile::tempdir().unwrap();
        let path = token_file(temp.path(), "invite.token", "\n  \n");

        let error = read_join_token_file(&path).expect_err("empty token file must be rejected");
        assert!(error.contains("is empty"), "{error}");
        assert!(error.contains(&path.display().to_string()), "{error}");
    }

    #[test]
    fn resolve_invite_token_sources_orders_and_deduplicates_every_source() {
        let temp = tempfile::tempdir().unwrap();
        let flag_file = token_file(temp.path(), "flag.token", "file-token\n");
        let environment_file = token_file(temp.path(), "env.token", "env-file-token\n");

        let (tokens, errors) = resolve_invite_token_sources(
            &["argv-token".to_string(), "file-token".to_string()],
            Some("inline-token"),
            &[flag_file],
            &[environment_file],
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            tokens,
            ["argv-token", "file-token", "inline-token", "env-file-token",]
        );
    }

    #[test]
    fn resolve_invite_token_sources_reads_a_shared_path_once() {
        let temp = tempfile::tempdir().unwrap();
        let shared = token_file(temp.path(), "shared.token", "shared-token\n");

        let (tokens, errors) = resolve_invite_token_sources(
            &[],
            None,
            std::slice::from_ref(&shared),
            std::slice::from_ref(&shared),
        );

        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(tokens, ["shared-token"]);
    }

    #[test]
    fn resolve_invite_token_sources_reports_every_unusable_file() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing.token");
        let empty = token_file(temp.path(), "empty.token", "");

        let (tokens, errors) = resolve_invite_token_sources(
            &["argv-token".to_string()],
            None,
            &[missing.clone(), empty.clone()],
            &[],
        );

        assert_eq!(tokens, ["argv-token"]);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(
            errors[0].contains(&missing.display().to_string()),
            "{errors:?}"
        );
        assert!(
            errors[1].contains(&empty.display().to_string()),
            "{errors:?}"
        );
    }

    #[test]
    fn apply_join_token_sources_folds_file_tokens_into_options() {
        let temp = tempfile::tempdir().unwrap();
        let path = token_file(temp.path(), "invite.token", "signed-token\n");
        let mut options = RuntimeOptions {
            join_files: vec![path],
            ..RuntimeOptions::default()
        };

        apply_join_token_sources(&mut options).expect("token file should resolve");

        assert_eq!(options.join, ["signed-token"]);
    }

    #[test]
    fn apply_join_token_sources_fails_fast_on_an_unreadable_file() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing.token");
        let mut options = RuntimeOptions {
            join: vec!["argv-token".to_string()],
            join_files: vec![missing.clone()],
            ..RuntimeOptions::default()
        };

        let error = apply_join_token_sources(&mut options)
            .expect_err("an explicitly configured unreadable token file must fail startup");

        assert!(
            error.to_string().contains(&missing.display().to_string()),
            "{error:#}"
        );
    }
}

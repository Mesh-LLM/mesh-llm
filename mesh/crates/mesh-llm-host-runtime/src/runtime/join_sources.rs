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
//! When neither names a file, the documented default `invite.token` beside
//! the resolved config file is used. One deterministic filename, never a
//! directory scan: owner keys, membership, and genesis files share that
//! directory, so picking a file by shape would be guessing at operator intent.
//! Deriving it from the resolved config path keeps
//! `MESH_LLM_CONFIG=/path/project/config.toml` pointed at
//! `/path/project/invite.token`, so project-local rejoin state works the same
//! way the in-home default does.
//!
//! File-backed tokens are re-read for every rejoin attempt, so rotating a
//! token is "write the new token to the file" and nothing else: no unit edit,
//! no restart, and the token never appears in argv.
//!
//! That promise only holds while a file-derived token is never frozen as a
//! literal. `RuntimeOptions::join` therefore stays argv/`MESH_LLM_JOIN`
//! literals only, and every startup consumer resolves through this module
//! instead of reading `options.join`.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::RuntimeOptions;

/// Inline invite token; equivalent to a single `--join <TOKEN>`.
pub const MESH_LLM_JOIN_ENV: &str = "MESH_LLM_JOIN";
/// Path to a file holding the invite token; equivalent to `--join-file <PATH>`.
pub const MESH_LLM_JOIN_FILE_ENV: &str = "MESH_LLM_JOIN_FILE";
/// Default invite-token filename, resolved beside the config file.
pub const DEFAULT_JOIN_TOKEN_FILE_NAME: &str = "invite.token";

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

/// The documented default join-token file: `invite.token` beside the resolved
/// config file, or `None` when no config path can be resolved.
///
/// This goes through the same accessor the runtime loads its config through,
/// so `--config /path/project/config.toml` and
/// `MESH_LLM_CONFIG=/path/project/config.toml` both resolve the default to
/// `/path/project/invite.token`.
pub(crate) fn default_join_token_file(config_override: Option<&Path>) -> Option<PathBuf> {
    let config = mesh_llm_config::config_path(config_override).ok()?;
    Some(config.parent()?.join(DEFAULT_JOIN_TOKEN_FILE_NAME))
}

/// The default token file to consult, given the file sources already named.
///
/// Only when nothing explicit named a file: `--join-file` and
/// `MESH_LLM_JOIN_FILE` both win outright, and the default is never consulted
/// alongside them. The candidate must also already exist — the default is
/// opt-in by existence, so a machine that never created the file simply has no
/// default source. An existing-but-unreadable or empty file is reported by
/// `read_join_token_file`, not skipped: its presence means an operator meant
/// to provide a token.
fn default_join_token_file_candidate(
    config_override: Option<&Path>,
    join_files: &[PathBuf],
    environment_files: &[PathBuf],
) -> Option<PathBuf> {
    if !join_files.is_empty() || !environment_files.is_empty() {
        return None;
    }
    let path = default_join_token_file(config_override)?;
    path.exists().then_some(path)
}

/// Every invite token the runtime should try, with provenance and one message
/// per unusable source.
pub(crate) struct ResolvedJoinTokens {
    /// Usable tokens, highest priority first: argv, `MESH_LLM_JOIN`, then each
    /// file-backed source in the order it was named.
    pub(crate) tokens: Vec<String>,
    /// The subset of `tokens` that came from a file-backed source. These are
    /// re-read on every rejoin tick, so they must never be frozen as literals.
    pub(crate) file_tokens: Vec<String>,
    /// One message per unusable source.
    pub(crate) errors: Vec<String>,
}

/// Resolve every invite token the runtime should try, plus one message per
/// unusable file-backed source.
///
/// Priority is argv, then `MESH_LLM_JOIN`, then `--join-file`, then
/// `MESH_LLM_JOIN_FILE`; duplicates are dropped. Returning messages instead
/// of an error keeps both callers honest: startup escalates them to a hard
/// failure, while the 60s rejoin loop reports each one once and keeps
/// retrying, so a long-running service cannot be stranded silently by a token
/// file it cannot read. An inline `MESH_LLM_JOIN` that is set but blank is one
/// of those messages rather than "no source": a `MESH_LLM_JOIN=` line in a
/// service env file silently serving standalone is the same silent-failure
/// class the file-backed sources already reject at startup.
pub(crate) fn resolve_invite_token_sources(
    literals: &[String],
    inline_env: Option<&str>,
    join_files: &[PathBuf],
    environment_files: &[PathBuf],
) -> ResolvedJoinTokens {
    let mut tokens = Vec::new();
    for literal in literals {
        push_unique(&mut tokens, literal);
    }

    let mut errors = Vec::new();
    match inline_env {
        None => {}
        Some(inline) if inline.trim().is_empty() => errors.push(format!(
            "{MESH_LLM_JOIN_ENV} is set but empty; set it to the invite token or unset it"
        )),
        Some(inline) => push_unique(&mut tokens, inline),
    }

    let mut files: Vec<PathBuf> = join_files.to_vec();
    for path in environment_files {
        if !files.contains(path) {
            files.push(path.clone());
        }
    }

    let mut file_tokens = Vec::new();
    for path in files {
        match read_join_token_file(&path) {
            Ok(token) => {
                push_unique(&mut file_tokens, &token);
                push_unique(&mut tokens, &token);
            }
            Err(message) => errors.push(message),
        }
    }

    ResolvedJoinTokens {
        tokens,
        file_tokens,
        errors,
    }
}

/// `resolve_invite_token_sources` against the live process environment.
///
/// `config_override` is the runtime's resolved config path (`--config` /
/// `MESH_LLM_CONFIG`), used only to locate the default join-token file. The
/// default is consulted only when neither `--join-file` nor
/// `MESH_LLM_JOIN_FILE` named a file, and it is opt-in by existence: the file
/// exists because an operator created it, so its absence just means "no token
/// file" while an unreadable or empty one is a broken configuration that
/// `read_join_token_file` reports. Silently serving standalone because a
/// token file an operator created cannot be read is the exact failure mode
/// this module exists to remove.
pub(crate) fn resolve_invite_tokens(
    literals: &[String],
    join_files: &[PathBuf],
    config_override: Option<&Path>,
) -> ResolvedJoinTokens {
    let inline_env = std::env::var(MESH_LLM_JOIN_ENV).ok();
    let mut environment_files = environment_join_token_files();
    if let Some(default) =
        default_join_token_file_candidate(config_override, join_files, &environment_files)
    {
        environment_files.push(default);
    }
    resolve_invite_token_sources(
        literals,
        inline_env.as_deref(),
        join_files,
        &environment_files,
    )
}

/// The file-backed subset of [`RuntimeOptions::effective_join_tokens`].
///
/// A caller that has to decide whether a token it just joined with may be
/// recorded as a literal asks here: a token this returns came from a file that
/// the rejoin loop re-reads every tick, so freezing it would outlive a
/// rotation.
pub(crate) fn file_backed_join_tokens(options: &RuntimeOptions) -> Vec<String> {
    resolve_invite_tokens(
        &options.join,
        &options.join_files,
        options.config.as_deref(),
    )
    .file_tokens
}

/// Validate every configured invite-token source at startup, hard-failing on
/// the first unusable one.
///
/// Startup is the place to be strict: a serve that cannot read its configured
/// token must say so rather than quietly running standalone, and the error
/// lands in the service log where an operator will see it.
///
/// This validates only. `options.join` keeps argv/`MESH_LLM_JOIN` literals so
/// no file-derived token is frozen into the process; join attempts resolve
/// through `resolve_invite_tokens` (or
/// `RuntimeOptions::effective_join_tokens`) instead.
pub(crate) fn validate_join_token_sources(options: &RuntimeOptions) -> Result<()> {
    let resolved = resolve_invite_tokens(
        &options.join,
        &options.join_files,
        options.config.as_deref(),
    );
    if let Some(first) = resolved.errors.first() {
        bail!("{first}");
    }
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

        let resolved = resolve_invite_token_sources(
            &["argv-token".to_string(), "file-token".to_string()],
            Some("inline-token"),
            &[flag_file],
            &[environment_file],
        );

        assert!(resolved.errors.is_empty(), "{:?}", resolved.errors);
        assert_eq!(
            resolved.tokens,
            ["argv-token", "file-token", "inline-token", "env-file-token",]
        );
        assert_eq!(resolved.file_tokens, ["file-token", "env-file-token"]);
    }

    #[test]
    fn resolve_invite_token_sources_reads_a_shared_path_once() {
        let temp = tempfile::tempdir().unwrap();
        let shared = token_file(temp.path(), "shared.token", "shared-token\n");

        let resolved = resolve_invite_token_sources(
            &[],
            None,
            std::slice::from_ref(&shared),
            std::slice::from_ref(&shared),
        );

        assert!(resolved.errors.is_empty(), "{:?}", resolved.errors);
        assert_eq!(resolved.tokens, ["shared-token"]);
        assert_eq!(resolved.file_tokens, ["shared-token"]);
    }

    #[test]
    fn resolve_invite_token_sources_reports_every_unusable_file() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing.token");
        let empty = token_file(temp.path(), "empty.token", "");

        let resolved = resolve_invite_token_sources(
            &["argv-token".to_string()],
            None,
            &[missing.clone(), empty.clone()],
            &[],
        );

        assert_eq!(resolved.tokens, ["argv-token"]);
        assert_eq!(resolved.errors.len(), 2, "{:?}", resolved.errors);
        assert!(
            resolved.errors[0].contains(&missing.display().to_string()),
            "{:?}",
            resolved.errors
        );
        assert!(
            resolved.errors[1].contains(&empty.display().to_string()),
            "{:?}",
            resolved.errors
        );
    }

    #[test]
    fn resolve_invite_token_sources_rejects_a_set_but_blank_inline_env() {
        let resolved = resolve_invite_token_sources(&[], Some("   "), &[], &[]);

        assert!(
            resolved.tokens.is_empty(),
            "a blank MESH_LLM_JOIN carries no token: {:?}",
            resolved.tokens
        );
        assert_eq!(resolved.errors.len(), 1, "{:?}", resolved.errors);
        assert!(
            resolved.errors[0].contains(MESH_LLM_JOIN_ENV),
            "{:?}",
            resolved.errors
        );
    }

    #[test]
    fn resolve_invite_token_sources_treats_an_unset_inline_env_as_no_source() {
        let resolved = resolve_invite_token_sources(&["argv-token".to_string()], None, &[], &[]);

        assert_eq!(resolved.tokens, ["argv-token"]);
        assert!(resolved.errors.is_empty(), "{:?}", resolved.errors);
    }

    #[test]
    fn default_join_token_file_sits_beside_the_resolved_config() {
        let config = PathBuf::from("/path/project/config.toml");

        assert_eq!(
            default_join_token_file(Some(&config)),
            Some(PathBuf::from("/path/project").join(DEFAULT_JOIN_TOKEN_FILE_NAME))
        );
    }

    #[test]
    fn resolve_invite_tokens_uses_the_default_file_beside_the_config() {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config.toml");
        token_file(temp.path(), DEFAULT_JOIN_TOKEN_FILE_NAME, "default-token\n");

        let resolved = resolve_invite_tokens(&[], &[], Some(&config));

        assert!(resolved.errors.is_empty(), "{:?}", resolved.errors);
        assert_eq!(resolved.tokens, ["default-token"]);
        assert_eq!(
            resolved.file_tokens,
            ["default-token"],
            "the default file is file-backed, so its token must stay rotatable"
        );
    }

    #[test]
    fn resolve_invite_tokens_ignores_an_absent_default_file() {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config.toml");

        let resolved = resolve_invite_tokens(&[], &[], Some(&config));

        assert!(
            resolved.tokens.is_empty(),
            "an absent default file is not a source: {:?}",
            resolved.tokens
        );
        assert!(
            resolved.errors.is_empty(),
            "an absent default file is not an error: {:?}",
            resolved.errors
        );
    }

    #[test]
    fn resolve_invite_tokens_reports_a_present_but_blank_default_file() {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config.toml");
        token_file(temp.path(), DEFAULT_JOIN_TOKEN_FILE_NAME, "   \n");

        let resolved = resolve_invite_tokens(&[], &[], Some(&config));

        assert!(
            resolved.tokens.is_empty(),
            "a blank default file carries no token: {:?}",
            resolved.tokens
        );
        assert_eq!(resolved.errors.len(), 1, "{:?}", resolved.errors);
        assert!(
            resolved.errors[0].contains("is empty"),
            "{:?}",
            resolved.errors
        );
    }

    #[test]
    fn resolve_invite_tokens_prefers_an_explicit_file_over_the_default() {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config.toml");
        let explicit = token_file(temp.path(), "explicit.token", "explicit-token\n");
        token_file(temp.path(), DEFAULT_JOIN_TOKEN_FILE_NAME, "default-token\n");

        let resolved = resolve_invite_tokens(&[], std::slice::from_ref(&explicit), Some(&config));

        assert!(resolved.errors.is_empty(), "{:?}", resolved.errors);
        assert_eq!(resolved.tokens, ["explicit-token"]);
    }

    #[test]
    fn validate_join_token_sources_leaves_options_join_as_literals() {
        let temp = tempfile::tempdir().unwrap();
        let path = token_file(temp.path(), "invite.token", "signed-token\n");
        let options = RuntimeOptions {
            join_files: vec![path],
            ..RuntimeOptions::default()
        };

        validate_join_token_sources(&options).expect("token file should resolve");

        assert!(
            options.join.is_empty(),
            "a file-derived token must never be folded into options.join: {:?}",
            options.join
        );
    }

    #[test]
    fn validate_join_token_sources_fails_fast_on_an_unreadable_file() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing.token");
        let options = RuntimeOptions {
            join: vec!["argv-token".to_string()],
            join_files: vec![missing.clone()],
            ..RuntimeOptions::default()
        };

        let error = validate_join_token_sources(&options)
            .expect_err("an explicitly configured unreadable token file must fail startup");

        assert!(
            error.to_string().contains(&missing.display().to_string()),
            "{error:#}"
        );
        assert_eq!(
            options.join,
            ["argv-token"],
            "validation must not rewrite the literal token set"
        );
    }
}

use std::ffi::OsString;

/// Placeholder shown where a credential appeared in the raw invocation.
/// Metacharacter-free on purpose: the suggestion is meant to be pasted
/// into a shell, where `<...>` would be parsed as redirection.
const REDACTED: &str = "REDACTED";

const RELAY_AUTH_FLAG: &str = "--relay-auth";
const JOIN_FLAG: &str = "--join";
const JOIN_SHORT: &str = "-j";

/// Replace credential-bearing values in raw argv before they are echoed in
/// the legacy-surface warning's suggested command.
///
/// `--relay-auth` carries `URL=TOKEN`: the URL is the relay's public
/// identity and survives, the token does not. `--join` carries an invite
/// token outright, so the whole value goes. Every spelling clap accepts for
/// these two flags is covered: the space-separated and `=`-fused long forms,
/// plus `-j` and its attached `-jTOKEN` form. `--join-file` is a path, not a
/// credential, and passes through.
///
/// The function works on raw argv rather than the parsed `Cli` so the
/// suggestion keeps the caller's own spelling instead of a canonicalized
/// rewrite. Malformed `--relay-auth` values with no `=` separator are
/// redacted whole; guessing which half is safe is how tokens leak. A
/// non-UTF-8 value following a credential flag is redacted whole too,
/// while a standalone non-UTF-8 argument passes through: the only values
/// that can reach the warning with non-UTF-8 bytes on a successful parse
/// are PathBuf-typed (`--model`, `--config`), which are paths, not
/// credentials.
#[must_use]
pub(super) fn redact_suggestion_args(args: &[OsString]) -> Vec<OsString> {
    #[derive(Clone, Copy)]
    enum Secret {
        RelayAuth,
        Join,
    }

    let mut redacted = Vec::with_capacity(args.len());
    let mut expected_secret: Option<Secret> = None;

    for (index, arg) in args.iter().enumerate() {
        // argv[0] is the program name, never a flag or a value.
        if index == 0 {
            redacted.push(arg.clone());
            continue;
        }

        let secret = expected_secret.take();
        if let Some(Secret::RelayAuth) = secret {
            redacted.push(redact_relay_auth_value(arg));
            continue;
        }
        if let Some(Secret::Join) = secret {
            redacted.push(OsString::from(REDACTED));
            continue;
        }

        let Some(text) = arg.to_str() else {
            // Passes through, lossily rendered downstream: PathBuf-typed
            // values (`--model`, `--config`) legitimately reach the
            // warning with non-UTF-8 bytes, and the fused credential
            // spellings cannot (clap rejects non-UTF-8 values for their
            // String value parsers before a warning is built). A
            // non-UTF-8 VALUE after a credential flag is still redacted
            // by the branches above.
            redacted.push(arg.clone());
            continue;
        };

        if text == RELAY_AUTH_FLAG {
            expected_secret = Some(Secret::RelayAuth);
            redacted.push(arg.clone());
        } else if let Some(value) = text.strip_prefix(&format!("{RELAY_AUTH_FLAG}=")) {
            redacted.push(OsString::from(format!(
                "{RELAY_AUTH_FLAG}={}",
                redact_relay_auth_token(value)
            )));
        } else if text == JOIN_FLAG || text == JOIN_SHORT {
            expected_secret = Some(Secret::Join);
            redacted.push(arg.clone());
        } else if let Some(_value) = text.strip_prefix(&format!("{JOIN_FLAG}=")) {
            redacted.push(OsString::from(format!("{JOIN_FLAG}={REDACTED}")));
        } else if text.len() > JOIN_SHORT.len() && text.starts_with(JOIN_SHORT) {
            // `-j` is the only short flag on the root surface, so anything
            // single-dashed starting with it is `-j` with an attached value.
            redacted.push(OsString::from(format!("{JOIN_SHORT}{REDACTED}")));
        } else {
            redacted.push(arg.clone());
        }
    }

    redacted
}

/// Redact the token half of a `URL=TOKEN` value, keeping the public URL.
fn redact_relay_auth_value(value: &OsString) -> OsString {
    match value.to_str() {
        Some(text) => OsString::from(redact_relay_auth_token(text)),
        None => OsString::from(REDACTED),
    }
}

fn redact_relay_auth_token(value: &str) -> String {
    match value.split_once('=') {
        // Splits on the first `=` only, matching the parser: tokens may
        // themselves contain `=` (base64 padding, JWTs).
        Some((url, _token)) => format!("{url}={REDACTED}"),
        None => REDACTED.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Cli;
    use crate::parser::normalization::value_taking_option_names;

    fn redact(raw: &[&str]) -> Vec<String> {
        redact_suggestion_args(&raw.iter().map(OsString::from).collect::<Vec<_>>())
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn relay_auth_space_form_keeps_url_and_drops_token() {
        assert_eq!(
            redact(&[
                "mesh-llm",
                "--auto",
                "--relay-auth",
                "https://gated.example/=token"
            ]),
            vec![
                "mesh-llm",
                "--auto",
                "--relay-auth",
                "https://gated.example/=REDACTED"
            ]
        );
    }

    #[test]
    fn relay_auth_equals_form_keeps_url_and_drops_token() {
        assert_eq!(
            redact(&[
                "mesh-llm",
                "--relay-auth=https://gated.example/=token",
                "--auto"
            ]),
            vec![
                "mesh-llm",
                "--relay-auth=https://gated.example/=REDACTED",
                "--auto"
            ]
        );
    }

    #[test]
    fn relay_auth_token_containing_equals_is_fully_redacted() {
        // NIP-98-style base64 padding: only the first `=` separates URL from
        // token, so everything after it must disappear.
        assert_eq!(
            redact(&[
                "mesh-llm",
                "--relay-auth",
                "https://gated.example/=eyJhbGciOiJFZERTQSJ9.payload=="
            ]),
            vec![
                "mesh-llm",
                "--relay-auth",
                "https://gated.example/=REDACTED"
            ]
        );
    }

    #[test]
    fn relay_auth_value_without_separator_is_redacted_whole() {
        // Unreachable through a successful parse (the value parser rejects
        // it), but this pass also guards future callers: never guess which
        // half of a malformed credential is safe to echo.
        assert_eq!(
            redact(&["mesh-llm", "--relay-auth", "tokensecret"]),
            vec!["mesh-llm", "--relay-auth", "REDACTED"]
        );
    }

    #[test]
    fn join_values_are_redacted_in_every_spelling() {
        assert_eq!(
            redact(&["mesh-llm", "--join", "token123", "--auto"]),
            vec!["mesh-llm", "--join", "REDACTED", "--auto"]
        );
        assert_eq!(
            redact(&["mesh-llm", "--join=token123"]),
            vec!["mesh-llm", "--join=REDACTED"]
        );
        assert_eq!(
            redact(&["mesh-llm", "-j", "token123"]),
            vec!["mesh-llm", "-j", "REDACTED"]
        );
        assert_eq!(
            redact(&["mesh-llm", "-jtoken123"]),
            vec!["mesh-llm", "-jREDACTED"]
        );
    }

    #[test]
    fn repeated_credential_flags_are_all_redacted() {
        assert_eq!(
            redact(&[
                "mesh-llm",
                "--relay-auth",
                "https://a.example/=one",
                "--join",
                "token123",
                "--relay-auth=https://b.example/=two",
                "--join=token456"
            ]),
            vec![
                "mesh-llm",
                "--relay-auth",
                "https://a.example/=REDACTED",
                "--join",
                "REDACTED",
                "--relay-auth=https://b.example/=REDACTED",
                "--join=REDACTED"
            ]
        );
    }

    #[test]
    fn non_credential_args_pass_through_unchanged() {
        let raw = [
            "mesh-llm",
            "--auto",
            "--model",
            "Qwen3-8B-Q4_K_M",
            "--relay",
            "https://gated.example/",
            "--join-file",
            "/tmp/token.txt",
            "--discover=MyMesh",
        ];
        assert_eq!(redact(&raw), raw);
    }

    #[test]
    fn dangling_credential_flag_at_argv_end_is_kept() {
        // clap rejects a missing value, so this cannot reach the warning in
        // practice; the pass must still terminate without touching anything.
        assert_eq!(
            redact(&["mesh-llm", "--auto", "--relay-auth"]),
            vec!["mesh-llm", "--auto", "--relay-auth"]
        );
    }

    /// Staleness guard: the redaction spellings are hand-maintained, so
    /// derive the COMPLETE spelling set of both credential flags from clap
    /// (primary long, every alias, every short) and assert it is exactly
    /// the set this pass handles. Membership checks are not enough: adding
    /// an alias would keep them green while the new spelling leaks. A
    /// rename, a dropped short, or a new alias all turn this red and point
    /// here instead of silently leaving a spelling unredacted.
    #[test]
    fn credential_flag_spellings_are_exactly_the_redacted_set() {
        fn spellings_for(primary: &str) -> Vec<String> {
            use clap::CommandFactory;

            let command = Cli::command();
            let argument = command
                .get_arguments()
                .find(|argument| argument.get_long() == Some(primary))
                .unwrap_or_else(|| panic!("clap no longer declares --{primary}"));
            let longs = argument
                .get_long()
                .into_iter()
                .chain(argument.get_all_aliases().into_iter().flatten())
                .map(|long| format!("--{long}"));
            let shorts = argument
                .get_short()
                .into_iter()
                .chain(argument.get_all_short_aliases().into_iter().flatten())
                .map(|short| format!("-{short}"));
            longs.chain(shorts).collect()
        }

        let mut relay_auth = spellings_for("relay-auth");
        let mut join = spellings_for("join");
        relay_auth.sort();
        join.sort();

        assert_eq!(
            relay_auth,
            vec![RELAY_AUTH_FLAG],
            "clap accepts spellings of --relay-auth the pass does not redact"
        );
        assert_eq!(
            join,
            vec![JOIN_FLAG, JOIN_SHORT],
            "clap accepts spellings of --join the pass does not redact"
        );

        // And both flags are still value-taking, so the space-separated
        // form is live alongside the fused form.
        let options = value_taking_option_names();
        for option in [RELAY_AUTH_FLAG, JOIN_FLAG, JOIN_SHORT] {
            assert!(
                options.contains(&option.to_string()),
                "{option} no longer takes a value; re-check the redaction pass"
            );
        }
    }

    /// Standalone non-UTF-8 arguments pass through (rendered lossily by
    /// the shell display), because PathBuf-typed values like `--model`
    /// legitimately reach the warning with non-UTF-8 bytes on a successful
    /// parse. The fused credential spellings (`--relay-auth=<bytes>`)
    /// cannot reach it at all: clap rejects non-UTF-8 values for their
    /// String value parsers before any warning is built. A non-UTF-8
    /// value FOLLOWING a credential flag is redacted whole by the
    /// value-position branches above.
    #[cfg(unix)]
    #[test]
    fn non_utf8_arguments_pass_through_except_credential_values() {
        use std::os::unix::ffi::OsStringExt;

        let args = vec![
            OsString::from("mesh-llm"),
            OsString::from("--model"),
            OsString::from_vec(b"/tmp/model\xff.gguf".to_vec()),
            OsString::from("--relay-auth"),
            OsString::from_vec(b"\xff\xfe".to_vec()),
        ];
        let redacted = redact_suggestion_args(&args);

        // The model path survives lossily; the credential value does not.
        assert_eq!(redacted[2].to_string_lossy(), "/tmp/model\u{FFFD}.gguf");
        assert_eq!(redacted[4].to_string_lossy(), "REDACTED");
    }

    /// The `starts_with("-j")` branch relies on `-j` being the ONLY short
    /// flag on the root surface: with any second short, a cluster like
    /// `-vj TOKEN` parses as that flag plus `join=TOKEN`, a spelling the
    /// pass does not recognize. Derived so the first new root short turns
    /// this red instead of silently reopening the leak.
    #[test]
    fn dash_j_is_the_only_short_flag_on_the_root_surface() {
        use clap::CommandFactory;

        let mut shorts: Vec<String> = Cli::command()
            .get_arguments()
            .flat_map(|argument| {
                argument
                    .get_short()
                    .into_iter()
                    .chain(argument.get_all_short_aliases().into_iter().flatten())
                    .map(|short| format!("-{short}"))
            })
            .collect();
        shorts.sort();

        assert_eq!(
            shorts,
            vec![JOIN_SHORT],
            "a new root short flag changes how short clusters parse; re-check the redaction pass"
        );
    }

    /// Nothing in a clap declaration marks a value as a credential, so the
    /// pass's flag list is maintained by hand. This heuristic guard fails
    /// when a NEW root flag appears whose name says it carries one, so the
    /// next secret-bearing flag cannot be echoed by the suggestion just
    /// because nobody remembered this module. Root-level only: with a
    /// subcommand present the warning never fires, so subcommand flags
    /// cannot leak through this path.
    #[test]
    fn credential_named_root_flags_are_all_redacted() {
        use clap::CommandFactory;

        const KEYWORDS: [&str; 6] = ["auth", "token", "secret", "pass", "key", "credential"];

        let mut credential_named: Vec<String> = Cli::command()
            .get_arguments()
            .flat_map(|argument| {
                argument
                    .get_long()
                    .into_iter()
                    .chain(argument.get_all_aliases().into_iter().flatten())
                    .map(|long| format!("--{long}"))
            })
            .filter(|name| {
                let lower = name.to_lowercase();
                KEYWORDS.iter().any(|keyword| lower.contains(keyword))
            })
            .collect();
        credential_named.sort();

        // Allow list, with the reason each name is safe to echo:
        // - `--relay-auth` is the one real hit: this module redacts it.
        // - `--release-signer-key` carries allowed PUBLIC signer keys for
        //   release-attestation policy; verification keys are not secrets.
        // - `--owner-key` is a path to a keystore file, not key material.
        // - The `--speculative-*-tokens` family counts LLM tokens, not
        //   credentials.
        let mut expected: Vec<String> = [
            "--owner-key",
            "--relay-auth",
            "--release-signer-key",
            "--speculative-extension-max-tokens",
            "--speculative-native-mtp-reject-cooldown-tokens",
            "--speculative-ngram-max-proposal-tokens",
            "--speculative-verify-window-max-tokens",
            "--speculative-verify-window-min-tokens",
            "--speculative-verify-window-runahead-tokens",
        ]
        .iter()
        .map(|flag| (*flag).to_string())
        .collect();
        expected.sort();

        assert_eq!(
            credential_named, expected,
            "a new credential-named root flag appeared; redact it in this module, or allow it here with a justification if it is safe to echo"
        );
    }
}

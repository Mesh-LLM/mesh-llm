use serde_json::{Map, Value};

const MAX_PRESENTATION_CHARS: usize = 1024;

pub(super) fn safe_native_params(params: &[(String, Value)]) -> Map<String, Value> {
    params
        .iter()
        .filter(|(key, _)| safe_param_key(key))
        .filter_map(|(key, value)| match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => Some((key.clone(), value.clone())),
            Value::String(value) => Some((key.clone(), Value::String(sanitize_text(value)))),
            Value::Array(_) | Value::Object(_) => None,
        })
        .collect()
}

pub(super) fn sanitize_map(mut fields: Map<String, Value>) -> Map<String, Value> {
    fields.retain(|key, value| {
        safe_param_key(key)
            && match value {
                Value::Null | Value::Bool(_) | Value::Number(_) => true,
                Value::String(text) => {
                    *text = sanitize_text(text);
                    true
                }
                Value::Array(_) | Value::Object(_) => false,
            }
    });
    fields
}

pub(super) fn sanitize_text(input: &str) -> String {
    let redact_paths = !interactive_output();
    let masked = if redact_paths {
        mask_operator_home(input, operator_home().as_deref())
    } else {
        input.to_owned()
    };
    let sanitized = redact_sensitive_fragments(&redact_url_query_values(&masked), redact_paths);
    if sanitized.chars().count() <= MAX_PRESENTATION_CHARS {
        sanitized
    } else {
        format!(
            "{}... [TRUNCATED]",
            sanitized
                .chars()
                .take(MAX_PRESENTATION_CHARS)
                .collect::<String>()
        )
    }
}

/// Is the process's diagnostics stream attached to a terminal?
///
/// Interactive sessions get actionable errors: the operator's own
/// command output already prints local paths verbatim (cache
/// locations, config paths, keystores), so redacting paths only in
/// event text turns guidance — a `--config` path the operator just
/// typed — into `[REDACTED_PATH]`. Unattended contexts keep the
/// redaction: when stderr is redirected rather than attached, the
/// reader is no longer the operator at the keyboard but launchd's
/// `stderr.log`, the systemd journal, container logs, or CI capture
/// — collected, archived, and attached to bug reports, where the
/// operator's home-directory layout should not land.
///
/// The interactive dashboard is the attached case with fd 2 itself
/// redirected: while it owns the screen, `console_capture` points
/// stderr at its capture pipe, so the raw probe alone would read
/// "unattended" for the operator watching the Events panel. An
/// installed capture means a dashboard is live on the controlling
/// terminal, which counts as attached.
fn interactive_output() -> bool {
    #[cfg(test)]
    if let Some(forced) = TEST_ATTACHMENT_OVERRIDE.with(std::cell::Cell::get) {
        return forced;
    }
    std::io::IsTerminal::is_terminal(&std::io::stderr())
        || super::super::console_capture::capture_installed()
}

#[cfg(test)]
thread_local! {
    /// Test-only attachment override: `Some(attached)` while a test
    /// forces the probe, `None` when unforced (the real stderr probe
    /// runs). Thread-local by design — tests run on parallel threads,
    /// and a shared flag would let one test's teardown un-force
    /// another test's override mid-flight.
    static TEST_ATTACHMENT_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(in crate::output) mod test_support {
    /// Force the attachment probe for the duration of a test body.
    /// Restores the real probe on drop.
    pub(in crate::output) struct AttachmentOverride;

    impl AttachmentOverride {
        pub(in crate::output) fn set(attached: bool) -> Self {
            super::TEST_ATTACHMENT_OVERRIDE.with(|cell| cell.set(Some(attached)));
            Self
        }
    }

    impl Drop for AttachmentOverride {
        fn drop(&mut self) {
            super::TEST_ATTACHMENT_OVERRIDE.with(|cell| cell.set(None));
        }
    }
}

/// Keep the useful error category and surrounding diagnostic text while
/// replacing only credential-shaped values on an attached terminal, and
/// additionally redacting absolute paths when the diagnostics stream is
/// redirected (see [`interactive_output`]). Durable artifacts keep the
/// host logging policy's unconditional path redaction; this terminal
/// layer applies its own for unattended readers — every absolute-path
/// word becomes `[REDACTED_PATH]`, not the host policy's `~` home
/// masking that preserves the path tail.
fn redact_sensitive_fragments(input: &str, redact_paths: bool) -> String {
    let mut redact_next = false;
    let mut sensitive_key_pending = false;
    let mut previous = String::new();
    let mut output = Vec::new();

    for word in input.split_whitespace() {
        let trimmed = trim_presentation_punctuation(word);
        let normalized = trimmed.to_ascii_lowercase();

        if redact_next {
            sensitive_key_pending = false;
            if matches!(normalized.as_str(), "bearer" | "basic") {
                output.push(word.to_owned());
            } else {
                redact_next = false;
                output.push(redact_word(word, "[REDACTED]"));
            }
        } else if sensitive_key_pending && matches!(trimmed, ":" | "=") {
            sensitive_key_pending = false;
            redact_next = true;
            output.push(word.to_owned());
        } else {
            sensitive_key_pending = false;
            if matches!(normalized.as_str(), "bearer" | "basic") {
                redact_next = true;
                output.push(word.to_owned());
            } else if redact_paths && is_private_absolute_path(trimmed) {
                output.push(redact_word(word, "[REDACTED_PATH]"));
            } else if is_credential_value(trimmed) {
                output.push(redact_word(word, "[REDACTED]"));
            } else if !trimmed.contains(['?', '&'])
                && let Some((key, separator, value)) = split_key_value(trimmed)
                && is_sensitive_key(key)
                && !is_invite_token_context(&previous, key)
            {
                if value.is_empty()
                    || matches!(value.to_ascii_lowercase().as_str(), "bearer" | "basic")
                {
                    redact_next = true;
                    output.push(word.to_owned());
                } else if value == "[REDACTED]" {
                    output.push(word.to_owned());
                } else {
                    output.push(format!("{key}{separator}[REDACTED]"));
                }
            } else {
                sensitive_key_pending =
                    is_sensitive_key(trimmed) && !is_invite_token_context(&previous, trimmed);
                output.push(word.to_owned());
            }
        }
        previous = normalized;
    }

    output.join(" ")
}

fn trim_presentation_punctuation(word: &str) -> &str {
    word.trim_matches(|ch: char| {
        matches!(
            ch,
            '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
        )
    })
}

fn redact_word(word: &str, replacement: &str) -> String {
    let trimmed = trim_presentation_punctuation(word);
    let start = word.find(trimmed).unwrap_or(0);
    let end = start + trimmed.len();
    format!("{}{replacement}{}", &word[..start], &word[end..])
}

fn split_key_value(value: &str) -> Option<(&str, char, &str)> {
    let index = value.find(['=', ':'])?;
    let separator = value[index..].chars().next()?;
    Some((
        &value[..index],
        separator,
        &value[index + separator.len_utf8()..],
    ))
}

pub(super) fn native_category(category: &str) -> &'static str {
    match category {
        "backend" => "backend",
        "model" => "model",
        "memory" => "memory",
        "kv_cache" => "kv_cache",
        "tokenizer" => "tokenizer",
        _ => "runtime",
    }
}

pub(super) fn json_scalar(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => "[OMITTED]".to_string(),
    }
}

fn safe_param_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    key.len() <= 64
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        && !matches!(
            normalized.as_str(),
            "timestamp"
                | "level"
                | "event"
                | "message"
                | "schema_version"
                | "event_id"
                | "request_id"
                | "attempt_id"
                | "channel"
                | "sequence"
                | "outcome"
        )
        && !is_sensitive_key(key)
        && !normalized.contains("path")
        && !normalized.contains("url")
        && !normalized.contains("id")
}

fn is_private_absolute_path(word: &str) -> bool {
    // A bare `/` is the separator in text such as `runtime 0.76.1 / ABI 0.1.71`;
    // revealing the filesystem root is no leak.
    (word.starts_with('/') && word != "/")
        || word.to_ascii_lowercase().starts_with("file://")
        || is_windows_absolute_path(word)
        || word.contains(HOME_SENTINEL)
}

/// Stands in for the operator's home directory from before the text is split
/// into words until the word that held it is redacted whole, so a home with a
/// space (`C:\Users\Jane Doe`) cannot fall across two words. A private-use
/// character: log text does not carry it, and punctuation trimming keeps it.
const HOME_SENTINEL: &str = "\u{E000}";

fn mask_operator_home(input: &str, home: Option<&str>) -> String {
    let Some(home) = home.filter(|home| !home.is_empty()) else {
        return input.to_owned();
    };
    // Windows paths compare regardless of case. ASCII folding keeps every
    // byte offset, so matches found in the folded text map back to `input`.
    let (text, needle) = if cfg!(windows) {
        (input.to_ascii_lowercase(), home.to_ascii_lowercase())
    } else {
        (input.to_owned(), home.to_owned())
    };
    let mut masked = String::with_capacity(input.len());
    let mut copied = 0;
    for (start, _) in text.match_indices(&needle) {
        masked.push_str(&input[copied..start]);
        masked.push_str(HOME_SENTINEL);
        copied = start + needle.len();
    }
    masked.push_str(&input[copied..]);
    masked
}

/// The operator's home directory: `HOME`, or `USERPROFILE` on Windows, where
/// `HOME` is usually unset. Only an absolute path counts, so a Git Bash
/// `HOME` such as `/c/Users/name` falls through to `USERPROFILE`.
fn operator_home() -> Option<String> {
    ["HOME", "USERPROFILE"].into_iter().find_map(|variable| {
        let home = std::env::var(variable).ok()?;
        (!home.is_empty() && std::path::Path::new(&home).is_absolute()).then_some(home)
    })
}

fn is_windows_absolute_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    let drive_root = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/');
    let unc_or_device =
        value.starts_with("\\\\") || value.starts_with("//") || value.starts_with("\\??\\");
    drive_root || unc_or_device
}

/// Apply the host logging policy's query-value rule before projecting text.
/// Endpoint and non-sensitive query metadata remain useful to an operator,
/// but credentials may not survive in any TUI surface.
fn redact_url_query_values(input: &str) -> String {
    input
        .split_whitespace()
        .map(redact_query_in_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact_query_in_token(token: &str) -> String {
    let token = redact_url_userinfo(token);
    let Some((base, query_and_fragment)) = token.split_once('?') else {
        return token;
    };
    let (query, fragment) = query_and_fragment
        .split_once('#')
        .map_or((query_and_fragment, ""), |(query, fragment)| {
            (query, fragment)
        });
    let query = query
        .split('&')
        .filter(|parameter| !parameter.is_empty())
        .map(redact_query_parameter)
        .collect::<Vec<_>>()
        .join("&");
    let fragment_suffix = if fragment.is_empty() {
        String::new()
    } else {
        format!("#{fragment}")
    };

    if query.is_empty() {
        format!("{base}{fragment_suffix}")
    } else {
        format!("{base}?{query}{fragment_suffix}")
    }
}

fn redact_url_userinfo(token: &str) -> String {
    let Some(scheme_end) = token.find("://") else {
        return token.to_owned();
    };
    let authority_start = scheme_end + 3;
    let authority_end = token[authority_start..]
        .find(['/', '?', '#'])
        .map_or(token.len(), |offset| authority_start + offset);
    let Some(at_offset) = token[authority_start..authority_end].rfind('@') else {
        return token.to_owned();
    };
    let at = authority_start + at_offset;
    if at == authority_start {
        return token.to_owned();
    }
    format!("{}[REDACTED]{}", &token[..authority_start], &token[at..])
}

fn redact_query_parameter(parameter: &str) -> String {
    let Some((key, value)) = parameter.split_once('=') else {
        return if is_sensitive_key(parameter) {
            format!("{parameter}=[REDACTED]")
        } else {
            parameter.to_string()
        };
    };

    if is_sensitive_key(key) || is_credential_value(value) {
        format!("{key}=[REDACTED]")
    } else {
        parameter.to_string()
    }
}

/// Match the host policy's credential categories by their semantic key shape.
/// This is deliberately broader than a presentation-only marker list, so new
/// token and credential parameter spellings fail closed in TUI projections.
fn is_sensitive_key(key: &str) -> bool {
    let key = key
        .trim_matches(|ch: char| matches!(ch, '"' | '\'' | ':' | ','))
        .to_ascii_lowercase();
    (!is_invite_token_key(&key) && key.contains("token"))
        || key.contains("password")
        || key.contains("secret")
        || key == "auth"
        || key == "authorization"
        || key.starts_with("auth_")
        || key == "bearer"
        || key == "key"
        || key.ends_with("key")
        || key == "session_id"
        || matches!(key.as_str(), "prompt" | "completion" | "response" | "query")
}

fn is_invite_token_key(key: &str) -> bool {
    matches!(key, "invite_token" | "invite-token" | "invitetoken")
}

fn is_invite_token_context(previous: &str, key: &str) -> bool {
    is_invite_token_key(&key.to_ascii_lowercase())
        || (previous.trim_matches(|ch: char| !ch.is_ascii_alphanumeric()) == "invite"
            && key
                .trim_matches(|ch: char| !ch.is_ascii_alphanumeric())
                .eq_ignore_ascii_case("token"))
}

fn is_credential_value(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    value.starts_with("bearer ")
        || value.starts_with("basic ")
        || value.starts_with("sk_")
        || value.starts_with("sk-")
        || value.starts_with("ghp_")
}

#[cfg(test)]
mod tests {
    use super::test_support::AttachmentOverride;
    use super::{safe_native_params, sanitize_text};
    use serde_json::json;

    #[test]
    fn preserves_local_paths_including_homes_with_spaces_when_attached() {
        let _attached = AttachmentOverride::set(true);
        // Paths render on the operator's own terminal, where command output
        // already prints them verbatim; redacting them only in event text
        // made failure guidance unactionable (see the T-002 regression pins
        // below).
        let home = r"C:\Users\Jane Doe";
        for input in [
            format!(r"path={home}\models\model.gguf"),
            format!(r"failed to open {home}\cache\index.json"),
        ] {
            assert_eq!(sanitize_text(&input), input);
        }
    }

    #[test]
    fn keeps_paths_after_benign_keys_while_redacting_after_sensitive_keys() {
        // Attached-mode assertion: pin the probe so a build account whose
        // home literally is /home or /home/operator cannot flip the
        // expected mode under the harness's captured stderr.
        let _attached = AttachmentOverride::set(true);
        let sanitized = sanitize_text("path=/home/operator/model.gguf token=supersecret");

        assert!(
            sanitized.contains("path=/home/operator/model.gguf"),
            "{sanitized:?}"
        );
        assert!(sanitized.contains("token=[REDACTED]"), "{sanitized:?}");
        assert!(!sanitized.contains("supersecret"), "{sanitized:?}");
    }

    #[test]
    fn preserves_the_operators_own_paths_in_cli_guidance_and_errors_when_attached() {
        let _attached = AttachmentOverride::set(true);
        // T-002 regression pins from the 2026-11-17 behavioral campaign:
        // every one of these message shapes came back from the terminal
        // projection with `[REDACTED_PATH]` where the operator's own
        // argument or path belonged.
        let cases = [
            "top-level serving flags now map to `mesh-llm serve`. Please use: mesh-llm serve --model /models/qwen.gguf --console 3131",
            "Invalid config /home/operator/.mesh-llm/config.toml failed to parse config TOML",
            "reading owner-control credential from /home/operator/.mesh-llm/owner-control-token-v2",
            "mesh-llm update only works for release-bundle installs. Current executable: /usr/local/bin/mesh-llm",
            "Eager startup model '/models/qwen.gguf' failed to resolve after daemon surfaces became ready",
        ];

        for input in cases {
            assert_eq!(sanitize_text(input), input);
        }
    }

    #[test]
    fn redacts_sensitive_url_query_values_without_hiding_safe_metadata() {
        let sanitized =
            sanitize_text("upstream provider.example/v1?token=supersecret&format=json&page=1");

        assert!(!sanitized.contains("supersecret"));
        assert!(sanitized.contains("token=[REDACTED]"));
        assert!(sanitized.contains("format=json"));
        assert!(sanitized.contains("page=1"));
    }

    #[test]
    fn redacts_spaced_credentials_and_preserves_surrounding_punctuation() {
        let cases = [
            ("password: secret-after-colon", "password: [REDACTED]"),
            ("token = secret-after-equals", "token = [REDACTED]"),
            ("api_key= 'secret-after-empty',", "api_key= '[REDACTED]',"),
            ("secret : (secret-in-parens);", "secret : ([REDACTED]);"),
            (
                "authorization Bearer bearer-value,",
                "authorization Bearer [REDACTED],",
            ),
            (
                "authorization Basic basic-value;",
                "authorization Basic [REDACTED];",
            ),
            (
                "Authorization:Bearer compact-value",
                "Authorization:Bearer [REDACTED]",
            ),
            (
                "Authorization = Basic compact-basic",
                "Authorization = Basic [REDACTED]",
            ),
            (
                "Authorization: Bearer spaced-value",
                "Authorization: Bearer [REDACTED]",
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(sanitize_text(input), expected);
        }
    }

    #[test]
    fn redacts_url_userinfo_and_query_credentials_but_keeps_diagnostic_context() {
        let sanitized = sanitize_text(
            "upstream https://operator:p@ssword@provider.example/v1/chat?api_key=query-secret&model=qwen#attempt-2 failed",
        );

        assert_eq!(
            sanitized,
            "upstream https://[REDACTED]@provider.example/v1/chat?api_key=[REDACTED]&model=qwen#attempt-2 failed"
        );
        assert!(!sanitized.contains("operator"));
        assert!(!sanitized.contains("p@ssword"));
        assert!(!sanitized.contains("query-secret"));
    }

    #[test]
    fn preserves_generated_mesh_invite_tokens_in_fields_messages_and_urls() {
        let invite = "eyJpZCI6ImdlbmVyYXRlZC1tZXNoLWludml0ZSJ9";

        assert_eq!(
            sanitize_text(&format!("Invite token: {invite}")),
            format!("Invite token: {invite}")
        );
        assert_eq!(
            sanitize_text(&format!("invite_token={invite}")),
            format!("invite_token={invite}")
        );
        assert_eq!(
            sanitize_text(&format!("mesh://join?invite_token={invite}&role=client")),
            format!("mesh://join?invite_token={invite}&role=client")
        );

        let params = vec![("invite_token".to_string(), json!(invite))];
        assert_eq!(
            safe_native_params(&params).get("invite_token"),
            Some(&json!(invite))
        );
    }

    #[test]
    fn preserves_unix_windows_unc_and_device_paths_when_attached() {
        let _attached = AttachmentOverride::set(true);
        let cases = [
            "/Users/operator/.mesh-llm/config.toml",
            r"C:\Users\operator\.mesh-llm\config.toml",
            r"\\server\private\config.toml",
            r"\??\C:\private\config.toml",
        ];

        for path in cases {
            let input = format!("startup failed at {path}");
            assert_eq!(sanitize_text(&input), input);
        }
    }

    #[test]
    fn keeps_slash_separators_and_adjacent_paths_when_attached() {
        let _attached = AttachmentOverride::set(true);
        assert_eq!(
            sanitize_text("no runtime for Skippy runtime 0.76.1 / ABI 0.1.71"),
            "no runtime for Skippy runtime 0.76.1 / ABI 0.1.71"
        );
        assert_eq!(
            sanitize_text("copied / to /srv/private/model.gguf"),
            "copied / to /srv/private/model.gguf"
        );
        assert_eq!(
            sanitize_text("startup failed at (/home/operator/x)"),
            "startup failed at (/home/operator/x)"
        );
    }

    /// The same warning, unattended (stderr redirected — service
    /// journal, container log, CI capture): the diagnostic category
    /// survives, the operator's paths do not.
    #[test]
    fn forced_attached_mode_preserves_paths() {
        // Before any override the probe consults the real stderr (a
        // captured pipe under the test harness): either value is
        // possible, so only assert the override once set.
        let _guard = AttachmentOverride::set(true);
        let attached_input = "startup failed at (/home/operator/x)";
        assert_eq!(sanitize_text(attached_input), attached_input);
        drop(_guard);
    }

    #[test]
    fn unattended_output_keeps_the_category_and_redacts_the_paths() {
        // Pin the mode: unforced, the test would ride on the harness's
        // captured stderr, and `--nocapture` at a terminal flips the
        // probe to attached.
        let _unattended = AttachmentOverride::set(false);
        assert_eq!(
            sanitize_text("no runtime for Skippy runtime 0.76.1 / ABI 0.1.71"),
            "no runtime for Skippy runtime 0.76.1 / ABI 0.1.71"
        );
        assert_eq!(
            sanitize_text("copied / to /srv/private/model.gguf"),
            "copied / to [REDACTED_PATH]"
        );
        assert_eq!(
            sanitize_text("startup failed at (/home/operator/x)"),
            "startup failed at ([REDACTED_PATH])"
        );
        // Credentials stay redacted in both modes.
        assert_eq!(
            sanitize_text("authorization Bearer secret-token"),
            "authorization Bearer [REDACTED]"
        );
    }

    #[test]
    fn redacts_the_operator_logging_privacy_credential_corpus() {
        let cases = [
            (
                "provider endpoint https://provider.example/v1?access_token=secret-access&model=qwen"
                    .to_string(),
                "secret-access".to_string(),
            ),
            (
                "callback provider.example/v1?api_key=secret-api-key&attempt=2".to_string(),
                "secret-api-key".to_string(),
            ),
            (
                "authorization Bearer secret-bearer-token".to_string(),
                "secret-bearer-token".to_string(),
            ),
            (
                "credential sk-supersecret-key".to_string(),
                "sk-supersecret-key".to_string(),
            ),
        ];

        for (input, secret) in cases {
            let sanitized = sanitize_text(&input);
            assert!(
                !sanitized.contains(&secret),
                "sanitized output leaked {secret:?}: {sanitized:?}"
            );
        }
    }

    #[test]
    fn preserves_fatal_context_and_local_paths_while_redacting_credentials_when_attached() {
        let _attached = AttachmentOverride::set(true);
        let sanitized = sanitize_text("startup failed opening /home/operator/config.toml");
        assert_eq!(
            sanitized,
            "startup failed opening /home/operator/config.toml"
        );

        let sanitized = sanitize_text("startup failed: authorization Bearer secret-token");
        assert_eq!(sanitized, "startup failed: authorization Bearer [REDACTED]");
    }

    #[test]
    fn excludes_sensitive_native_keys_case_insensitively() {
        let params = vec![
            ("API_KEY".to_string(), json!("secret")),
            ("ModelPath".to_string(), json!(r#"C:\private\model.gguf"#)),
            ("RequestID".to_string(), json!("request-1")),
            ("batch_size".to_string(), json!(32)),
        ];

        let safe = safe_native_params(&params);

        assert_eq!(safe.len(), 1);
        assert_eq!(safe.get("batch_size"), Some(&json!(32)));
    }
}

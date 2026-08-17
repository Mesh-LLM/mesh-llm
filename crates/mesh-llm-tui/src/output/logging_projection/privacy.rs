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
    let normalized = redact_sensitive_fragments(&redact_url_query_values(input));
    if normalized.chars().count() <= MAX_PRESENTATION_CHARS {
        normalized
    } else {
        format!(
            "{}... [TRUNCATED]",
            normalized
                .chars()
                .take(MAX_PRESENTATION_CHARS)
                .collect::<String>()
        )
    }
}

/// Keep the useful error category and surrounding diagnostic text while
/// replacing only the sensitive value or local path. Fatal output used to
/// collapse an entire message to `[REDACTED]`, which made ordinary startup
/// failures impossible to diagnose.
fn redact_sensitive_fragments(input: &str) -> String {
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
            } else if is_private_absolute_path(trimmed) {
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
    word.starts_with('/')
        || word.to_ascii_lowercase().starts_with("file://")
        || is_windows_absolute_path(word)
        || std::env::var("HOME")
            .ok()
            .is_some_and(|home| !home.is_empty() && word.contains(&home))
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
    use super::{safe_native_params, sanitize_text};
    use serde_json::json;

    fn operator_home() -> Option<String> {
        ["HOME", "USERPROFILE"].into_iter().find_map(|variable| {
            let home = std::env::var(variable).ok()?;
            (!home.is_empty() && std::path::Path::new(&home).is_absolute()).then_some(home)
        })
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
    fn redacts_unix_windows_unc_and_device_paths() {
        let cases = [
            "/Users/operator/.mesh-llm/config.toml",
            r"C:\Users\operator\.mesh-llm\config.toml",
            r"\\server\private\config.toml",
            r"\??\C:\private\config.toml",
        ];

        for path in cases {
            assert_eq!(
                sanitize_text(&format!("startup failed at {path}")),
                "startup failed at [REDACTED_PATH]"
            );
        }
    }

    #[test]
    fn redacts_the_operator_logging_privacy_corpus() {
        let mut cases = vec![
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
            (
                r#"native error at C:\Users\operator\.mesh-llm\config.toml"#.to_string(),
                r#"C:\Users\operator"#.to_string(),
            ),
            (
                r#"native error at \\server\private\model.gguf"#.to_string(),
                r#"\\server\private"#.to_string(),
            ),
            (
                r#"native error at \\?\C:\private\model.gguf"#.to_string(),
                r#"\\?\C:\private"#.to_string(),
            ),
        ];
        if let Some(home) = operator_home() {
            cases.push((format!("native error at {home}/.mesh-llm/token"), home));
        }

        for (input, secret) in cases {
            let sanitized = sanitize_text(&input);
            assert!(
                !sanitized.contains(&secret),
                "sanitized output leaked {secret:?}: {sanitized:?}"
            );
        }
    }

    #[test]
    fn preserves_fatal_context_while_selectively_redacting_credentials_and_paths() {
        let mut cases = vec![
            r"startup failed opening C:\Users\operator\config.toml".to_owned(),
            r"startup failed opening \\server\private\config.toml".to_owned(),
            "startup failed: authorization Bearer secret-token".to_owned(),
        ];
        if let Some(home) = operator_home() {
            cases.push(format!(
                "startup failed opening {home}/.mesh-llm/config.toml"
            ));
        }
        for input in cases {
            let sanitized = sanitize_text(&input);
            assert_ne!(sanitized, "[REDACTED]");
            assert!(sanitized.contains("startup failed"));
        }
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

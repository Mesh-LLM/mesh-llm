use super::descriptors::{Descriptor, GLOBAL_REDACTED, RawKind};
use super::raw_options::raw_option_is_allowed;
use super::vocabulary::{is_boolean_option, is_redacted_marker, is_static_summary_token};

#[derive(Clone, Copy)]
enum Phase {
    Booleans,
    Port,
    Redacted,
}

fn descriptor_matches(tokens: &[&str], descriptor: Descriptor) -> bool {
    descriptor.path.len() <= tokens.len()
        && descriptor
            .path
            .iter()
            .enumerate()
            .all(|(index, token)| is_static_summary_token(token) && tokens[index] == *token)
}

fn raw_value_is_valid(raw: RawKind, tokens: &[&str], index: usize) -> Option<usize> {
    let (option, valid_values): (&str, &[&str]) = match raw {
        RawKind::Backend => ("--backend", &["metal", "cuda", "hip", "intel"]),
        RawKind::Mode => ("--mode", &["disabled", "metrics", "enforce"]),
        RawKind::None => return Some(index),
    };
    (tokens.get(index) == Some(&option)
        && tokens
            .get(index + 1)
            .is_some_and(|value| valid_values.contains(value)))
    .then_some(index + 2)
}

pub(super) fn validate_descriptor(tokens: &[&str], descriptor: Descriptor) -> bool {
    if !descriptor_matches(tokens, descriptor) {
        return false;
    }
    let Some(mut index) = raw_value_is_valid(descriptor.raw, tokens, descriptor.path.len()) else {
        return false;
    };
    let mut phase = Phase::Booleans;
    let mut seen_booleans = Vec::new();
    let mut seen_redacted = Vec::new();
    let mut seen_tokens = Vec::new();
    let mut port_seen = false;
    let mut global_phase = false;
    let mut last_global_rank = 0;

    while index < tokens.len() {
        let Some(token) = tokens.get(index) else {
            return false;
        };
        if is_boolean_option(token) {
            if !matches!(phase, Phase::Booleans)
                || !descriptor.booleans.contains(token)
                || seen_booleans.contains(token)
            {
                return false;
            }
            seen_booleans.push(*token);
            seen_tokens.push(*token);
            index += 1;
            continue;
        }
        if *token == "--port" {
            if matches!(phase, Phase::Redacted)
                || port_seen
                || !descriptor.has_port
                || !raw_option_is_allowed(tokens, index, token)
            {
                return false;
            }
            let Some(value) = tokens.get(index + 1) else {
                return false;
            };
            if value.is_empty()
                || !value.bytes().all(|byte| byte.is_ascii_digit())
                || value.parse::<u16>().is_err()
            {
                return false;
            }
            port_seen = true;
            phase = Phase::Port;
            index += 2;
            continue;
        }
        if is_redacted_marker(token) {
            let is_global = GLOBAL_REDACTED.contains(token);
            if global_phase && !is_global {
                return false;
            }
            if is_global {
                let global_rank = match *token {
                    "--join" => 1,
                    "--root-relay" => 2,
                    "--relay-auth" => 3,
                    _ => 0,
                };
                if global_rank <= last_global_rank {
                    return false;
                }
                global_phase = true;
                last_global_rank = global_rank;
            }
            if (!descriptor.redacted.contains(token) && !GLOBAL_REDACTED.contains(token))
                || seen_redacted.contains(token)
                || tokens.get(index + 1) != Some(&"[REDACTED]")
            {
                return false;
            }
            seen_redacted.push(*token);
            seen_tokens.push(*token);
            phase = Phase::Redacted;
            index += 2;
            continue;
        }
        return false;
    }
    !descriptor
        .conflicts
        .iter()
        .any(|pair| pair.iter().all(|flag| seen_tokens.contains(flag)))
}

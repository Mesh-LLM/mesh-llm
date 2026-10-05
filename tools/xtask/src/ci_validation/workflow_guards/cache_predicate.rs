//! Conservative structural proof of a required cache authorization conjunct.
//! Calls and comparisons remain opaque; this does not evaluate GitHub expressions.
fn unwrap(value: &str) -> &str {
    let mut value = value.trim();
    if let Some(inner) = value.strip_prefix("${{").and_then(|v| v.strip_suffix("}}")) {
        value = inner.trim();
    }
    while value.starts_with('(') && closing(value) == Some(value.len() - 1) {
        value = value[1..value.len() - 1].trim();
    }
    value
}
fn closing(value: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut quoted = false;
    for (index, byte) in value.bytes().enumerate() {
        if byte == b'\'' {
            quoted = !quoted;
        }
        if quoted {
            continue;
        }
        if byte == b'(' {
            depth += 1;
        }
        if byte == b')' {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}
fn split<'a>(value: &'a str, operator: &[u8; 2]) -> Vec<&'a str> {
    let bytes = value.as_bytes();
    let (mut depth, mut quoted, mut start) = (0usize, false, 0usize);
    let mut parts = Vec::new();
    for index in 0..bytes.len().saturating_sub(1) {
        match bytes[index] {
            b'\'' => quoted = !quoted,
            b'(' if !quoted => depth += 1,
            b')' if !quoted => depth = depth.saturating_sub(1),
            _ => {}
        }
        if !quoted && depth == 0 && &bytes[index..index + 2] == operator {
            parts.push(value[start..index].trim());
            start = index + 2;
        }
    }
    if !parts.is_empty() {
        parts.push(value[start..].trim());
    }
    parts
}
fn normalized(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}
pub(super) fn requires(value: &str, clause: &str) -> bool {
    let value = unwrap(value);
    // Disabled fallback branches cannot authorize a cache operation.
    if matches!(value, "false" | "''") {
        return true;
    }
    // A complete OR expression can itself be a required conjunct. Recognize
    // that exact clause before decomposing its alternatives into weaker atoms.
    if normalized(value) == normalized(unwrap(clause)) {
        return true;
    }
    let alternatives = split(value, b"||");
    if !alternatives.is_empty() {
        return alternatives.iter().all(|v| requires(v, clause));
    }
    let conjuncts = split(value, b"&&");
    if !conjuncts.is_empty() {
        return conjuncts.iter().any(|v| requires(v, clause));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::requires;

    #[test]
    fn composite_conjuncts_require_every_reachable_authorization_branch() {
        let scope = "scheduled == 'true' || workload == 'cpu'";
        assert!(requires(&format!("({scope}) && trusted"), scope));
        assert!(requires(
            &format!("({scope}) && trusted"),
            &format!("({scope})")
        ));
        assert!(requires(scope, &format!("${{{{ ({scope}) }}}}")));
        assert!(!requires(&format!("({scope}) || true"), scope));
        assert!(!requires(&format!("true || ({scope})"), scope));
        assert!(!requires(
            &format!("(({scope}) && trusted) || bypass"),
            scope
        ));
        assert!(!requires(scope, "scheduled == 'true'"));
        assert!(requires(&format!("({scope}) || false"), scope));
        assert!(requires("false", scope));
    }
}

pub(super) fn local_path(reference: &str) -> Option<String> {
    if reference.starts_with('#')
        || reference.starts_with("data:")
        || reference.starts_with("mailto:")
    {
        return None;
    }
    let cleaned: String = reference
        .trim_start_matches(|ch| ch <= '\u{20}')
        .chars()
        .filter(|ch| !matches!(ch, '\t' | '\r' | '\n'))
        .collect();
    if let Some((scheme, _)) = cleaned.split_once(':')
        && scheme.starts_with(|ch: char| ch.is_ascii_alphabetic())
        && scheme
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
    {
        return None;
    }
    let path = cleaned.split(['?', '#']).next().unwrap_or("");
    let path = if let Some(authority) = path.strip_prefix("//") {
        let (host, path) = authority.split_once('/').unwrap_or((authority, ""));
        if !host.is_empty() {
            return None;
        }
        format!("/{path}")
    } else {
        path.to_owned()
    };
    let last_slash = path.rfind('/').map_or(0, |offset| offset + 1);
    let end = path[last_slash..]
        .find(';')
        .map_or(path.len(), |offset| last_slash + offset);
    let path = &path[..end];
    if path.is_empty() || path == "/" {
        return None;
    }
    Some(path.strip_prefix('/').unwrap_or(path).to_owned())
}

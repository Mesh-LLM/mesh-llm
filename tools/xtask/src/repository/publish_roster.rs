use crate::command::DynResult;
use std::collections::BTreeSet;

/// Reads the curated roster as data. Shell expressions are never evaluated.
pub(crate) fn parse(contents: &str) -> DynResult<Vec<String>> {
    let mut in_array = false;
    let mut closed = false;
    let mut crates = Vec::new();
    let mut seen = BTreeSet::new();
    for line in contents.lines() {
        let line = line.trim();
        if line == "publish_crates=(" {
            if in_array || closed {
                return Err("duplicate publish_crates array".into());
            }
            in_array = true;
            continue;
        }
        if !in_array {
            continue;
        }
        if line == ")" {
            in_array = false;
            closed = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name = if line.len() >= 2 && line.starts_with('"') && line.ends_with('"') {
            &line[1..line.len() - 1]
        } else {
            line
        };
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_".contains(&byte))
        {
            return Err(format!("invalid literal publish crate `{line}`").into());
        }
        if !seen.insert(name.to_owned()) {
            return Err(format!("duplicate publish crate `{name}`").into());
        }
        crates.push(name.to_owned());
        if crates.len() > 1024 {
            return Err("publish roster exceeds 1024 packages".into());
        }
    }
    if !closed || in_array || crates.is_empty() {
        return Err("missing, unterminated or empty publish_crates array".into());
    }
    Ok(crates)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roster_is_literal_data_and_preserves_curated_order() {
        assert_eq!(
            super::parse("exit 99\npublish_crates=(\n # note\n provider\n \"consumer\"\n)\n")
                .unwrap(),
            ["provider", "consumer"]
        );
        for source in [
            "publish_crates=(\n $(touch sentinel)\n)",
            "publish_crates=(\n a b\n)",
            "publish_crates=(\n \"\n)",
            "publish_crates=(\n a\n a\n)",
            "publish_crates=(\n a\n",
            "publish_crates=(\n)\n",
            "publish_crates=(\n a\n)\npublish_crates=(\n b\n)",
        ] {
            assert!(super::parse(source).is_err(), "accepted {source}");
        }
    }
}

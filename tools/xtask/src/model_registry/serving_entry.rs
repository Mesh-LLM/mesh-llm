//! Filename policy shared by registry selection and the family battery artifact owner.
//! This classifies serving entry names; it does not inspect model bytes or caches.
pub(crate) fn rank(name: &str) -> Option<u8> {
    if name.contains(['\n', '\r', '\0']) {
        return None;
    }
    let basename = name.rsplit('/').next()?.to_ascii_lowercase();
    let stem = basename.strip_suffix(".gguf")?;
    if basename.contains("mmproj") {
        return None;
    }
    let numbered = stem.rsplit_once("-of-").and_then(|(part, total)| {
        let (_, number) = part.rsplit_once('-')?;
        let digits =
            |value: &str| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit());
        (digits(number) && digits(total)).then_some((number, total))
    });
    match numbered {
        Some((number, total))
            if number.trim_start_matches('0') == "1"
                && !total.trim_start_matches('0').is_empty() =>
        {
            Some(0)
        }
        Some(_) => None,
        None => Some(1),
    }
}
#[cfg(test)]
mod tests {
    use super::rank;
    #[test]
    fn nonnumeric_of_names_are_unsharded_and_numeric_later_shards_are_refused() {
        for name in [
            "Model-of-Thought.gguf",
            "nested/Model-1-of-Thought.gguf",
            "Model-x-of-2.gguf",
            "MODEL-1-of-.GGUF",
        ] {
            assert_eq!(rank(name), Some(1), "{name}");
        }
        assert_eq!(rank("Model-00001-of-00002.gguf"), Some(0));
        for name in [
            "Model-00002-of-00002.gguf",
            "Model-00000-of-00002.gguf",
            "Model-00001-of-00000.gguf",
            "mmproj.gguf",
            "notes.txt",
            "unsafe\n.gguf",
        ] {
            assert_eq!(rank(name), None, "{name}");
        }
    }
}

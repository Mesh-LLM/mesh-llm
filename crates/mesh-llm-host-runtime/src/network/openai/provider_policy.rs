/// Provider models that require an explicit model id and must never be selected
/// by `auto` or the virtual `mesh` model.
pub(crate) fn is_explicit_only_model(model: &str) -> bool {
    model == "apple/system" || model.starts_with("apple/system@")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_system_aliases_are_explicit_only() {
        assert!(is_explicit_only_model("apple/system"));
        assert!(is_explicit_only_model("apple/system@27.0"));
        assert!(!is_explicit_only_model("org/apple-system-7b:q4_k_m"));
    }
}

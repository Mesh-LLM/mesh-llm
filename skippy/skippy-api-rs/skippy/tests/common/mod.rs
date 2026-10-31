//! Helpers shared by the integration-test binaries.

use std::env::VarError;

/// The one rule for model-path variables: unset and set-but-empty both mean
/// "no model for this lane" (`None`). Pure, so the rule is testable without
/// touching the process environment.
pub fn normalise(value: Result<String, VarError>) -> Option<String> {
    value.ok().filter(|path| !path.is_empty())
}

/// Reads a `SKIPPY_TEST_*` model-path variable through `normalise`. The just
/// recipe assigns all four variables identically and unconditionally, so an
/// empty value is the recipe's spelling of "not supplied" and must behave
/// exactly like unset.
pub fn model_env(variable: &str) -> Option<String> {
    normalise(std::env::var(variable))
}

#[cfg(test)]
mod tests {
    use super::normalise;
    use std::env::VarError;

    #[test]
    fn an_unset_variable_means_absent() {
        assert_eq!(normalise(Err(VarError::NotPresent)), None);
    }

    #[test]
    fn an_empty_variable_means_absent() {
        assert_eq!(normalise(Ok(String::new())), None);
    }

    #[test]
    fn a_set_variable_yields_its_path() {
        assert_eq!(normalise(Ok("x".to_owned())), Some("x".to_owned()));
    }
}

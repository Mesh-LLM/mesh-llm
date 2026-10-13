use super::{Options, seconds};
use std::time::Duration;

#[test]
fn migration_lifecycle_budget_rejects_invalid_or_unbounded_inputs() {
    for input in [
        "",
        "0",
        "01",
        "-1",
        "+1",
        "1.5",
        " 1",
        "1 ",
        "86401",
        "18446744073709551616",
    ] {
        let result = seconds(input);

        assert!(result.is_err());
    }
}

#[test]
fn migration_lifecycle_budget_accepts_supported_bounds() {
    for (input, expected) in [("1", 1), ("60", 60), ("86400", 86400)] {
        let result = seconds(input);

        assert_eq!(result.unwrap(), Duration::from_secs(expected));
    }
}

#[test]
fn migration_lifecycle_arguments_reject_missing_duplicate_and_unknown_flags() {
    for input in [
        vec![],
        vec!["--binary"],
        vec!["--binary", "relative"],
        vec!["--binary", "/one", "--binary", "/two"],
        vec!["--unknown", "value"],
        vec!["--ready-max-wait", "--binary"],
    ] {
        let arguments = input.into_iter().map(str::to_owned).collect::<Vec<_>>();

        let result = Options::parse(&arguments);

        assert!(result.is_err());
    }
}

pub(crate) fn python_int_text(text: &str) -> Option<String> {
    text.parse::<i128>().ok().map(|number| number.to_string())
}

pub(crate) fn python_int(text: &str) -> Option<i64> {
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{python_int, python_int_text};

    #[test]
    fn decimal_inputs_are_bounded_without_interpreter_coercion() {
        assert_eq!(python_int("7"), Some(7));
        assert_eq!(python_int("-10"), Some(-10));
        for rejected in [" 7\n", "-1_0", "1.0", "9223372036854775808"] {
            assert_eq!(python_int(rejected), None);
        }
        assert_eq!(
            python_int_text("170141183460469231731687303715884105728"),
            None
        );
    }
}

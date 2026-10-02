//! The value formatting `render_markdown` of `collect-ci-metrics.py` relies
//! on: `human()` durations, `markdown_escape()`, `or 'n/a'` truthiness and
//! the `:.1%` share format.

use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome};
use crate::ci_operations::ci_metrics_value::{Value, display};

static NULL: Value = Value::Null;

/// `mapping[key]` of a report dict; a missing key reads as `None`.
pub(crate) fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&NULL)
}

/// `mapping[key][:top]` of a report list.
pub(crate) fn head<'a>(value: &'a Value, key: &str, top: usize) -> &'a [Value] {
    match field(value, key) {
        Value::Array(items) => &items[..items.len().min(top)],
        _ => &[],
    }
}

/// Render an available worker count, including zero.
pub(crate) fn or_na(value: &Value) -> String {
    match value {
        Value::Int(count) if *count >= 0 => count.to_string(),
        _ => "n/a".to_owned(),
    }
}

/// `markdown_escape(value)`: `str(value)` with pipes escaped and newlines
/// flattened to spaces.
pub(crate) fn escape_text(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

pub(crate) fn escape(value: &Value) -> String {
    escape_text(&display(value))
}

/// `markdown_escape(value or 'n/a')`.
pub(crate) fn escape_or_na(value: &Value) -> String {
    match value {
        Value::Str(text) if !text.is_empty() => escape_text(text),
        _ => "n/a".to_owned(),
    }
}

/// `int(round(seconds))`: ties to even; `NaN` is a caught `ValueError`,
/// infinity an uncaught `OverflowError`.
fn rounded(value: &Value) -> Outcome<Option<i128>> {
    match value {
        Value::Null => Ok(None),
        Value::Int(int) if *int >= 0 => Ok(Some(*int)),
        Value::Float(float) if float.is_finite() && *float >= 0.0 => {
            format!("{:.0}", float.round_ties_even())
                .parse()
                .map(Some)
                .map_err(|_| {
                    Failure::Reported("CI duration is outside the supported range".to_owned())
                })
        }
        _ => Err(Failure::Reported(
            "CI duration must be a finite nonnegative number".to_owned(),
        )),
    }
}

/// `human(seconds)`: `n/a`, `Ns`, `Nm Ns` or `Nh Nm Ns` with Python's
/// floor `divmod`.
pub(crate) fn human(value: &Value) -> Outcome<String> {
    let Some(total) = rounded(value)? else {
        return Ok("n/a".to_owned());
    };
    let (hours, remainder) = (total.div_euclid(3600), total.rem_euclid(3600));
    let (minutes, seconds) = (remainder.div_euclid(60), remainder.rem_euclid(60));
    if hours != 0 {
        return Ok(format!("{hours}h {minutes}m {seconds}s"));
    }
    if minutes != 0 {
        return Ok(format!("{minutes}m {seconds}s"));
    }
    Ok(format!("{seconds}s"))
}

/// `f"{share:.1%}"`: the float times 100, correctly rounded to one place.
pub(crate) fn percent(value: &Value) -> Outcome<String> {
    let share = match value {
        Value::Float(float) => *float,
        Value::Int(int) => int.to_string().parse().unwrap_or(f64::INFINITY),
        _ => {
            return Err(Failure::Reported(
                "CI share must be a number between zero and one".to_owned(),
            ));
        }
    };
    if !share.is_finite() || !(0.0..=1.0).contains(&share) {
        return Err(Failure::Reported(
            "CI share must be a finite number between zero and one".to_owned(),
        ));
    }
    let scaled = share * 100.0;
    Ok(format!("{scaled:.1}%"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_formats_available_nonnegative_durations() {
        let cases = [
            (Value::Null, "n/a"),
            (Value::Float(0.5), "0s"),
            (Value::Float(1.5), "2s"),
            (Value::Float(59.6), "1m 0s"),
            (Value::Int(3600), "1h 0m 0s"),
            (Value::Float(3725.0), "1h 2m 5s"),
        ];
        for (value, expected) in cases {
            assert_eq!(human(&value).ok().as_deref(), Some(expected));
        }
    }

    #[test]
    fn shares_and_dimensions_have_domain_formats() {
        assert_eq!(
            percent(&Value::Float(0.3333)).ok().as_deref(),
            Some("33.3%")
        );
        assert_eq!(percent(&Value::Float(1.0)).ok().as_deref(), Some("100.0%"));
        assert_eq!(escape(&Value::text("a|b\nc")), "a\\|b c");
        assert_eq!(escape_or_na(&Value::text("")), "n/a");
        assert_eq!(escape_or_na(&Value::Bool(true)), "n/a");
        assert_eq!(or_na(&Value::Int(0)), "0");
    }

    #[test]
    fn invalid_numeric_report_values_are_rejected() {
        for value in [
            Value::Bool(true),
            Value::Float(-1.0),
            Value::Float(f64::NAN),
            Value::Float(f64::INFINITY),
            Value::text("1"),
        ] {
            assert!(human(&value).is_err());
            assert!(percent(&value).is_err());
        }
        assert!(percent(&Value::Float(1.1)).is_err());
    }
}

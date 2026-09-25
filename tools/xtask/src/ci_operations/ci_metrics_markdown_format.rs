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

/// Python truthiness.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Int(int) => *int != 0,
        Value::BigInt(_) => true,
        Value::Float(float) => *float != 0.0,
        Value::Str(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
    }
}

/// `value or 'n/a'`, as `str`.
pub(crate) fn or_na(value: &Value) -> String {
    if truthy(value) {
        display(value)
    } else {
        "n/a".to_owned()
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
    escape_text(&or_na(value))
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Str(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        _ => "int",
    }
}

/// `int(round(seconds))`: ties to even; `NaN` is a caught `ValueError`,
/// infinity an uncaught `OverflowError`.
fn rounded(value: &Value) -> Outcome<Option<i128>> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(flag) => Ok(Some(i128::from(*flag))),
        Value::Int(int) => Ok(Some(*int)),
        Value::Float(float) if float.is_nan() => Err(Failure::Reported(
            "cannot convert float NaN to integer".to_owned(),
        )),
        Value::Float(float) if float.is_infinite() => Err(Failure::Uncaught(
            "OverflowError: cannot convert float infinity to integer".to_owned(),
        )),
        Value::Float(float) => format!("{:.0}", float.round_ties_even())
            .parse()
            .map(Some)
            .map_err(|_| {
                Failure::Uncaught("OverflowError: seconds beyond the i128 range".to_owned())
            }),
        other => Err(Failure::Uncaught(format!(
            "TypeError: type {} doesn't define __round__ method",
            type_name(other)
        ))),
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
        Value::Bool(flag) => f64::from(u8::from(*flag)),
        other => {
            return Err(Failure::Reported(format!(
                "Unknown format code '%' for object of type '{}'",
                type_name(other)
            )));
        }
    };
    let scaled = share * 100.0;
    if scaled.is_nan() {
        return Ok("nan%".to_owned());
    }
    if scaled.is_infinite() {
        return Ok(if scaled > 0.0 { "inf%" } else { "-inf%" }.to_owned());
    }
    Ok(format!("{scaled:.1}%"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_matches_python() {
        let cases = [
            (Value::Null, "n/a"),
            (Value::Float(0.5), "0s"),
            (Value::Float(1.5), "2s"),
            (Value::Float(59.6), "1m 0s"),
            (Value::Int(3600), "1h 0m 0s"),
            (Value::Float(3725.0), "1h 2m 5s"),
            (Value::Float(-61.0), "-1h 58m 59s"),
        ];
        for (value, expected) in cases {
            assert_eq!(human(&value).ok().as_deref(), Some(expected));
        }
    }

    #[test]
    fn percent_and_escape_match_python() {
        assert_eq!(
            percent(&Value::Float(0.3333)).ok().as_deref(),
            Some("33.3%")
        );
        assert_eq!(percent(&Value::Float(1.0)).ok().as_deref(), Some("100.0%"));
        assert_eq!(escape(&Value::text("a|b\nc")), "a\\|b c");
        assert_eq!(escape_or_na(&Value::text("")), "n/a");
    }
}

use super::ci_metrics_calendar::days_in_month;
use crate::ci_operations::json_access::Outcome;

pub(crate) fn check_timestamp(value: &str) -> Outcome<()> {
    if value.len() != 14 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("producer timestamp must contain fourteen ASCII digits".into());
    }
    let field = |start: usize, end: usize| {
        value[start..end]
            .parse::<i64>()
            .map_err(|error| error.to_string())
    };
    let year = field(0, 4)?;
    let month = field(4, 6)?;
    let day = field(6, 8)?;
    let hour = field(8, 10)?;
    let minute = field(10, 12)?;
    let second = field(12, 14)?;
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err("producer timestamp contains an invalid date or time".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_timestamp;

    #[test]
    fn fixed_width_timestamp_validates_calendar_bounds() {
        for valid in ["20260914210022", "20240229000000"] {
            assert!(check_timestamp(valid).is_ok());
        }
        for invalid in [
            "20261301120000",
            "20260230120000",
            "20260101235960",
            "00000101000000",
            "20260100000000",
            "20250229000000",
            "2026091",
            "abcdefghijklmno",
        ] {
            assert!(check_timestamp(invalid).is_err());
        }
    }
}

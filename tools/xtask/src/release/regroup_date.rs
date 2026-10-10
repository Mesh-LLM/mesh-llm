pub(crate) fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let field = |start: usize, end: usize| {
        bytes[start..end].iter().try_fold(0_u32, |value, digit| {
            digit
                .is_ascii_digit()
                .then(|| value * 10 + u32::from(*digit - b'0'))
        })
    };
    let (Some(year), Some(month), Some(day)) = (field(0, 4), field(5, 7), field(8, 10)) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=9999).contains(&year) && (1..=12).contains(&month) && (1..=days).contains(&day)
}

#[cfg(test)]
mod tests {
    use super::is_iso_date;

    #[test]
    fn release_dates_require_calendar_format_and_valid_day() {
        for accepted in ["2026-01-01", "2024-02-29", "0001-01-01", "9999-12-31"] {
            assert!(is_iso_date(accepted), "{accepted}");
        }
        for rejected in [
            "0000-01-01",
            "2026-02-29",
            "2026-13-01",
            "2026-01-00",
            "20260101",
            "2026010100",
            "2026-W01",
            "2026W011xx",
            "2026-1-01",
            "",
        ] {
            assert!(!is_iso_date(rejected), "{rejected}");
        }
    }
}

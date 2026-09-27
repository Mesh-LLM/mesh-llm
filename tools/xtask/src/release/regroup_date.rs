//! Whether CPython 3.13's C `date.fromisoformat(text)` accepts `text`:
//! a UTF-8 length of 7, 8 or 10 bytes, then `YYYY[-]MM[-]DD` or
//! `YYYY[-]Www[[-]D]`, with trailing bytes past the parsed fields ignored.

/// `parse_digits(p, n)`: exactly `n` ASCII digits (NUL past the end).
fn digits(bytes: &[u8], at: &mut usize, count: usize) -> Option<i64> {
    let mut value = 0;
    for _ in 0..count {
        let byte = bytes.get(*at).copied().unwrap_or(0);
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + i64::from(byte - b'0');
        *at += 1;
    }
    Some(value)
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// `date.fromisoformat(text)` succeeds.
pub(crate) fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    if !matches!(bytes.len(), 7 | 8 | 10) {
        return false;
    }
    let mut at = 0;
    let Some(year) = digits(bytes, &mut at, 4) else {
        return false;
    };
    let separator = bytes.get(at) == Some(&b'-');
    if separator {
        at += 1;
    }
    if bytes.get(at) == Some(&b'W') {
        at += 1;
        return iso_week(bytes, at, year, separator);
    }
    let Some(month) = digits(bytes, &mut at, 2) else {
        return false;
    };
    if separator {
        if bytes.get(at) != Some(&b'-') {
            return false;
        }
        at += 1;
    }
    let Some(day) = digits(bytes, &mut at, 2) else {
        return false;
    };
    (1..=9999).contains(&year)
        && (1..=12).contains(&month)
        && (1..=days_in_month(year, month)).contains(&day)
}

/// The ISO-calendar branch and `iso_to_ymd`'s range checks.
fn iso_week(bytes: &[u8], mut at: usize, year: i64, separator: bool) -> bool {
    let Some(week) = digits(bytes, &mut at, 2) else {
        return false;
    };
    let mut day = 1;
    if at < bytes.len() {
        if separator {
            if bytes.get(at) != Some(&b'-') {
                return false;
            }
            at += 1;
        }
        match digits(bytes, &mut at, 1) {
            Some(value) => day = value,
            None => return false,
        }
    }
    if !(1..=9999).contains(&year) {
        return false;
    }
    if !(1..53).contains(&week) {
        let prior = year - 1;
        let ordinal = prior * 365 + prior / 4 - prior / 100 + prior / 400 + 1;
        let first_weekday = (ordinal + 6) % 7;
        let long_year = first_weekday == 3 || (first_weekday == 2 && is_leap(year));
        if week != 53 || !long_year {
            return false;
        }
    }
    // 9999-12-31 is (9999, 52, 5); anything later leaves the year range.
    (1..8).contains(&day) && !(year == 9999 && week == 52 && day > 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_release_regroup_dates_match_fromisoformat() {
        let accepted = [
            "2026-01-01",
            "20260101",
            "2026010100",
            "20260101.0",
            "2026-W01",
            "2026W011",
            "2026W011xx",
            "2026-W53",
            "2020-W53",
            "2026-W01-3",
            "2024-02-29",
            "0001-W01",
            "9999-W52-5",
        ];
        let rejected = [
            "0000-01-01",
            "2026-02-29",
            "2026-W01-8",
            "2026W01-",
            "2026-0101",
            "20261301",
            "2026-02-29x",
            "2026-W1x",
            "0000-W01",
            "9999-W52-7",
            "2021-W53",
            "2026-1-01",
            "",
        ];
        for text in accepted {
            assert!(is_iso_date(text), "{text}");
        }
        for text in rejected {
            assert!(!is_iso_date(text), "{text}");
        }
    }
}

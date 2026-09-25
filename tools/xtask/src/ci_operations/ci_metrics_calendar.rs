//! Proleptic Gregorian calendar arithmetic for `ci_metrics_time`, as
//! CPython's `datetime` implements it (ordinals count from 0001-01-01).

pub(crate) fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub(crate) fn days_before_year(year: i32) -> i64 {
    let y = i64::from(year) - 1;
    y * 365 + y / 4 - y / 100 + y / 400
}

pub(crate) fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 0001-01-01 of a valid date.
pub(crate) fn ordinal(year: i64, month: i64, day: i64) -> i64 {
    let before_year = (year - 1) * 365 + (year - 1) / 4 - (year - 1) / 100 + (year - 1) / 400;
    let before_month: i64 = (1..month).map(|m| days_in_month(year, m)).sum();
    before_year + before_month + day - 1
}

pub(crate) fn civil(mut days: i64) -> (i64, i64, i64) {
    let mut year = 1;
    loop {
        let length = if is_leap(year) { 366 } else { 365 };
        if days < length {
            break;
        }
        days -= length;
        year += 1;
    }
    let mut month = 1;
    while days >= days_in_month(year, month) {
        days -= days_in_month(year, month);
        month += 1;
    }
    (year, month, days + 1)
}

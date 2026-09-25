//! `timestamp()` of `collect-ci-metrics.py`: CPython 3.13's C
//! `datetime.fromisoformat` (separator detection, ISO week dates, compact
//! and extended times, `,`/`.` fractions truncated to microseconds, `Z` and
//! `±HH[:MM[:SS[.ffffff]]]` offsets), naive values taken as UTC, conversion
//! to UTC, and `isoformat()` of the result. Instants are microseconds since
//! 0001-01-01T00:00:00 UTC.

use crate::ci_operations::ci_metrics_calendar::{
    civil, days_before_year, days_in_month, is_leap, ordinal,
};
use crate::repository::python_text::repr;

const DAY_US: i64 = 86_400_000_000;
/// Ordinal (days since 0001-01-01) of 1971-01-01; earlier years are `None`.
const FIRST_KEPT_YEAR: i32 = 1971;

/// A UTC instant with microsecond precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Instant(i64);

pub(crate) enum TimeError {
    /// `ValueError`, reported as `invalid timestamp {value!r}`.
    Invalid(String),
    /// `OverflowError` from `astimezone` (uncaught by the legacy script).
    Overflow,
}

/// `timestamp(value)` for a non-empty string.
pub(crate) fn timestamp(value: &str) -> Result<Option<Instant>, TimeError> {
    let value = match value.strip_suffix('Z') {
        Some(stem) => format!("{stem}+00:00"),
        None => value.to_owned(),
    };
    let invalid = || TimeError::Invalid(format!("invalid timestamp {}", repr(&value)));
    let (local, offset) = from_isoformat(&value).ok_or_else(invalid)?;
    let utc = local - offset;
    if !(0..days_before_year(10_000) * DAY_US).contains(&utc) {
        return Err(TimeError::Overflow);
    }
    Ok((utc >= days_before_year(FIRST_KEPT_YEAR) * DAY_US).then_some(Instant(utc)))
}

impl Instant {
    pub(crate) fn micros(self) -> i64 {
        self.0
    }

    /// `dt.datetime.now(dt.timezone.utc)`.
    pub(crate) fn now() -> Self {
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| i64::try_from(since.as_micros()).unwrap_or(0));
        Self(days_before_year(1970) * DAY_US + unix)
    }
}

/// `(end - start).total_seconds()` when not negative.
pub(crate) fn elapsed(start: Option<Instant>, end: Option<Instant>) -> Option<f64> {
    let micros = end?.0 - start?.0;
    let seconds = micros as f64 / 1e6;
    (micros >= 0).then_some(seconds)
}

/// `datetime.isoformat()` of an aware UTC datetime.
pub(crate) fn isoformat(instant: Instant) -> String {
    let days = instant.0.div_euclid(DAY_US);
    let micros = instant.0.rem_euclid(DAY_US);
    let (year, month, day) = civil(days);
    let seconds = micros / 1_000_000;
    let fraction = micros % 1_000_000;
    let mut text = format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    );
    if fraction != 0 {
        text.push_str(&format!(".{fraction:06}"));
    }
    text.push_str("+00:00");
    text
}

/// NUL-terminated byte view, as the C parser reads past `len`.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn at(&self, index: usize) -> u8 {
        self.0.get(index).copied().unwrap_or(0)
    }

    fn digits(&self, start: usize, count: usize) -> Option<(i64, usize)> {
        let mut value = 0;
        for index in start..start + count {
            let byte = self.at(index);
            if !byte.is_ascii_digit() {
                return None;
            }
            value = value * 10 + i64::from(byte - b'0');
        }
        Some((value, start + count))
    }
}

/// Local microseconds and UTC offset microseconds, or `None` when invalid.
fn from_isoformat(text: &str) -> Option<(i64, i64)> {
    if text.chars().count() < 7 {
        return None;
    }
    let bytes = Bytes(text.as_bytes());
    let separator = separator_location(&bytes)?;
    let (year, month, day) = parse_date(&bytes, separator)?;
    let mut time = (0, 0, 0, 0, None);
    if bytes.0.len() > separator {
        let width = match bytes.at(separator) {
            byte if byte & 0x80 == 0 => 1,
            byte if byte & 0xf0 == 0xe0 => 3,
            byte if byte & 0xf0 == 0xf0 => 4,
            _ => 2,
        };
        time = parse_time(&Bytes(bytes.0.get(separator + width..).unwrap_or_default()))?;
    }
    let (hour, minute, second, micro, offset) = time;
    let valid = (1..=9999).contains(&year)
        && (1..=12).contains(&month)
        && (1..=days_in_month(year, month)).contains(&day)
        && hour < 24
        && minute < 60
        && second < 60;
    let offset = offset.unwrap_or(0);
    if !valid || offset.abs() >= DAY_US {
        return None;
    }
    let local = ordinal(year, month, day) * DAY_US
        + ((hour * 60 + minute) * 60 + second) * 1_000_000
        + micro;
    Some((local, offset))
}

fn separator_location(bytes: &Bytes<'_>) -> Option<usize> {
    let len = bytes.0.len();
    if len == 7 {
        return Some(7);
    }
    if bytes.at(4) == b'-' {
        if bytes.at(5) != b'W' {
            return Some(10);
        }
        if len > 8 && bytes.at(8) == b'-' {
            if len == 9 {
                return None;
            }
            return Some(if len > 10 && bytes.at(10).is_ascii_digit() {
                8
            } else {
                10
            });
        }
        return Some(8);
    }
    if bytes.at(4) != b'W' {
        return Some(8);
    }
    let mut index = 7;
    while index < len && bytes.at(index).is_ascii_digit() {
        index += 1;
    }
    Some(if index < 9 {
        index
    } else if index % 2 == 0 {
        7
    } else {
        8
    })
}

fn parse_date(bytes: &Bytes<'_>, len: usize) -> Option<(i64, i64, i64)> {
    let (year, mut p) = bytes.digits(0, 4)?;
    let uses_separator = bytes.at(p) == b'-';
    if uses_separator {
        p += 1;
    }
    if bytes.at(p) == b'W' {
        let (week, mut p) = bytes.digits(p + 1, 2)?;
        let mut weekday = 1;
        if p < len {
            if uses_separator {
                if bytes.at(p) != b'-' {
                    return None;
                }
                p += 1;
            }
            weekday = bytes.digits(p, 1)?.0;
        }
        return iso_to_ymd(year, week, weekday);
    }
    let (month, mut p) = bytes.digits(p, 2)?;
    if uses_separator {
        if bytes.at(p) != b'-' {
            return None;
        }
        p += 1;
    }
    let (day, _) = bytes.digits(p, 2)?;
    Some((year, month, day))
}

fn iso_to_ymd(year: i64, week: i64, weekday: i64) -> Option<(i64, i64, i64)> {
    if !(1..=9999).contains(&year) {
        return None;
    }
    let first = ordinal(year, 1, 1);
    let first_weekday = first % 7;
    let long_year = first_weekday == 3 || (first_weekday == 2 && is_leap(year));
    if !(1..53).contains(&week) && !(week == 53 && long_year) {
        return None;
    }
    if !(1..8).contains(&weekday) {
        return None;
    }
    let mut monday = first - first_weekday;
    if first_weekday > 3 {
        monday += 7;
    }
    Some(civil(monday + (week - 1) * 7 + weekday - 1))
}

type Time = (i64, i64, i64, i64, Option<i64>);

fn parse_time(bytes: &Bytes<'_>) -> Option<Time> {
    let end = bytes.0.len();
    let tz = (0..end.max(1))
        .find(|index| matches!(bytes.at(*index), b'Z' | b'+' | b'-'))
        .unwrap_or(end.max(1));
    let (clock, trailing) = parse_clock(bytes, 0, tz)?;
    if tz >= end {
        return (!trailing && tz == end).then_some((clock[0], clock[1], clock[2], clock[3], None));
    }
    if bytes.at(tz) == b'Z' {
        return (tz + 1 == end).then_some((clock[0], clock[1], clock[2], clock[3], Some(0)));
    }
    let sign = if bytes.at(tz) == b'-' { -1 } else { 1 };
    let (zone, zone_trailing) = parse_clock(bytes, tz + 1, end)?;
    if zone_trailing {
        return None;
    }
    let offset = sign * ((zone[0] * 3600 + zone[1] * 60 + zone[2]) * 1_000_000 + zone[3]);
    Some((clock[0], clock[1], clock[2], clock[3], Some(offset)))
}

/// `parse_hh_mm_ss_ff`: fields plus whether unparsed text remains.
fn parse_clock(bytes: &Bytes<'_>, start: usize, end: usize) -> Option<([i64; 4], bool)> {
    let mut values = [0_i64; 4];
    let mut p = start;
    let mut has_separator = true;
    for (index, slot) in values.iter_mut().take(3).enumerate() {
        let (value, next) = bytes.digits(p, 2)?;
        *slot = value;
        let c = bytes.at(next);
        p = next + 1;
        if index == 0 {
            has_separator = c == b':';
        }
        if c == b'.' || c == b',' {
            if p >= end {
                return None;
            }
            break;
        } else if p >= end {
            return Some((values, c != 0));
        } else if has_separator && c == b':' {
            continue;
        } else if !has_separator {
            p -= 1;
        } else {
            return None;
        }
    }
    let count = (end - p).min(6);
    let (micro, mut p) = bytes.digits(p, count)?;
    values[3] = micro * 10_i64.pow(u32::try_from(6 - count).unwrap_or(0));
    while bytes.at(p).is_ascii_digit() {
        p += 1;
    }
    Some((values, bytes.at(p) != 0))
}

use super::ci_metrics_calendar::{civil, days_before_year, days_in_month, ordinal};

const DAY_US: i64 = 86_400_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Instant(i64);

pub(crate) enum TimeError {
    Invalid(String),
    Overflow,
}

pub(crate) fn timestamp(value: &str) -> Result<Option<Instant>, TimeError> {
    let (local, offset) = parse_timestamp(value)
        .ok_or_else(|| TimeError::Invalid(format!("invalid timestamp {value:?}")))?;
    let utc = local - offset;
    if !(0..days_before_year(10_000) * DAY_US).contains(&utc) {
        return Err(TimeError::Overflow);
    }
    Ok((utc >= days_before_year(1971) * DAY_US).then_some(Instant(utc)))
}

impl Instant {
    pub(crate) fn micros(self) -> i64 {
        self.0
    }

    pub(crate) fn now() -> Self {
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| i64::try_from(since.as_micros()).unwrap_or(0));
        Self(days_before_year(1970) * DAY_US + unix)
    }
}

pub(crate) fn elapsed(start: Option<Instant>, end: Option<Instant>) -> Option<f64> {
    let micros = end?.0 - start?.0;
    (micros >= 0).then_some(micros as f64 / 1e6)
}

pub(crate) fn isoformat(instant: Instant) -> String {
    let (year, month, day) = civil(instant.0.div_euclid(DAY_US));
    let micros = instant.0.rem_euclid(DAY_US);
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

fn decimal(text: &str, width: usize) -> Option<i64> {
    (text.len() == width && text.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| text.parse().ok())
        .flatten()
}

fn parse_timestamp(text: &str) -> Option<(i64, i64)> {
    if !text.is_ascii() {
        return None;
    }
    let (date, time) = text.split_once('T').or_else(|| text.split_once(' '))?;
    let parts = date.split('-').collect::<Vec<_>>();
    let [year, month, day] = parts.as_slice() else {
        return None;
    };
    let year = decimal(year, 4)?;
    let month = decimal(month, 2)?;
    let day = decimal(day, 2)?;
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
    {
        return None;
    }
    let (clock, offset) = split_zone(time)?;
    Some((
        ordinal(year, month, day) * DAY_US + parse_clock(clock)?,
        offset,
    ))
}

fn split_zone(text: &str) -> Option<(&str, i64)> {
    if let Some(clock) = text.strip_suffix('Z') {
        return Some((clock, 0));
    }
    let Some(index) = text.find(['+', '-']) else {
        return Some((text, 0));
    };
    let sign = if text.as_bytes()[index] == b'-' {
        -1
    } else {
        1
    };
    let zone = &text[index + 1..];
    let (hour, minute) = zone.split_once(':')?;
    let hour = decimal(hour, 2)?;
    let minute = decimal(minute, 2)?;
    if hour > 23 || minute > 59 {
        return None;
    }
    Some((&text[..index], sign * (hour * 60 + minute) * 60_000_000))
}

fn parse_clock(text: &str) -> Option<i64> {
    let (clock, fraction) = text.split_once('.').unwrap_or((text, ""));
    let fields = clock.split(':').collect::<Vec<_>>();
    let [hour, minute, second] = fields.as_slice() else {
        return None;
    };
    let hour = decimal(hour, 2)?;
    let minute = decimal(minute, 2)?;
    let second = decimal(second, 2)?;
    if hour > 23
        || minute > 59
        || second > 59
        || fraction.len() > 6
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || (text.contains('.') && fraction.is_empty())
    {
        return None;
    }
    let micro = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<i64>().ok()? * 10_i64.pow(u32::try_from(6 - fraction.len()).ok()?)
    };
    Some((hour * 3600 + minute * 60 + second) * 1_000_000 + micro)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_iso_timestamp_offsets_and_fractions_are_preserved() {
        let utc = timestamp("2026-07-01T00:00:01.5Z").ok().flatten().unwrap();
        let offset = timestamp("2026-07-01T02:00:01.500000+02:00")
            .ok()
            .flatten()
            .unwrap();
        assert_eq!(utc, offset);
        assert_eq!(isoformat(utc), "2026-07-01T00:00:01.500000+00:00");
    }

    #[test]
    fn invalid_dates_and_interpreter_specific_forms_are_rejected() {
        for value in [
            "2026-02-30T00:00:00Z",
            "20260701T000000",
            "2026-W27-3T00:00:00",
            "2026-07-01x00:00:00",
            "2026-07-01T00:00:00,5Z",
        ] {
            assert!(timestamp(value).is_err());
        }
    }
}

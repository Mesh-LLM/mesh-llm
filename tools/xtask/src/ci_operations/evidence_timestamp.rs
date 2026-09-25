//! `datetime.strptime(value, "%Y%m%d%H%M%S")` as used on producer origin
//! timestamps, including CPython's error wording.

use crate::ci_operations::python_access::Outcome;

/// `datetime.strptime(value, "%Y%m%d%H%M%S")` for fourteen ASCII digits,
/// with the regex backtracking and range errors of CPython's `_strptime`.
pub(crate) fn check_timestamp(value: &str) -> Outcome<()> {
    const D: &str = "0123456789";
    // Each field lists its regex alternatives in order; each alternative is
    // one allowed character set per position.
    const FIELDS: [&[&[&str]]; 6] = [
        &[&[D, D, D, D]],
        &[&["1", "012"], &["0", "123456789"], &["123456789"]],
        &[
            &["3", "01"],
            &["12", D],
            &["0", "123456789"],
            &["123456789"],
        ],
        &[&["2", "0123"], &["01", D], &[D]],
        &[&["012345", D], &[D]],
        &[&["6", "01"], &["012345", D], &[D]],
    ];
    let bytes = value.as_bytes();
    let Some(ends) = match_fields(&FIELDS, bytes, 0) else {
        let shown = crate::repository::python_text::repr(value);
        return Err(format!(
            "time data {shown} does not match format '%Y%m%d%H%M%S'"
        ));
    };
    let end = ends.last().copied().unwrap_or(0);
    if end != bytes.len() {
        return Err(format!("unconverted data remains: {}", &value[end..]));
    }
    let mut start = 0;
    let parts: Vec<u32> = ends
        .iter()
        .map(|&stop| {
            let part = value[start..stop].parse().unwrap_or(0);
            start = stop;
            part
        })
        .collect();
    let (year, month, day, second) = (parts[0], parts[1], parts[2], parts[5]);
    if year == 0 {
        return Err("year 0 is out of range".to_owned());
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day > days[month as usize - 1] {
        return Err("day is out of range for month".to_owned());
    }
    if second > 59 {
        return Err("second must be in 0..59".to_owned());
    }
    Ok(())
}

fn alternative_len(sets: &[&str], bytes: &[u8], pos: usize) -> Option<usize> {
    for (offset, set) in sets.iter().enumerate() {
        let byte = *bytes.get(pos + offset)?;
        if !set.as_bytes().contains(&byte) {
            return None;
        }
    }
    Some(sets.len())
}

fn match_fields(fields: &[&[&[&str]]], bytes: &[u8], pos: usize) -> Option<Vec<usize>> {
    let Some((first, rest)) = fields.split_first() else {
        return Some(Vec::new());
    };
    first.iter().find_map(|sets| {
        let end = pos + alternative_len(sets, bytes, pos)?;
        let mut ends = match_fields(rest, bytes, end)?;
        ends.insert(0, end);
        Some(ends)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_operations_timestamp_matches_strptime() {
        for (value, expected) in [
            ("20260914210022", Ok(())),
            ("20240229000000", Ok(())),
            ("20261301120000", Err("unconverted data remains: 0")),
            ("20260230120000", Err("day is out of range for month")),
            ("20260101235960", Err("second must be in 0..59")),
            ("00000101000000", Err("year 0 is out of range")),
            (
                "20260100000000",
                Err("time data '20260100000000' does not match format '%Y%m%d%H%M%S'"),
            ),
            ("20250229000000", Err("day is out of range for month")),
        ] {
            assert_eq!(
                check_timestamp(value),
                expected.map_err(str::to_owned),
                "{value}"
            );
        }
    }
}

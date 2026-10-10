use crate::repository::check_report::CheckReport;
use std::cmp::Ordering;

struct Version<'a> {
    components: [&'a str; 3],
    prerelease: Option<Vec<Identifier<'a>>>,
}

enum Identifier<'a> {
    Numeric(&'a str),
    Text(&'a str),
}

fn decimal(left: &str, right: &str) -> Ordering {
    let left = left.trim_start_matches('0');
    let right = right.trim_start_matches('0');
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn parse(text: &str) -> Option<Version<'_>> {
    let (core, suffix) = text
        .split_once('-')
        .map_or((text, None), |(core, suffix)| (core, Some(suffix)));
    let [major, minor, patch] = core.split('.').collect::<Vec<_>>().try_into().ok()?;
    if ![major, minor, patch]
        .iter()
        .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return None;
    }
    let prerelease = suffix.map(|suffix| {
        suffix
            .split('.')
            .map(|part| {
                if part.bytes().all(|byte| byte.is_ascii_digit()) {
                    Identifier::Numeric(part)
                } else {
                    Identifier::Text(part)
                }
            })
            .collect::<Vec<_>>()
    });
    if suffix.is_some_and(|value| {
        value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'-')
    }) {
        return None;
    }
    Some(Version {
        components: [major, minor, patch],
        prerelease,
    })
}

fn compare(current: &Version<'_>, target: &Version<'_>) -> Ordering {
    for (left, right) in current.components.iter().zip(target.components) {
        let order = decimal(left, right);
        if order != Ordering::Equal {
            return order;
        }
    }
    match (&current.prerelease, &target.prerelease) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => {
            for (left, right) in left.iter().zip(right) {
                let order = match (left, right) {
                    (Identifier::Numeric(left), Identifier::Numeric(right)) => decimal(left, right),
                    (Identifier::Numeric(_), Identifier::Text(_)) => Ordering::Less,
                    (Identifier::Text(_), Identifier::Numeric(_)) => Ordering::Greater,
                    (Identifier::Text(left), Identifier::Text(right)) => left.cmp(right),
                };
                if order != Ordering::Equal {
                    return order;
                }
            }
            left.len().cmp(&right.len())
        }
    }
}

pub(crate) fn run(args: &[String]) -> CheckReport {
    let [current, target] = args else {
        return CheckReport {
            code: 2,
            ..CheckReport::default()
        };
    };
    let (Some(current), Some(target)) = (parse(current), parse(target)) else {
        return CheckReport {
            code: 2,
            ..CheckReport::default()
        };
    };
    CheckReport {
        code: i32::from(compare(&current, &target) == Ordering::Greater),
        ..CheckReport::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_release_preflight_compares_versions_and_prereleases() {
        let cases = [
            ("0.76.1", "0.76.2", 0),
            ("0.76.1", "0.76.1", 0),
            ("0.76.2", "0.76.1", 1),
            ("0.76.2-rc.2", "0.76.2", 0),
            ("0.76.2", "0.76.2-rc.2", 1),
            ("0.76.2-rc.10", "0.76.2-rc.2", 1),
            ("0.76.2-rc.2", "0.76.2-rc.10", 0),
            ("0.76.2-rc.2", "0.76.2-rc.2.1", 0),
            ("999999999999999999999.0.0", "1000000000000000000000.0.0", 0),
            ("invalid", "0.76.2", 2),
        ];
        for (current, target, expected) in cases {
            assert_eq!(
                run(&[current.to_owned(), target.to_owned()]).code,
                expected,
                "{current} -> {target}"
            );
        }
    }
}

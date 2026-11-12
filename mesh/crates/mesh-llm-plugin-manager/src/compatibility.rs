//! Host compatibility of a plugin release.
//!
//! A release declares which MeshLLM host versions it works with. The
//! declaration travels two ways: the catalog entry records it for the pinned
//! release, and the packaged `plugin.toml` records it for the archive that is
//! actually installed. An install retains the declaration it was installed
//! with, so later host upgrades can judge the installed version without
//! inferring it from the catalog's newest release.
//!
//! Missing metadata means [`Verdict::Unknown`], never incompatible: a legacy
//! install that predates compatibility metadata must keep working, and only
//! its existing runtime checks (protocol negotiation) guard it.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// The host-version requirement one release declares. Both bounds are
/// inclusive; either may be absent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostCompatibility {
    /// Oldest host version this release works with, such as `0.81.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<String>,
    /// Newest supported host version, inclusive: `0.81.0` includes that
    /// release, but excludes `0.81.1` and later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<String>,
}

impl HostCompatibility {
    pub fn validate(&self) -> Result<()> {
        for bound in [&self.min, &self.max].into_iter().flatten() {
            parse_bound(bound).map_err(|error| {
                anyhow::anyhow!("invalid host compatibility bound {bound:?}: {error}")
            })?;
        }
        Ok(())
    }
}

/// Whether a release works with `host_version`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Every declared bound includes `host_version`.
    Compatible,
    /// A declared bound excludes `host_version`; the reason says which.
    Incompatible(String),
    /// No usable declaration: nothing can be concluded from metadata alone.
    Unknown(String),
}

impl Verdict {
    /// The reason a release does not work with the host, when it is known to
    /// be incompatible.
    pub fn incompatible_reason(&self) -> Option<&str> {
        match self {
            Self::Incompatible(reason) => Some(reason),
            _ => None,
        }
    }
}

/// Judge `compatibility` (a retained or catalog declaration) against
/// `host_version`. An absent declaration is [`Verdict::Unknown`], and so is
/// one that cannot be parsed: unknown metadata never makes a plugin
/// incompatible, and a host upgrade reports it instead of guessing.
pub fn evaluate(compatibility: Option<&HostCompatibility>, host_version: &str) -> Verdict {
    let Some(compatibility) = compatibility else {
        return Verdict::Unknown("no recorded host compatibility".to_string());
    };
    if compatibility.min.is_none() && compatibility.max.is_none() {
        return Verdict::Unknown("empty host compatibility".to_string());
    }
    if let Some(min) = compatibility.min.as_deref() {
        match compare(host_version, min) {
            Ok(std::cmp::Ordering::Less) => {
                return Verdict::Incompatible(format!(
                    "requires MeshLLM {min} or newer (host is {host_version})"
                ));
            }
            Ok(_) => {}
            Err(error) => {
                return Verdict::Unknown(format!("malformed compatibility bound {min:?}: {error}"));
            }
        }
    }
    if let Some(max) = compatibility.max.as_deref() {
        match compare(host_version, max) {
            Ok(std::cmp::Ordering::Greater) => {
                return Verdict::Incompatible(format!(
                    "requires MeshLLM {max} or older (host is {host_version})"
                ));
            }
            Ok(_) => {}
            Err(error) => {
                return Verdict::Unknown(format!("malformed compatibility bound {max:?}: {error}"));
            }
        }
    }
    Verdict::Compatible
}

/// Whether `host_version` satisfies `bound` under the same rules as
/// [`evaluate`], for callers that need the answer without the verdict type.
pub fn satisfies(host_version: &str, bound: &str) -> bool {
    compare(host_version, bound).is_ok_and(|ordering| ordering != std::cmp::Ordering::Less)
}

/// Compare two host versions numerically. A `v` prefix is ignored, missing
/// components are zero, and anything after the third numeric component is
/// ignored for ordering. A version whose leading components are not numeric
/// cannot be compared.
fn compare(left: &str, right: &str) -> Result<std::cmp::Ordering> {
    let left = parse_bound(left)?;
    let right = parse_bound(right)?;
    Ok(left.cmp(&right))
}

fn parse_bound(version: &str) -> Result<(u64, u64, u64)> {
    let trimmed = version.trim().trim_start_matches('v');
    let mut parts = trimmed.split('.');
    let mut numbers = [0u64; 3];
    for slot in numbers.iter_mut() {
        match parts.next() {
            None => break,
            Some(component) => {
                let digits = component
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .count();
                if digits == 0 || digits != component.len() {
                    bail!("{version:?} is not a numeric host version");
                }
                *slot = component
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("{version:?}: {error}"))?;
            }
        }
    }
    Ok((numbers[0], numbers[1], numbers[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verdict(min: Option<&str>, max: Option<&str>, host: &str) -> Verdict {
        evaluate(
            Some(&HostCompatibility {
                min: min.map(str::to_string),
                max: max.map(str::to_string),
            }),
            host,
        )
    }

    #[test]
    fn bounds_include_the_host_version() {
        assert_eq!(verdict(Some("0.81.0"), None, "0.81.0"), Verdict::Compatible);
        assert_eq!(verdict(Some("0.80.0"), None, "0.81.0"), Verdict::Compatible);
        assert_eq!(
            verdict(Some("0.80.0"), Some("0.82.0"), "0.81.0"),
            Verdict::Compatible
        );
    }

    #[test]
    fn both_bounds_are_inclusive_at_equality() {
        assert_eq!(verdict(None, Some("0.81.0"), "0.81.0"), Verdict::Compatible);
        assert_eq!(
            verdict(Some("0.81.0"), Some("0.81.0"), "0.81.0"),
            Verdict::Compatible
        );
        assert!(matches!(
            verdict(None, Some("0.81.0"), "0.81.1"),
            Verdict::Incompatible(_)
        ));
    }

    #[test]
    fn a_host_older_than_the_min_bound_is_incompatible() {
        assert_eq!(
            verdict(Some("0.82.0"), None, "0.81.0"),
            Verdict::Incompatible("requires MeshLLM 0.82.0 or newer (host is 0.81.0)".into())
        );
    }

    #[test]
    fn a_host_newer_than_the_max_bound_is_incompatible() {
        assert_eq!(
            verdict(None, Some("0.81.0"), "0.82.0"),
            Verdict::Incompatible("requires MeshLLM 0.81.0 or older (host is 0.82.0)".into())
        );
    }

    #[test]
    fn v_prefixes_and_missing_components_compare_numerically() {
        assert_eq!(verdict(Some("v0.81"), None, "0.81.0"), Verdict::Compatible);
        assert_eq!(verdict(Some("0.81"), None, "v0.81.1"), Verdict::Compatible);
        assert!(
            matches!(
                verdict(Some("0.81.1"), None, "0.81"),
                Verdict::Incompatible(_)
            ),
            "0.81 is older than the 0.81.1 minimum"
        );
    }

    #[test]
    fn missing_or_malformed_metadata_is_unknown_not_incompatible() {
        assert_eq!(
            evaluate(None, "0.81.0"),
            Verdict::Unknown("no recorded host compatibility".into())
        );
        assert_eq!(
            evaluate(Some(&HostCompatibility::default()), "0.81.0"),
            Verdict::Unknown("empty host compatibility".into())
        );
        assert!(
            verdict(Some("not-a-version"), None, "0.81.0")
                .unknown_reason()
                .contains("malformed")
        );
        assert!(
            verdict(Some("0.81.0-beta"), None, "0.81.0")
                .unknown_reason()
                .contains("malformed")
        );
    }

    #[test]
    fn compatibility_json_round_trips_and_omits_absent_bounds() {
        let full: HostCompatibility =
            serde_json::from_str(r#"{"min":"0.81.0","max":"0.82.0"}"#).unwrap();
        assert_eq!(full.min.as_deref(), Some("0.81.0"));
        let empty: HostCompatibility = serde_json::from_str("{}").unwrap();
        assert_eq!(
            serde_json::to_string(&empty).unwrap(),
            "{}",
            "absent bounds are not written back"
        );
    }

    impl Verdict {
        fn unknown_reason(&self) -> String {
            match self {
                Verdict::Unknown(reason) => reason.clone(),
                other => panic!("{other:?}"),
            }
        }
    }
}

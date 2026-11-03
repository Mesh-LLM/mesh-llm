//! Verify a lane projection against the complete digest-bound canonical plan.
//!
//! A subset check cannot prove coverage: omitting an entire planned matrix
//! row can make its job skip while still leaving every retained row valid.

use super::{Checked, error};
use crate::DynResult;
use crate::repository::check_report::CheckReport;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;

const OPTIONS: [&str; 7] = [
    "--canonical-plan",
    "--canonical-plan-file",
    "--plan-digest",
    "--lane-plan",
    "--lane-plan-file",
    "--lane",
    "--source-sha",
];

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if matches!(args, [argument] if argument == "-h" || argument == "--help") {
        return CheckReport::success("usage: ci-controller ci validate-projection (--canonical-plan JSON | --canonical-plan-file FILE) --plan-digest SHA256 (--lane-plan JSON | --lane-plan-file FILE) --lane NAME --source-sha SHA\n".to_owned()).emit();
    }
    let result = parse_options(args).and_then(|options| {
        let canonical = input(&options, "--canonical-plan", "--canonical-plan-file")?;
        let lane_plan = input(&options, "--lane-plan", "--lane-plan-file")?;
        verify(
            &canonical,
            options["--plan-digest"],
            &lane_plan,
            options["--lane"],
            options["--source-sha"],
        )
    });
    match result {
        Ok(()) => CheckReport::default(),
        Err(message) => error(message),
    }
    .emit()
}

fn parse_options(args: &[String]) -> Checked<BTreeMap<&'static str, &str>> {
    let mut result = BTreeMap::new();
    let mut items = args.iter();
    while let Some(arg) = items.next() {
        let (name, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(name, value)| (name, Some(value)));
        let option = OPTIONS
            .iter()
            .find(|option| **option == name)
            .ok_or_else(|| format!("unknown projection option: {name}"))?;
        let value = match inline {
            Some(value) => value,
            None => items.next().map(String::as_str).unwrap_or_default(),
        };
        if value.is_empty() || value.starts_with("--") || result.insert(*option, value).is_some() {
            return Err(format!("{option} requires exactly one nonempty value"));
        }
    }
    for option in ["--plan-digest", "--lane", "--source-sha"] {
        if !result.contains_key(option) {
            return Err(format!("missing projection option {option}"));
        }
    }
    Ok(result)
}

fn input(options: &BTreeMap<&str, &str>, inline: &str, file: &str) -> Checked<String> {
    match (options.get(inline), options.get(file)) {
        (Some(value), None) => Ok((*value).to_owned()),
        (None, Some(path)) => {
            fs::read_to_string(path).map_err(|error| format!("cannot read {file}: {error}"))
        }
        _ => Err(format!("supply exactly one of {inline} and {file}")),
    }
}

pub(super) fn verify(
    canonical: &str,
    digest: &str,
    projection: &str,
    lane: &str,
    source_sha: &str,
) -> Checked<()> {
    if hex::encode(Sha256::digest(canonical.as_bytes())) != digest {
        return Err("plan digest does not match the canonical plan".to_owned());
    }
    if source_sha.len() != 40
        || !source_sha
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err("source SHA must be 40 lowercase hexadecimal characters".to_owned());
    }
    let plan = strict(canonical, "canonical plan")?;
    let actual = strict(projection, "lane plan")?;
    if plan.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("canonical plan schema_version must be 1".to_owned());
    }
    if plan.get("source_sha").and_then(Value::as_str) != Some(source_sha) {
        return Err("canonical plan does not match source SHA".to_owned());
    }
    let expected = expected_projection(&plan, lane)?;
    if actual != expected {
        return Err(format!(
            "lane plan does not match canonical {lane} projection"
        ));
    }
    Ok(())
}

/// `validate-lane` can bind to a canonical plan even when no separate source
/// SHA was supplied; in that mode verify the complete embedded projection.
pub(super) fn verify_embedded(canonical: &str, digest: &str, lane_plan: &str) -> Checked<()> {
    let plan = strict(canonical, "canonical plan")?;
    let projection = strict(lane_plan, "lane plan")?;
    let lane = projection
        .get("lane")
        .and_then(Value::as_str)
        .ok_or_else(|| "lane plan needs a lane".to_owned())?;
    let source = plan
        .get("source_sha")
        .and_then(Value::as_str)
        .ok_or_else(|| "canonical plan needs a source SHA".to_owned())?;
    verify(canonical, digest, lane_plan, lane, source)
}

fn rows<'a>(plan: &'a Value, name: &str) -> Checked<&'a [Value]> {
    plan.get("matrices")
        .and_then(|matrices| matrices.get(name))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("canonical plan matrix {name} must be an array"))
}

fn expected_projection(plan: &Value, lane: &str) -> Checked<Value> {
    let slices = plan
        .get("required_slices")
        .and_then(Value::as_array)
        .ok_or_else(|| "canonical plan required_slices must be an array".to_owned())?;
    let selected = |name: &str| slices.iter().any(|item| item.as_str() == Some(name));
    let platform_rows = |name: &str, platform: &str| -> Checked<Vec<Value>> {
        Ok(rows(plan, name)?
            .iter()
            .filter(|row| row.get("platform").and_then(Value::as_str) == Some(platform))
            .cloned()
            .collect())
    };
    let smoke = |metal: bool| -> Checked<Vec<Value>> {
        Ok(rows(plan, "smoke")?
            .iter()
            .filter(|row| {
                (row.get("id").and_then(Value::as_str) == Some("metal-model-load")) == metal
            })
            .cloned()
            .collect())
    };
    let (matrices, required) = match lane {
        "quality" => (
            json!({"clippy": rows(plan, "clippy")?}),
            selected("quality") || selected("runner-contract"),
        ),
        "website" => (json!({}), selected("web")),
        "linux" => {
            let rust = rows(plan, "rust_tests")?;
            let hosts = platform_rows("hosts", lane)?;
            let runtime = platform_rows("runtime_products", lane)?;
            let smoke = smoke(false)?;
            let sdk = platform_rows("sdk", lane)?;
            let required = !rust.is_empty()
                || !hosts.is_empty()
                || !runtime.is_empty()
                || !smoke.is_empty()
                || !sdk.is_empty()
                || selected("static-abi");
            (
                json!({"rust_tests": rust, "hosts": hosts, "runtime_products": runtime, "smoke": smoke, "sdk": sdk}),
                required,
            )
        }
        "macos" => {
            let hosts = platform_rows("hosts", lane)?;
            let runtime = platform_rows("runtime_products", lane)?;
            let checks = platform_rows("platform_checks", lane)?;
            let smoke = smoke(true)?;
            let sdk = platform_rows("sdk", lane)?;
            let required = !hosts.is_empty()
                || !runtime.is_empty()
                || !checks.is_empty()
                || !smoke.is_empty()
                || !sdk.is_empty();
            (
                json!({"hosts": hosts, "runtime_products": runtime, "platform_checks": checks, "smoke": smoke, "sdk": sdk}),
                required,
            )
        }
        "windows" => {
            let hosts = platform_rows("hosts", lane)?;
            let runtime = platform_rows("runtime_products", lane)?;
            let checks = platform_rows("platform_checks", lane)?;
            let smoke = if runtime.is_empty() {
                vec![]
            } else {
                rows(plan, "smoke")?
                    .iter()
                    .filter(|row| row.get("id").and_then(Value::as_str) == Some("core"))
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let required =
                !hosts.is_empty() || !runtime.is_empty() || !checks.is_empty() || !smoke.is_empty();
            (
                json!({"hosts": hosts, "runtime_products": runtime, "platform_checks": checks, "smoke": smoke}),
                required,
            )
        }
        _ => return Err(format!("unknown CI lane {lane}")),
    };
    let mut expected = Map::new();
    for field in [
        "profile",
        "domains",
        "required_slices",
        "signals",
        "budgets",
    ] {
        expected.insert(
            field.to_owned(),
            plan.get(field)
                .ok_or_else(|| format!("canonical plan missing {field}"))?
                .clone(),
        );
    }
    expected.insert("lane".to_owned(), json!(lane));
    expected.insert("required".to_owned(), json!(required));
    expected.insert("matrices".to_owned(), matrices);
    Ok(Value::Object(expected))
}

/// JSON with duplicate object keys rejected recursively, including escaped keys.
struct Strict(Value);

fn strict(value: &str, label: &str) -> Checked<Value> {
    serde_json::from_str::<Strict>(value)
        .map(|value| value.0)
        .map_err(|error| format!("{label} is invalid or ambiguous JSON: {error}"))
}

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = Strict;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_unit<E: de::Error>(self) -> Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Strict, E> {
                Ok(Strict(json!(value)))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Strict, E> {
                Ok(Strict(json!(value)))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Strict, E> {
                Ok(Strict(json!(value)))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Strict, E> {
                Number::from_f64(value)
                    .map(|number| Strict(Value::Number(number)))
                    .ok_or_else(|| E::custom("non-finite number"))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Strict, E> {
                Ok(Strict(json!(value)))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Strict, E> {
                Ok(Strict(json!(value)))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Strict, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<Strict>()? {
                    values.push(value.0);
                }
                Ok(Strict(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Strict, A::Error> {
                let mut values = Map::new();
                while let Some((key, value)) = map.next_entry::<String, Strict>()? {
                    if values.insert(key, value.0).is_some() {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(Strict(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn canonical() -> Value {
        json!({
            "schema_version": 1,
            "source_sha": SOURCE,
            "profile": "main",
            "domains": ["runtime"],
            "required_slices": ["quality", "web", "static-abi"],
            "signals": {"rust_changed": true},
            "budgets": {"total_max_workers": 8},
            "matrices": {
                "clippy": [{"id": "quality"}],
                "rust_tests": [{"id": "tests"}],
                "hosts": [{"id": "linux-host", "platform": "linux"}, {"id": "windows-host", "platform": "windows"}],
                "runtime_products": [{"id": "linux-cpu", "platform": "linux"}, {"id": "windows-cpu", "platform": "windows"}],
                "platform_checks": [{"id": "mac-unit", "platform": "macos"}],
                "smoke": [{"id": "core"}, {"id": "metal-model-load"}],
                "sdk": [{"id": "swift", "platform": "macos"}]
            }
        })
    }

    fn digest(canonical: &str) -> String {
        hex::encode(Sha256::digest(canonical.as_bytes()))
    }

    #[test]
    fn complete_windows_projection_matches_and_omitted_runtime_does_not() {
        let plan = canonical();
        let serialized = plan.to_string();
        let original = json!({
            "lane": "windows", "required": true, "profile": "main",
            "domains": ["runtime"], "required_slices": ["quality", "web", "static-abi"],
            "signals": {"rust_changed": true}, "budgets": {"total_max_workers": 8},
            "matrices": {
                "hosts": [{"id": "windows-host", "platform": "windows"}],
                "runtime_products": [{"id": "windows-cpu", "platform": "windows"}],
                "platform_checks": [], "smoke": [{"id": "core"}]
            }
        });
        let call = |lane: &Value| {
            verify(
                &serialized,
                &digest(&serialized),
                &lane.to_string(),
                "windows",
                SOURCE,
            )
        };
        assert_eq!(call(&original), Ok(()));
        let mut omitted = original.clone();
        omitted["matrices"]["runtime_products"] = json!([]);
        assert!(call(&omitted).is_err());
        let mut skipped = original.clone();
        skipped["required"] = json!(false);
        assert!(call(&skipped).is_err());
        let mut extra = original.clone();
        extra["matrices"]["hosts"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"invented"}));
        assert!(call(&extra).is_err());
    }

    #[test]
    fn all_lane_selections_come_from_the_canonical_fields() {
        let plan = canonical();
        let expected = [
            ("quality", true, 1),
            ("website", true, 0),
            ("linux", true, 5),
            ("macos", true, 5),
            ("windows", true, 4),
        ];
        for (lane, required, matrix_count) in expected {
            let projection = expected_projection(&plan, lane).unwrap();
            assert_eq!(projection["lane"], lane);
            assert_eq!(projection["required"], required);
            assert_eq!(
                projection["matrices"].as_object().unwrap().len(),
                matrix_count
            );
        }
        assert_eq!(
            expected_projection(&plan, "macos").unwrap()["matrices"]["smoke"],
            json!([{"id":"metal-model-load"}])
        );
        assert_eq!(
            expected_projection(&plan, "linux").unwrap()["matrices"]["smoke"],
            json!([{"id":"core"}])
        );
        let mut no_windows = plan.clone();
        no_windows["matrices"]["runtime_products"] = json!([]);
        assert_eq!(
            expected_projection(&no_windows, "windows").unwrap()["matrices"]["smoke"],
            json!([])
        );
    }

    #[test]
    fn identity_digest_and_duplicate_keys_fail_closed() {
        let plan = canonical().to_string();
        let lane = expected_projection(&canonical(), "website")
            .unwrap()
            .to_string();
        assert!(verify(&plan, &digest(&plan), &lane, "website", SOURCE).is_ok());
        assert!(verify(&plan, &"0".repeat(64), &lane, "website", SOURCE).is_err());
        assert!(verify(&plan, &digest(&plan), &lane, "website", &"b".repeat(40)).is_err());
        assert!(verify(&plan, &digest(&plan), &lane, "linux", SOURCE).is_err());
        let duplicate = lane.replacen(
            "\"required\":true",
            "\"required\":false,\"requ\\u0069red\":true",
            1,
        );
        assert_ne!(duplicate, lane);
        assert!(verify(&plan, &digest(&plan), &duplicate, "website", SOURCE).is_err());
        let duplicate_plan = plan.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        );
        assert_ne!(duplicate_plan, plan);
        assert!(
            verify(
                &duplicate_plan,
                &digest(&duplicate_plan),
                &lane,
                "website",
                SOURCE
            )
            .is_err()
        );
    }
}

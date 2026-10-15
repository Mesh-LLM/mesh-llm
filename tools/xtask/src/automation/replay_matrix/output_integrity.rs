//! Opt-in structural completeness gate for sustained correctness replays.
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn incomplete(record: &Value) -> bool {
    record.get("error").is_some()
        || record["output_problems"]
            .as_array()
            .is_none_or(|problems| !problems.is_empty())
}

pub(super) fn apply(document: &mut Value) -> DynResult<bool> {
    let checks = document["results"]
        .as_array()
        .ok_or("missing replay results")?
        .iter()
        .flat_map(|result| std::iter::once(&result["warmup"]).chain(result["cells"].as_array().into_iter().flatten()))
        .map(|cell| {
            let incomplete = cell["output_incomplete_requests"].as_u64();
            let passed = incomplete == Some(0)
                && cell["requests"].as_u64().is_some_and(|count| count > 0);
            json!({"name":format!("complete-output:c{}",cell["concurrency"]),
                "passed":passed,"detail":format!("incomplete={incomplete:?} required=0; missing evidence fails")})
        })
        .collect::<Vec<_>>();
    let passed = !checks.is_empty() && checks.iter().all(|check| check["passed"] == true);
    let gates = &mut document["gates"];
    gates["evaluated"] = true.into();
    gates["checks"]
        .as_array_mut()
        .ok_or("missing replay gate checks")?
        .extend(checks);
    gates["passed"] = (passed && gates["passed"] != false).into();
    Ok(passed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_output_requires_every_measured_cell_to_have_complete_evidence() {
        for (count, requests, expected) in [
            (json!(0), 3, true),
            (json!(1), 3, false),
            (Value::Null, 3, false),
            (json!(0), 0, false),
        ] {
            let mut document = json!({"results":[{"warmup":{"requests":1,"output_incomplete_requests":0},"cells":[{"concurrency":1,
                "requests":requests,"output_incomplete_requests":count}]}],
                "gates":{"passed":null,"checks":[]}});
            assert_eq!(apply(&mut document).unwrap(), expected);
            assert_eq!(document["gates"]["passed"], expected);
        }
        let mut empty = json!({"results":[],"gates":{"checks":[],"passed":null}});
        assert!(!apply(&mut empty).unwrap());
    }

    #[test]
    fn completeness_does_not_erase_another_failed_gate() {
        let mut document = json!({"results":[{"warmup":{"requests":1,"output_incomplete_requests":0},"cells":[{"requests":3,
            "output_incomplete_requests":0}]}],"gates":{"checks":[],"passed":false}});
        assert!(apply(&mut document).unwrap());
        assert_eq!(document["gates"]["passed"], false);
    }
}

#[cfg(test)]
mod warmup_tests {
    use super::*;
    #[test]
    fn truncated_or_missing_warmup_fails_even_when_every_measured_output_passed() {
        for warmup in [
            Value::Null,
            json!({"requests":4,"output_incomplete_requests":1}),
        ] {
            let mut document = json!({"results":[{"warmup":warmup,"cells":[{"requests":50,"output_incomplete_requests":0}]}],"gates":{"checks":[],"passed":true}});
            assert!(!apply(&mut document).unwrap());
            assert_eq!(document["gates"]["passed"], false);
        }
    }
}

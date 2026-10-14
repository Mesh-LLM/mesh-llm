//! Opt-in structural completeness gate for sustained correctness replays.
use crate::command::DynResult;
use serde_json::{Value, json};

pub(super) fn apply(document: &mut Value) -> DynResult<bool> {
    let checks = document["results"]
        .as_array()
        .ok_or("missing replay results")?
        .iter()
        .flat_map(|result| result["cells"].as_array().into_iter().flatten())
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
            let mut document = json!({"results":[{"cells":[{"concurrency":1,
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
        let mut document = json!({"results":[{"cells":[{"requests":3,
            "output_incomplete_requests":0}]}],"gates":{"checks":[],"passed":false}});
        assert!(apply(&mut document).unwrap());
        assert_eq!(document["gates"]["passed"], false);
    }
}

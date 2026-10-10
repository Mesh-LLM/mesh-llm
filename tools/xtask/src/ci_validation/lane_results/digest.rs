//! Binds a lane projection to the digest-bound canonical plan without
//! changing either schema: the digest is the SHA-256 of the exact `plan_json`
//! bytes (as `plan-ci/action.yml` and `ci-control.yml` compute it), and the
//! projection's shared fields and matrix rows must come from that plan.

use super::Checked;
use crate::ci_plan::document::Json;
use sha2::{Digest, Sha256};

const SHARED: [&str; 5] = [
    "profile",
    "domains",
    "required_slices",
    "signals",
    "budgets",
];

pub(super) fn verify(lane_plan: &Json, digest: &str, canonical: &str) -> Checked<()> {
    if hex::encode(Sha256::digest(canonical.as_bytes())) != digest {
        return Err("plan digest does not match the canonical plan".to_owned());
    }
    let plan = Json::parse(canonical.as_bytes())
        .map_err(|error| format!("canonical plan is not valid JSON: {error}"))?;
    if plan
        .get("schema_version")
        .is_none_or(|version| !version.equals_one())
    {
        return Err("canonical plan schema_version must be 1".to_owned());
    }
    for field in SHARED {
        if lane_plan.get(field) != plan.get(field) {
            return Err(format!(
                "lane plan {field} does not match the canonical plan"
            ));
        }
    }
    let matrices = lane_plan
        .get("matrices")
        .map(Json::as_object)
        .unwrap_or_default()
        .unwrap_or_default();
    for (matrix, rows) in matrices {
        let source = plan
            .get("matrices")
            .and_then(|all| all.get(matrix))
            .and_then(Json::as_array)
            .unwrap_or_default();
        for row in rows.as_array().unwrap_or_default() {
            if !source.contains(row) {
                let id = row.get("id").and_then(Json::as_str).unwrap_or("");
                return Err(format!(
                    "lane plan matrix {matrix} row '{id}' is not in the canonical plan"
                ));
            }
        }
    }
    Ok(())
}

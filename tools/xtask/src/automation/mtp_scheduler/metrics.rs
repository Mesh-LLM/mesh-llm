use crate::command::DynResult;
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn count(v: &Value, k: &str) -> DynResult<u64> {
    v[k].as_u64()
        .ok_or_else(|| format!("invalid client {k}").into())
}
fn finite(v: &Value, k: &str) -> DynResult<f64> {
    v[k].as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or_else(|| format!("invalid client {k}").into())
}
pub(super) fn summarize(v: &Value, requests: usize, concurrency: usize) -> DynResult<Value> {
    if count(v, "requests")? != requests as u64 || count(v, "concurrency")? != concurrency as u64 {
        return Err("client request/concurrency mismatch".into());
    }
    let rows = v["per_request"].as_array().ok_or("missing per_request")?;
    if rows.len() != requests {
        return Err("incomplete client request roster".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut elapsed = Vec::new();
    let mut drafted = 0;
    let mut accepted = 0;
    for row in rows {
        if let Some((latency, has_draft, is_accepted)) = request(row, requests, &mut ids)? {
            elapsed.push(latency);
            drafted += u64::from(has_draft);
            accepted += u64::from(is_accepted);
        }
    }
    let successful = elapsed.len() as u64;
    let failed = requests as u64 - successful;
    for (key, actual) in [
        ("successful", successful),
        ("failed", failed),
        ("drafted", drafted),
        ("accepted", accepted),
    ] {
        if count(v, key)? != actual {
            return Err("client aggregate/request mismatch".into());
        }
    }
    let makespan = finite(v, "makespan_ms")?;
    if makespan == 0.0 {
        return Err("zero client makespan".into());
    }
    let throughput = finite(v, "throughput_rps")?;
    let acceptance = finite(v, "acceptance_rate")?;
    let expected = if drafted == 0 {
        0.0
    } else {
        accepted as f64 / drafted as f64
    };
    if (acceptance - expected).abs() > 1e-9
        || (throughput - successful as f64 / (makespan / 1000.0)).abs() > 1e-6 * throughput.max(1.0)
    {
        return Err("client rate/request mismatch".into());
    }
    elapsed.sort_by(f64::total_cmp);
    let median = (!elapsed.is_empty()).then(|| {
        let m = elapsed.len() / 2;
        if elapsed.len() % 2 == 0 {
            elapsed[m - 1] / 2.0 + elapsed[m] / 2.0
        } else {
            elapsed[m]
        }
    });
    let p99 = (!elapsed.is_empty()).then(|| elapsed[(elapsed.len() * 99).div_ceil(100) - 1]);
    Ok(
        json!({"requests":requests,"successful":successful,"failed":failed,"throughput_rps":throughput,"latency_p50_ms":median,"latency_p99_ms":p99,"drafted":drafted,"accepted":accepted,"acceptance_rate":acceptance}),
    )
}
fn successful(row: &Value) -> DynResult<BTreeMap<u64, &Value>> {
    let mut result = BTreeMap::new();
    for request in row["per_request"]
        .as_array()
        .ok_or("missing parity roster")?
    {
        if request.get("error").is_none() {
            result.insert(count(request, "request_id")?, request);
        }
    }
    Ok(result)
}
pub(super) fn parity(old: &Value, new: &Value) -> DynResult<Value> {
    let old = old["concurrency_sweep"]
        .as_array()
        .ok_or("missing old sweep")?;
    let new = new["concurrency_sweep"]
        .as_array()
        .ok_or("missing new sweep")?;
    if old.len() != new.len() {
        return Err("incomplete old/new sweep".into());
    }
    let mut output = Vec::new();
    for (a, b) in old.iter().zip(new) {
        if a["concurrency"] != b["concurrency"] {
            return Err("old/new sweep ordering mismatch".into());
        }
        let arows = successful(a)?;
        let brows = successful(b)?;
        let mut comparable = 0;
        let mut exact = 0;
        let mut fields = 0;
        for (id, a) in &arows {
            if let Some(b) = brows.get(id) {
                comparable += 1;
                let matched = ["predicted", "draft", "verified", "accepted"]
                    .into_iter()
                    .filter(|key| a[*key] == b[*key])
                    .count();
                fields += matched;
                if matched == 4 {
                    exact += 1;
                }
            }
        }
        output.push(json!({"concurrency":a["concurrency"],"comparable_requests":comparable,"exact_requests":exact,"exact_field_matches":fields,"exact_field_total":comparable*4,"old_successful":arows.len(),"new_successful":brows.len(),"complete_request_coverage":comparable==arows.len()&&comparable==brows.len()&&arows.len()==a["metrics"]["requests"].as_u64().unwrap_or(0) as usize}));
    }
    Ok(json!(output))
}

fn request(
    row: &Value,
    requests: usize,
    ids: &mut std::collections::BTreeSet<u64>,
) -> DynResult<Option<(f64, bool, bool)>> {
    let id = count(row, "request_id")?;
    if id == 0 || id > requests as u64 || !ids.insert(id) {
        return Err("duplicate/out-of-range request identity".into());
    }
    if let Some(error) = row.get("error") {
        if error.as_str().is_none_or(str::is_empty) {
            return Err("invalid request error".into());
        }
        return Ok(None);
    }
    if ["predicted", "draft", "verified", "accepted"]
        .iter()
        .any(|key| row.get(*key).is_none())
    {
        return Err("missing client parity field".into());
    }
    match (
        row["draft"].as_i64(),
        row["verified"].as_i64(),
        row["accepted"].as_bool(),
    ) {
        (None, None, None)
            if row["draft"].is_null() && row["verified"].is_null() && row["accepted"].is_null() => {
        }
        (Some(draft), Some(verified), Some(accepted)) if accepted == (draft == verified) => {}
        _ => return Err("inconsistent client verification fields".into()),
    }
    let elapsed = finite(row, "elapsed_ms")?;

    if row["predicted"].as_i64().is_none() {
        return Err("missing predicted token".into());
    }
    Ok(Some((
        elapsed,
        !row["draft"].is_null(),
        row["accepted"] == true,
    )))
}

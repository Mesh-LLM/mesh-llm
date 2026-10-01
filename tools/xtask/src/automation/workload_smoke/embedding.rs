use crate::command::DynResult;
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Response<T> {
    object: String,
    model: String,
    data: Vec<Row<T>>,
    usage: Option<Usage>,
}
#[derive(Deserialize)]
struct Row<T> {
    object: String,
    index: usize,
    embedding: T,
}
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
}

fn envelope<T>(response: &Response<T>, model: &str, count: usize) -> DynResult<()> {
    if response.object != "list"
        || response.model != model
        || response.data.len() != count
        || response
            .data
            .iter()
            .enumerate()
            .any(|(index, row)| row.object != "embedding" || row.index != index)
    {
        return Err("embedding envelope, cardinality or indexes do not match request".into());
    }
    Ok(())
}

pub(super) fn validate(numeric: &[u8], encoded: &[u8], model: &str) -> DynResult<()> {
    let numeric: Response<Vec<f64>> = serde_json::from_slice(numeric)?;
    envelope(&numeric, model, 3)?;
    let first = &numeric.data[0].embedding;
    if first.is_empty() || numeric.usage.is_none_or(|usage| usage.prompt_tokens == 0) {
        return Err("embedding vector or prompt usage is empty".into());
    }
    for row in &numeric.data {
        let vector = &row.embedding;
        let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if vector.len() != first.len()
            || vector.iter().any(|value| !value.is_finite())
            || (norm - 1.0).abs() > 1e-4
        {
            return Err("embedding dimensions, finite values or normalization failed".into());
        }
    }
    let similarity = |vector: &[f64]| {
        first
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum::<f64>()
    };
    if similarity(&numeric.data[1].embedding) <= similarity(&numeric.data[2].embedding) {
        return Err("embedding semantic ordering failed".into());
    }
    let encoded: Response<String> = serde_json::from_slice(encoded)?;
    envelope(&encoded, model, 1)?;
    let raw = super::encoding::decode(&encoded.data[0].embedding)?;
    if raw.len()
        != first
            .len()
            .checked_mul(4)
            .ok_or("embedding length overflow")?
    {
        return Err("base64 embedding byte length differs".into());
    }
    for (chunk, expected) in raw.as_chunks::<4>().0.iter().zip(first) {
        let value = f64::from(f32::from_le_bytes(*chunk));
        if !value.is_finite()
            || (value - expected).abs() > 1e-6_f64.max(1e-5 * value.abs().max(expected.abs()))
        {
            return Err("base64 embedding differs from float response".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn numeric() -> serde_json::Value {
        json!({"object":"list","model":"fixture","usage":{"prompt_tokens":3},"data":[
            {"object":"embedding","index":0,"embedding":[1,0]},
            {"object":"embedding","index":1,"embedding":[1,0]},
            {"object":"embedding","index":2,"embedding":[0,1]}]})
    }
    fn encoded() -> serde_json::Value {
        json!({"object":"list","model":"fixture","data":[{"object":"embedding","index":0,"embedding":"AACAPwAAAAA="}]})
    }
    fn check(numeric: &serde_json::Value, encoded: &serde_json::Value) -> DynResult<()> {
        validate(
            &serde_json::to_vec(numeric)?,
            &serde_json::to_vec(encoded)?,
            "fixture",
        )
    }
    #[test]
    fn valid_float_and_base64_when_metadata_matches() {
        assert!(check(&numeric(), &encoded()).is_ok());
    }
    #[test]
    fn rejects_invalid_envelopes_and_cardinality() {
        for (field, value) in [
            ("object", json!("embedding")),
            ("model", json!("another")),
            ("data", json!([])),
            ("data", json!([null])),
            ("data", json!(null)),
        ] {
            let mut response = encoded();
            response[field] = value;
            assert!(check(&numeric(), &response).is_err());
        }
        let mut response = encoded();
        response["data"] = json!([response["data"][0], response["data"][0]]);
        assert!(check(&numeric(), &response).is_err());
    }
    #[test]
    fn rejects_noninteger_indexes_and_boolean_components() {
        for index in [
            json!(false),
            json!(0.0),
            json!("0"),
            json!(-1),
            json!(1),
            json!(null),
        ] {
            let mut response = encoded();
            response["data"][0]["index"] = index;
            assert!(check(&numeric(), &response).is_err());
        }
        let mut response = numeric();
        response["data"][0]["embedding"] = json!([true, 0]);
        assert!(check(&response, &encoded()).is_err());
    }
    #[test]
    fn rejects_wrong_base64_values_lengths_and_nonfinite_values() {
        for payload in [
            "AAAAAAAAgD8=",
            "AACAPw==",
            "AADAfwAAAAA=",
            "invalid",
            "AACAPwAAAAA!",
        ] {
            let mut response = encoded();
            response["data"][0]["embedding"] = json!(payload);
            assert!(check(&numeric(), &response).is_err());
        }
    }
    #[test]
    fn allows_float32_rounding() {
        let mut response = encoded();
        let bytes: Vec<_> = [1.0000001_f32, 1e-7_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        response["data"][0]["embedding"] = json!(super::super::encoding::encode(&bytes));
        assert!(check(&numeric(), &response).is_ok());
    }
}

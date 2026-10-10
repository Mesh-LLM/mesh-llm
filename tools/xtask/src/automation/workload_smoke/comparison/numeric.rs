use crate::command::DynResult;
use serde::Deserialize;

#[derive(Deserialize)]
struct Embeddings {
    data: Vec<Embedding>,
}
#[derive(Deserialize)]
struct Embedding {
    index: usize,
    embedding: Vec<f64>,
}

fn vectors(bytes: &[u8], count: usize) -> DynResult<Vec<Vec<f64>>> {
    let response: Embeddings = serde_json::from_slice(bytes)?;
    if response.data.len() != count {
        return Err("embedding oracle response has the wrong batch size".into());
    }
    response
        .data
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            if row.index != index {
                return Err("embedding oracle response has invalid indexes".into());
            }
            if row.embedding.is_empty() || row.embedding.iter().any(|n| !n.is_finite()) {
                return Err("embedding oracle response requires nonempty finite vectors".into());
            }
            Ok(row.embedding)
        })
        .collect()
}

fn unit(vector: &[f64]) -> DynResult<Vec<f64>> {
    let scale = vector.iter().map(|n| n.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err("embedding oracle response has a zero vector".into());
    }
    let norm = vector
        .iter()
        .map(|n| (n / scale).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(vector.iter().map(|n| (n / scale) / norm).collect())
}

pub(super) fn embeddings(candidate: &[u8], reference: &[u8], count: usize) -> DynResult<String> {
    let left = vectors(candidate, count)?;
    let right = vectors(reference, count)?;
    let mut max_delta = 0.0_f64;
    let mut min_cosine = 1.0_f64;
    for (a, b) in left.iter().zip(&right) {
        if a.len() != b.len() {
            return Err("embedding dimensions differ from monolithic reference".into());
        }
        let delta = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let cosine = unit(a)?
            .iter()
            .zip(unit(b)?)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        if !delta.is_finite() || !cosine.is_finite() {
            return Err("embedding comparison is not finite".into());
        }
        max_delta = max_delta.max(delta);
        min_cosine = min_cosine.min(cosine);
    }
    if max_delta > 1e-4 || min_cosine < 0.99999 {
        return Err(format!("embedding differs from monolithic reference: max_abs_delta={max_delta}, min_cosine={min_cosine}").into());
    }
    Ok(format!(
        "max_abs_delta={max_delta}, min_cosine={min_cosine}"
    ))
}

#[derive(Deserialize)]
struct Reranks {
    results: Vec<Rank>,
}
#[derive(Deserialize)]
struct Rank {
    index: usize,
    relevance_score: f64,
}
fn ranks(bytes: &[u8]) -> DynResult<Vec<Rank>> {
    let rows = serde_json::from_slice::<Reranks>(bytes)?.results;
    if rows.len() != 2
        || rows
            .iter()
            .any(|r| r.index >= 2 || !r.relevance_score.is_finite())
        || rows[0].index == rows[1].index
    {
        return Err("rerank oracle response has invalid cardinality, indexes or scores".into());
    }
    Ok(rows)
}
pub(super) fn rerank(candidate: &[u8], reference: &[u8]) -> DynResult<String> {
    let left = ranks(candidate)?;
    let right = ranks(reference)?;
    let mut delta = 0.0_f64;
    for (a, b) in left.iter().zip(&right) {
        if a.index != b.index {
            return Err("rerank differs from monolithic reference: wire order".into());
        }
        delta = delta.max((a.relevance_score - b.relevance_score).abs());
    }
    if !delta.is_finite() || delta > 1e-4 {
        return Err("rerank differs from monolithic reference: scores".into());
    }
    Ok(format!(
        "max_abs_delta={delta}, order={:?}",
        left.iter().map(|r| r.index).collect::<Vec<_>>()
    ))
}

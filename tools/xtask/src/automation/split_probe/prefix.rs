use crate::command::DynResult;
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, thiserror::Error)]
#[error("all follow-up requests are cold")]
pub(super) struct Cold;

pub(super) fn payloads(model: &str, output: &Path, nonce: &str) -> DynResult<()> {
    let mut prompt = format!(
        "Split prefix cache smoke shared context {nonce}. {}",
        "Every request keeps these tokens in the same order. ".repeat(48)
    );
    for (pair, extension) in [
        "First extension block remains reusable by later prompts. ",
        "Second extension block makes the reusable prefix longer. ",
        "Third extension block proves reuse keeps growing. ",
    ]
    .iter()
    .enumerate()
    {
        prompt.push_str(&extension.repeat(16));
        for repeat in 0..2 {
            let payload = serde_json::json!({"model":model,"messages":[{"role":"user","content":prompt}],"user":format!("ci-split-prefix-growth-{nonce}"),"stream":false,"max_tokens":1,"temperature":0});
            fs::write(
                output.join(format!("prompt-{}.json", pair * 2 + repeat + 1)),
                serde_json::to_vec(&payload)?,
            )?;
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Response {
    object: String,
    choices: Vec<serde_json::Value>,
    usage: Usage,
}
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Details,
}
#[derive(Default, Deserialize)]
struct Details {
    #[serde(default)]
    cached_tokens: u64,
}

fn validate(metrics: &[(u64, u64)], kind: &str) -> DynResult<()> {
    if metrics.len() != 6 {
        return Err("prefix probe requires six responses".into());
    }
    if metrics[0].1 != 0
        || !metrics
            .as_chunks::<2>()
            .0
            .iter()
            .all(|pair| pair[0].0 == pair[1].0)
        || !(metrics[0].0 < metrics[2].0 && metrics[2].0 < metrics[4].0)
    {
        return Err("cold, repeated or growing prompt counts failed".into());
    }
    if metrics[1..].iter().all(|(_, cached)| *cached == 0) {
        return Err(Cold.into());
    }
    let recurrent = kind == "kv-recurrent";
    let indexes = if recurrent { [1, 3, 5] } else { [0, 2, 4] };
    if !(metrics[indexes[0]].1 < metrics[indexes[1]].1
        && metrics[indexes[1]].1 < metrics[indexes[2]].1)
    {
        return Err("cache reuse did not grow".into());
    }
    // Resident capacity pressure can evict the deepest checkpoint, so a valid
    // repeat may leave a suffix uncached. Output equality in `verify` proves the
    // restored prefix still leads to the same continuation as the uncached request.
    for pair in metrics.as_chunks::<2>().0 {
        if pair[0].1 >= pair[0].0 || pair[1].1 <= pair[0].1 {
            return Err("growth suffix or repeated restore failed".into());
        }
    }
    Ok(())
}

fn continuation(response: &Response, index: usize) -> DynResult<String> {
    response
        .choices
        .first()
        .and_then(|choice| choice.pointer("/message/content"))
        .and_then(serde_json::Value::as_str)
        .filter(|content| !content.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("prefix request {index} returned no assistant continuation").into())
}

fn same_continuations(outputs: &[String]) -> DynResult<()> {
    for (pair, outputs) in outputs.as_chunks::<2>().0.iter().enumerate() {
        if outputs[0] != outputs[1] {
            let (first, repeat) = (pair * 2 + 1, pair * 2 + 2);
            return Err(format!(
                "warm request {repeat} diverged from uncached request {first}: {:?} != {:?}",
                outputs[0], outputs[1]
            )
            .into());
        }
    }
    Ok(())
}

pub(super) fn verify(directory: &Path, count: usize, kind: &str) -> DynResult<String> {
    let mut metrics = Vec::new();
    let mut outputs = Vec::new();
    for index in 1..=count {
        let response: Response =
            serde_json::from_slice(&fs::read(directory.join(format!("response-{index}.json")))?)?;
        if response.object != "chat.completion" || response.choices.is_empty() {
            return Err("prefix response is not a chat completion".into());
        }
        outputs.push(continuation(&response, index)?);
        metrics.push((
            response.usage.prompt_tokens,
            response.usage.prompt_tokens_details.cached_tokens,
        ));
    }
    validate(&metrics, kind)?;
    same_continuations(&outputs)?;
    Ok(format!(
        "Split prefix cache reuse grew and repeated prompts restored from cache: {}\n",
        metrics
            .iter()
            .enumerate()
            .map(|(index, (prompt, cached))| format!(
                "request {}: prompt_tokens={prompt}, cached_tokens={cached}",
                index + 1
            ))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_accepts_partial_repeats_but_requires_growing_prefix() {
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 99),
                    (200, 90),
                    (200, 199),
                    (300, 190),
                    (300, 299)
                ],
                "kv-dense"
            )
            .is_ok()
        );
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 10),
                    (200, 90),
                    (200, 100),
                    (300, 190),
                    (300, 200)
                ],
                "kv-dense"
            )
            .is_ok()
        );
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 10),
                    (200, 90),
                    (200, 100),
                    (300, 90),
                    (300, 100)
                ],
                "kv-dense"
            )
            .is_err()
        );
    }
    #[test]
    fn recurrent_accepts_checkpoint_aligned_growth_but_requires_repeats() {
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 32),
                    (200, 0),
                    (200, 64),
                    (300, 0),
                    (300, 96)
                ],
                "kv-recurrent"
            )
            .is_ok()
        );
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 32),
                    (200, 0),
                    (200, 32),
                    (300, 0),
                    (300, 32)
                ],
                "kv-recurrent"
            )
            .is_err()
        );
    }
    #[test]
    fn all_cold_is_transient_but_partial_miss_is_failure() {
        let error = validate(
            &[(100, 0), (100, 0), (200, 0), (200, 0), (300, 0), (300, 0)],
            "kv-dense",
        )
        .unwrap_err();
        assert!(error.downcast_ref::<Cold>().is_some());
        assert!(
            validate(
                &[
                    (100, 0),
                    (100, 99),
                    (200, 0),
                    (200, 0),
                    (300, 190),
                    (300, 299)
                ],
                "kv-dense"
            )
            .is_err()
        );
    }
}

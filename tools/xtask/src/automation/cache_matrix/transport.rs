//! Cache observations reuse the existing bounded JSON transport and protocol decoders.
use super::{Options, observation::Observation};
use crate::{
    automation::{guardrail_corpus::transport, openai_exchange},
    process::Cancellation,
};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

async fn cancelled(token: &Cancellation) {
    while !token.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn exchange(
    base: &str,
    suffix: &str,
    body: &Value,
    authorization: Option<&str>,
    deadline: Instant,
    token: &Cancellation,
) -> Result<transport::Response, String> {
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("cache matrix cancellation/deadline before request".into());
    }
    let url = url::Url::parse(base).map_err(|_| "invalid admitted endpoint")?;
    let curl_owned = url.scheme() == "https" || !matches!(url.host(), Some(url::Host::Ipv4(_)));
    let request = transport::exchange_owned_with_authorization(
        base,
        suffix,
        Some(body),
        deadline,
        token,
        authorization,
    );
    let response = if curl_owned {
        // The supervised curl owns deadline/cancellation and its reserved cleanup.
        // Await it to completion instead of dropping a spawn_blocking future.
        request
            .await
            .map_err(|_| "cache matrix bounded curl request refused")?
    } else {
        tokio::select! {
            biased;
            _=cancelled(token)=>return Err("cache matrix cancelled".into()),
            result=tokio::time::timeout_at(deadline.into(),request)=>
                result.map_err(|_| "cache matrix request deadline")?
                    .map_err(|_| "cache matrix HTTP request refused")?,
        }
    };
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("cache matrix cancellation/deadline after owned cleanup".into());
    }
    Ok(response)
}
pub(super) async fn observe(
    opts: &Options,
    index: usize,
    tail: &str,
    run: Option<u64>,
    token: &Cancellation,
    deadline: Instant,
) -> Result<Observation, String> {
    let skippy = index % 2 == 1;
    let enabled = index >= 2;
    let body = if skippy {
        json!({"model":opts.model,"messages":[{"role":"system","content":opts.prefix},
            {"role":"user","content":tail}],"temperature":0,"top_p":1,"max_tokens":opts.max_tokens})
    } else {
        let prompt = format!("System prefix:\n{}\n\nUser request:\n{tail}\n", opts.prefix);
        json!({"prompt":prompt,"n_predict":opts.max_tokens,"temperature":0,"top_k":1,
            "cache_prompt":enabled,"stream":false})
    };
    let started = Instant::now();
    let response = exchange(
        &opts.urls[index],
        if skippy {
            "chat/completions"
        } else {
            "completion"
        },
        &body,
        if skippy {
            opts.api_key.as_deref()
        } else {
            None
        },
        deadline,
        token,
    )
    .await?;
    let elapsed = started.elapsed();
    let (observed_elapsed, prompt, cached, hash) = project(
        skippy,
        response.status,
        &response.body,
        opts.max_tokens,
        elapsed,
    )?;
    Ok(Observation::new(
        run,
        observed_elapsed,
        prompt,
        cached,
        skippy,
        enabled,
        hash,
    ))
}
fn project(
    skippy: bool,
    status: u16,
    body: &[u8],
    tokens: u64,
    elapsed: Duration,
) -> Result<(f64, u64, Option<u64>, String), String> {
    if skippy {
        if !(200..300).contains(&status) {
            return Err("OpenAI nonstream HTTP status refused".into());
        }
        let evidence = openai_exchange::json_completion::decode_body(body, elapsed)?;
        Ok((
            evidence.elapsed_seconds,
            evidence.prompt_tokens,
            evidence.cached_tokens,
            evidence.content_sha256,
        ))
    } else {
        if status != 200 {
            return Err("native completion HTTP status refused".into());
        }
        let evidence = openai_exchange::native_completion::decode_body(body, tokens, elapsed)?;
        let total = evidence
            .prompt_n
            .checked_add(evidence.cache_n)
            .ok_or("native prompt overflow")?;
        Ok((
            evidence.elapsed_seconds,
            total,
            Some(evidence.cache_n),
            evidence.content_sha256,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_transport_preserves_native_exact_status_and_openai_usage_protocol_gates() {
        let openai = serde_json::to_vec(&json!({"choices":[{"message":{"content":"measured"}}],
            "usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":8}}}))
        .unwrap();
        assert_eq!(
            project(true, 200, &openai, 2, Duration::ZERO).unwrap().2,
            Some(8)
        );
        assert!(project(true, 302, &openai, 2, Duration::ZERO).is_err());
        assert!(project(true, 503, &openai, 2, Duration::ZERO).is_err());
        assert!(project(false, 201, &openai, 2, Duration::ZERO).is_err());
        for body in [
            b"data: [DONE]\n\n".as_slice(),
            b"{}",
            b"{\"error\":\"refused\"}",
        ] {
            assert!(project(true, 200, body, 2, Duration::ZERO).is_err());
        }
    }
}

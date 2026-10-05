use super::*;

// ─── OpenRouter backend ──────────────────────────────────────────────

pub(crate) const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// A `ModelBackend` that reaches a real open-weight model through OpenRouter.
///
/// The request body is constructed to match `LocalModelBackend` /
/// `RemoteModelBackend` exactly (same keys, same `apply_enable_thinking`
/// injection), so the model sees what a mesh worker would. The only additions
/// are the bearer header OpenRouter requires and the same `HTTP 400
/// reasoning-mandatory` retry the in-tree `HttpBackend` carries.
pub(crate) struct OpenRouterBackend {
    http: reqwest::Client,
    api_key: String,
}

impl OpenRouterBackend {
    pub(crate) fn new(api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap_or_default();
        Self { http, api_key }
    }

    pub(crate) fn build_body(
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> Value {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        // Identical thinking-flag injection to the shipped mesh backends.
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        body
    }

    pub(crate) async fn post(
        &self,
        body: &Value,
        timeout: Duration,
    ) -> Result<reqwest::Response, String> {
        self.http
            .post(OPENROUTER_URL)
            .bearer_auth(&self.api_key)
            .json(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("openrouter request failed: {e}"))
    }

    /// Call with a predeclared retry on transient upstream/infra errors.
    ///
    /// OpenRouter surfaces provider rate-limits and gateway hiccups as
    /// 429/502/503/504. The eval must NOT score these as *capability*
    /// failures (the expert review flagged this), so we retry with backoff and,
    /// if still failing, return a distinguishable `INFRA:` error the caller
    /// treats as "excluded / missing", never as a wrong answer.
    pub(crate) async fn chat_completion_retrying(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let mut last = String::new();
        for attempt in 0..4 {
            match self
                .chat_completion(
                    model,
                    messages,
                    tools,
                    max_tokens,
                    Duration::from_secs(90),
                    sampling,
                )
                .await
            {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let transient = ["429", "502", "503", "504", "temporarily", "aborted"]
                        .iter()
                        .any(|c| e.contains(c));
                    last = e;
                    if !transient {
                        return Err(last); // genuine error (e.g. 400) — surface it
                    }
                    // backoff: 0.5s, 1s, 2s
                    tokio::time::sleep(Duration::from_millis(500 << attempt)).await;
                }
            }
        }
        Err(format!("INFRA: {last}"))
    }
}

#[async_trait]
impl ModelBackend for OpenRouterBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let body = Self::build_body(model, messages, tools, max_tokens, sampling);
        let resp = self.post(&body, timeout).await?;
        let status = resp.status();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();

            // Same failure mode the in-tree HttpBackend handles: some
            // endpoints reject our thinking-disable flags with HTTP 400.
            // Verified against minimax — it failed 12/12 until the flags
            // were dropped. A strict endpoint should cost a slightly slower
            // worker, not the worker entirely.
            if status.as_u16() == 400
                && text.to_ascii_lowercase().contains("reasoning")
                && sampling.enable_thinking == Some(false)
            {
                let mut retry = body.clone();
                if let Some(obj) = retry.as_object_mut() {
                    obj.remove("reasoning_effort");
                    obj.remove("chat_template_kwargs");
                }
                let r = self.post(&retry, timeout).await?;
                let retry_status = r.status();
                if !retry_status.is_success() {
                    let t = r.text().await.unwrap_or_default();
                    return Err(format!("HTTP {retry_status}: {}", truncate(&t, 200)));
                }
                return r.json::<Value>().await.map_err(|e| format!("parse: {e}"));
            }
            return Err(format!("HTTP {status}: {}", truncate(&text, 200)));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

// ─── Mesh-realism wrapper ────────────────────────────────────────────

/// Per-worker fault profile simulating a consumer-hardware mesh node.
///
/// OpenRouter is faster and far more reliable than a laptop on home wifi, so
/// this wrapper adds the two things a real mesh has that a cloud API does not:
/// slowness and flakiness. This is exactly the surface Together's MoA has no
/// answer for, so it is the part worth stressing hardest.
#[derive(Clone, Copy)]
pub(crate) struct MeshFault {
    /// Fixed slowdown added before delegating (models cold-loading, weak GPUs).
    extra_latency_ms: u64,
    /// Random 0..jitter added on top, sampled per call.
    jitter_ms: u64,
    /// Probability in [0,1] the worker hard-fails instead of answering
    /// (peer reset, OOM, timeout).
    failure_rate: f64,
}

impl MeshFault {
    pub(crate) const RELIABLE_FAST: Self = Self {
        extra_latency_ms: 150,
        jitter_ms: 400,
        failure_rate: 0.0,
    };
    pub(crate) const TYPICAL: Self = Self {
        extra_latency_ms: 600,
        jitter_ms: 1500,
        failure_rate: 0.05,
    };
    /// A big-tier node that is powerful but slow to first token — the case
    /// `strong_patience` exists for.
    pub(crate) const SLOW_STRONG: Self = Self {
        extra_latency_ms: 3000,
        jitter_ms: 3000,
        failure_rate: 0.05,
    };
    /// A genuinely unreliable peer.
    pub(crate) const FLAKY: Self = Self {
        extra_latency_ms: 400,
        jitter_ms: 2000,
        failure_rate: 0.25,
    };
}

/// Wraps any backend, injecting deterministic-per-call latency and failures.
pub(crate) struct MeshRealismBackend {
    inner: Arc<dyn ModelBackend>,
    fault: MeshFault,
    /// Seed mixed with a per-call counter so faults are reproducible within a
    /// run but differ across workers and calls.
    seed: u64,
    calls: AtomicU64,
}

impl MeshRealismBackend {
    pub(crate) fn wrap(
        inner: Arc<dyn ModelBackend>,
        fault: MeshFault,
        seed: u64,
    ) -> Arc<dyn ModelBackend> {
        Arc::new(Self {
            inner,
            fault,
            seed,
            calls: AtomicU64::new(0),
        })
    }
}

#[async_trait]
impl ModelBackend for MeshRealismBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        let mut rng = SmallRng::new(self.seed ^ (n.wrapping_mul(0x9E37_79B9_7F4A_7C15)));

        let jitter = if self.fault.jitter_ms > 0 {
            rng.next_u64() % self.fault.jitter_ms
        } else {
            0
        };
        tokio::time::sleep(Duration::from_millis(self.fault.extra_latency_ms + jitter)).await;

        if self.fault.failure_rate > 0.0 && rng.next_f64() < self.fault.failure_rate {
            return Err(format!("simulated mesh fault: {model} peer unreachable"));
        }

        self.inner
            .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
            .await
    }
}

/// Minimal splitmix64 — avoids adding a `rand` dependency for fault injection.
pub(crate) struct SmallRng(u64);
impl SmallRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

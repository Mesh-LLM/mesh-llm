//! Reproducible latency comparison through both normal HTTP endpoints.
//! Fresh model instances separate readout/generation caches and loading is untimed.

use super::*;
use futures_util::{StreamExt, stream};
use skippy_runtime::decision::causal::{DecisionPlan, ReadoutMethod};
use skippy_runtime::{ChatTemplateMessage, ChatTemplateOptions};

const BASELINE_TEMPLATE: &str = "{{ messages[0].content }}";
const BASELINE_GRAMMAR: &str = "root ::= \"B\" | \"S\"";

/// Construct the requested number of independent benchmark questions.
fn questions(count: usize) -> Value {
    let mut body = request();
    body["questions"] = Value::Object((0..count).map(|index| {
        (format!("question_{index:02}"), json!({
            "type": "choice", "instructions": format!("Choose the responsible department for assessment {index}."),
            "criteria": {"B": "billing", "S": "shipping"}
        }))
    }).collect());
    body
}

/// Build one constrained-generation baseline request for each question.
fn generation_requests(backend: &StageOpenAiBackend, body: &Value) -> Result<Vec<Value>> {
    let request = serde_json::from_value(body.clone())?;
    let decision = crate::frontend::system_one::decision_request(&request)?;
    let runtime = backend.runtime.lock().unwrap();
    let plan = DecisionPlan::for_model(&runtime.model, &decision)?;
    plan.questions
        .iter()
        .map(|question| {
            anyhow::ensure!(
                question.method == ReadoutMethod::SelectedTokenLogits,
                "one-token baseline requires single-token labels"
            );
            let prompt = runtime.model.detokenize(&question.prompt)?;
            anyhow::ensure!(
                runtime.model.tokenize(&prompt, true)? == question.prompt,
                "baseline prompt must roundtrip to exactly the decision input tokens"
            );
            let rendered = runtime.model.apply_chat_template_with_options(
                &[ChatTemplateMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                }],
                ChatTemplateOptions {
                    chat_template: Some(BASELINE_TEMPLATE.into()),
                    grammar: Some(BASELINE_GRAMMAR.into()),
                    skip_chat_parsing: true,
                    use_jinja: true,
                    ..Default::default()
                },
            )?;
            anyhow::ensure!(
                runtime.model.tokenize(&rendered, true)? == question.prompt,
                "constrained chat baseline must preserve the exact decision input tokens"
            );
            Ok(
                json!({"model": backend.model_id, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1, "temperature": 0.0, "grammar": BASELINE_GRAMMAR}),
            )
        })
        .collect()
}

/// Measure the constrained-generation baseline and retain its usage evidence.
async fn sample_generation(backend: Arc<StageOpenAiBackend>, requests: &[Value]) -> Result<Value> {
    let started = std::time::Instant::now();
    let mut running = stream::iter(requests.iter().cloned().map(|request| {
        let backend = backend.clone();
        async move { post_route(backend, "/v1/chat/completions", request).await }
    }))
    .buffer_unordered(4);
    let mut usage = Vec::new();
    while let Some(result) = running.next().await {
        let (status, response) = result?;
        anyhow::ensure!(
            status == StatusCode::OK,
            "generation baseline failed: {response}"
        );
        anyhow::ensure!(
            response["usage"]["completion_tokens"] == 1,
            "baseline must generate exactly one token"
        );
        let answer = response["choices"][0]["message"]["content"]
            .as_str()
            .context("completion text")?;
        anyhow::ensure!(
            ["B", "S"].contains(&answer),
            "baseline escaped offered labels: {response}"
        );
        usage.push(response["usage"].clone());
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok(json!({"elapsed_ms": seconds * 1000.0,
        "questions_per_second": requests.len() as f64 / seconds, "usage": usage}))
}

/// Measure an unsampled decision request and retain its scoring evidence.
async fn sample_readout(backend: Arc<StageOpenAiBackend>, body: &Value) -> Result<Value> {
    let count = body["questions"].as_object().unwrap().len();
    let started = std::time::Instant::now();
    let (status, response) = post(backend, body.clone()).await?;
    anyhow::ensure!(status == StatusCode::OK, "readout failed: {response}");
    anyhow::ensure!(response["answers"].as_object().context("answers")?.len() == count);
    anyhow::ensure!(response["usage"]["output_tokens"] == 0);
    let seconds = started.elapsed().as_secs_f64();
    Ok(json!({"elapsed_ms": seconds * 1000.0,
        "questions_per_second": count as f64 / seconds,
        "usage": response["usage"], "readout_usage": response["readout_usage"]}))
}

/// Collect comparable baseline/readout measurements for a configured serving topology.
async fn measure(
    config: StageConfig,
    split: bool,
    count: usize,
    generation: bool,
) -> Result<Value> {
    let mut return_listener = None;
    let (mut backend, terminal) = if split {
        let [first, last] = split::partition(&config)?;
        let listener =
            crate::binary_transport::PredictionReturnListener::start(first.bind_addr.parse()?)?;
        let terminal = split::terminal(last).await?;
        let mut backend = split::source(first)?;
        if let InferenceBackendMode::EmbeddedStageZero {
            prediction_returns, ..
        } = &mut Arc::get_mut(&mut backend)
            .context("exclusive benchmark backend")?
            .mode
        {
            *prediction_returns = Some(listener.hub());
        }
        return_listener = Some(listener);
        (backend, Some(terminal))
    } else {
        (Arc::new(backend_with_config(config)?), None)
    };
    let mut defaults = backend.request_defaults.snapshot().as_ref().clone();
    defaults.chat_template = Some(BASELINE_TEMPLATE.into());
    defaults.skip_chat_parsing = Some(true);
    Arc::get_mut(&mut backend)
        .context("exclusive baseline backend")?
        .request_defaults = defaults.into();
    let body = questions(count);
    let requests = generation_requests(&backend, &body)?;
    let mut samples = Vec::new();
    // First request is cold; subsequent requests reuse only this method's cache.
    for sample in 0..4 {
        let measured = if generation {
            sample_generation(backend.clone(), &requests).await
        } else {
            sample_readout(backend.clone(), &body).await
        };
        samples.push(measured.with_context(|| format!("sample {sample} (zero is cold)"))?);
    }
    assert_eq!(
        backend
            .runtime
            .lock()
            .unwrap()
            .session_stats()
            .active_sessions,
        0
    );
    drop(backend);
    if let Some(terminal) = terminal {
        terminal.shutdown().await?;
    }
    drop(return_listener);
    Ok(
        json!({"route": if split {"split_tcp"} else {"local"}, "questions": count,
        "method": if generation {"constrained_one_token_generation"} else {"prefill_readout"},
        "cold": samples.remove(0), "repeated": samples}),
    )
}

/// Benchmark local and split execution with one and many independent questions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires SKIPPY_DECISION_MODEL and SKIPPY_DECISION_BENCHMARK_REPORT"]
async fn causal_decisions_benchmark_local_and_split_one_and_many() -> Result<()> {
    let output = env::var_os("SKIPPY_DECISION_BENCHMARK_REPORT")
        .context("SKIPPY_DECISION_BENCHMARK_REPORT is required")?;
    let mut rows = Vec::new();
    for split in [false, true] {
        for count in [1, 8] {
            for cache in [false, true] {
                for generation in [false, true] {
                    let mut config = model_config()?;
                    if !cache {
                        config.kv_cache.as_mut().unwrap().mode = StageKvCacheMode::Disabled;
                    }
                    let mut result = measure(config, split, count, generation).await
                        .with_context(|| format!("split={split}, questions={count}, cache={cache}, generation={generation}"))?;
                    result["cache_enabled"] = json!(cache);
                    rows.push(result);
                }
            }
        }
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "baseline": "same prompt tokens; operator literal-content template; native B/S grammar; greedy; max_tokens=1; emitted label checked",
            "timing": "in-process HTTP including admission and scheduler, excluding model load; split uses loopback TCP",
            "quality": "mechanical benchmark, not model accuracy or calibration certification",
            "rows": rows
        }))?,
    )?;
    Ok(())
}

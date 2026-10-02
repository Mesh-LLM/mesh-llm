use super::{
    recorded_requests::{Selection, Trajectory, build},
    stream_evidence::Stream,
};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::{
    io::Write,
    time::{Duration, Instant},
};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix execute-trajectory --input PATH --output PATH --base-url URL --model ID [--timeout SECONDS] [--max-output-tokens N] [--qualification-probe]",
        values: &[
            "--input",
            "--output",
            "--base-url",
            "--model",
            "--timeout",
            "--max-output-tokens",
        ],
        flags: &["--help", "--qualification-probe"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input = parsed.last("--input").ok_or("missing --input")?;
    let output = parsed.last("--output").ok_or("missing --output")?;
    let base = parsed.last("--base-url").ok_or("missing --base-url")?;
    let model = parsed.last("--model").ok_or("missing --model")?;
    let seconds: u64 = parsed.last("--timeout").unwrap_or("900").parse()?;
    if !(1..=86400).contains(&seconds) {
        return Err("timeout must be in 1..=86400".into());
    }
    let trajectory: Trajectory = serde_json::from_slice(&std::fs::read(input)?)?;
    let selection = Selection {
        model,
        maximum_output_tokens: parsed
            .last("--max-output-tokens")
            .unwrap_or("2048")
            .parse()?,
        turn_limit: None,
        qualification_probe: parsed.flag("--qualification-probe"),
    };
    let turns = build(&trajectory, &selection)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut file = std::fs::File::create(output)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut failed = false;
    for turn in turns {
        let started = Instant::now();
        let mut progress = super::progress::Record::boundary(
            if turn.qualification_probe {
                super::progress::Phase::Preflight
            } else {
                super::progress::Phase::Request
            },
            super::progress::Event::Started,
            output.to_owned(),
        );
        progress.request = Some(super::progress::Request {
            request_id: turn.request_id.clone(),
            session_id: turn.session_id.clone(),
            assistant_turn: turn.assistant_turn,
        });
        let (outcome, result) = runtime.block_on(super::progress::in_flight(async {
            let cancelled = async {
                while !cancellation.is_cancelled() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            };
            tokio::select! {
                () = cancelled => (super::progress::Outcome::Cancelled, Err("replay interrupted".to_owned())),
                result = tokio::time::timeout(Duration::from_secs(seconds), exchange(base, &turn)) => match result {
                    Ok(Ok(evidence)) => (super::progress::Outcome::Success, Ok(evidence)),
                    Ok(Err(error)) => (super::progress::Outcome::Error, Err(error.to_string())),
                    Err(error) => (super::progress::Outcome::Timeout, Err(error.to_string())),
                },
            }
        }, progress.clone()));
        progress.event = super::progress::Event::Completed;
        progress.elapsed_seconds = started.elapsed().as_secs_f64();
        progress.outcome = Some(outcome);
        progress.metrics = result
            .as_ref()
            .ok()
            .map(|evidence| super::progress::Metrics {
                prompt_tokens: Some(evidence.prompt_tokens),
                completion_tokens: Some(evidence.completion_tokens),
                ..Default::default()
            });
        let _ = super::progress::emit(&progress);
        let mut record = serde_json::to_value(&turn)?;
        let object = record.as_object_mut().ok_or("invalid turn record")?;
        object.remove("body");
        match result {
            Ok(evidence) => {
                let evidence = serde_json::to_value(evidence)?;
                object.extend(
                    evidence
                        .as_object()
                        .ok_or("invalid stream evidence")?
                        .clone(),
                );
            }
            Err(error) => {
                failed = true;
                object.insert("error".into(), error.to_string().into());
            }
        }
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
        file.flush()?;
        if cancellation.is_cancelled() {
            break;
        }
    }
    interrupt.finish()?;
    if failed {
        Err("trajectory execution failed; request evidence retained".into())
    } else {
        Ok(())
    }
}

pub(super) async fn exchange(
    base: &str,
    turn: &super::recorded_requests::Turn,
) -> Result<super::stream_evidence::Evidence, Box<dyn std::error::Error + Send + Sync>> {
    let started = Instant::now();
    let uri: hyper::Uri = format!("{}/chat/completions", base.trim_end_matches('/')).parse()?;
    if uri.scheme_str() != Some("http") {
        return Err("replay execution requires HTTP".into());
    }
    let connection = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(connection)).await?;
    let request = hyper::Request::builder()
        .method("POST")
        .uri(uri.path_and_query().ok_or("missing request path")?.as_str())
        .header("host", uri.authority().ok_or("missing authority")?.as_str())
        .header("content-type", "application/json")
        .header("authorization", "Bearer EMPTY")
        .body(Full::new(Bytes::from(serde_json::to_vec(&turn.body)?)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()).into());
        }
        let mut stream = Stream::default();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(bytes) = frame?.data_ref() {
                stream.consume(bytes, started.elapsed())?;
                if stream.terminal() {
                    break;
                }
            }
        }
        Ok(stream.finish(started.elapsed(), turn.qualification_probe)?)
    };
    tokio::pin!(response);
    tokio::select! { result = &mut response => result, result = connection => { result?; response.await } }
}

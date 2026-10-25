//! complete KV cohort execution with retained overlap sibling evidence.
use super::{
    kv_cache_probe::{self, Geometry},
    kv_conversation::{self, Record},
    kv_options::Options,
    kv_overlap,
    kv_reports::Row,
    kv_requests,
    kv_transcripts::Transcripts,
    transport::Http,
};
use crate::command::DynResult;
use std::{sync::Arc, time::Instant};

pub(super) async fn run(
    http: Arc<Http>,
    options: &Options,
    transcripts: &mut Transcripts,
) -> DynResult<Vec<Row>> {
    let mut rows = Vec::new();
    'models: for model in &options.models {
        for attempt in 1..=options.attempts {
            if http.is_cancelled() {
                break 'models;
            }
            let started = Instant::now();
            let outcome = kv_conversation::run(&http, model, attempt, options.pressure_turns).await;
            transcripts.write(model, attempt, "tool_loop", &outcome.records)?;
            rows.push(Row::new(
                (model, attempt, "tool_loop"),
                started,
                outcome.status_code,
                None,
                outcome.result,
            ));
            if http.is_cancelled() {
                break 'models;
            }
            rows.push(overlap(http.clone(), options, model, attempt, transcripts).await?);
        }
        for (phase, geometry) in [
            ("same_prefix_cache", Geometry::SamePrefix),
            ("exact_prefix_cache", Geometry::ExactBody),
        ] {
            if http.is_cancelled() {
                break 'models;
            }
            let started = Instant::now();
            let proof = kv_cache_probe::measure(
                &http,
                model,
                geometry,
                options.minimum_cached,
                options.suffix_limit,
            )
            .await;
            rows.push(Row::new(
                (model, 0, phase),
                started,
                proof.status_code,
                proof.metrics,
                proof.result,
            ));
        }
    }
    Ok(rows)
}

async fn overlap(
    http: Arc<Http>,
    options: &Options,
    model: &str,
    attempt: u32,
    transcripts: &mut Transcripts,
) -> DynResult<Row> {
    let started = Instant::now();
    let contexts = kv_requests::overlap(model, attempt, options.overlap_requests);
    let starts = match kv_overlap::dispatch(http.clone(), contexts).await {
        Ok(starts) => starts,
        Err(error) => {
            transcripts.write(
                model,
                attempt,
                "overlap_dispatch",
                &[Record {
                    phase: "failure".into(),
                    status_code: None,
                    call_id: None,
                    error: Some(error.clone()),
                }],
            )?;
            return Ok(Row::new(
                (model, attempt, "overlap_tool_loop"),
                started,
                None,
                None,
                Err(error),
            ));
        }
    };
    let mut failures = Vec::new();
    let mut last_status = None;
    for start in starts {
        let context = start.context;
        match start.reply {
            Ok(reply) => {
                let outcome =
                    kv_conversation::continue_overlap(&http, model, &context, reply).await;
                last_status = outcome.status_code.or(last_status);
                transcripts.write(
                    model,
                    attempt,
                    &format!("overlap_{}", context.label),
                    &outcome.records,
                )?;
                if let Err(error) = outcome.result {
                    failures.push(format!("{}: {error}", context.label));
                }
            }
            Err(error) => {
                last_status = error.status.or(last_status);
                transcripts.write(
                    model,
                    attempt,
                    &format!("overlap_{}", context.label),
                    &[Record {
                        phase: format!("overlap_{}", context.label),
                        status_code: error.status,
                        call_id: None,
                        error: Some(error.detail.clone()),
                    }],
                )?;
                failures.push(format!("{}: {}", context.label, error.detail));
            }
        }
    }
    if http.is_cancelled() {
        failures.push("KV overlap certification cancelled before cache proof".into());
        return Ok(Row::new(
            (model, attempt, "overlap_tool_loop"),
            started,
            last_status,
            None,
            Err(failures.join("; ")),
        ));
    }
    // Sibling failures do not erase successful histories or measured cache observations.
    let proof = kv_cache_probe::measure(
        &http,
        model,
        Geometry::AfterOverlap,
        options.minimum_cached,
        options.suffix_limit,
    )
    .await;
    let cache_detail = match proof.result {
        Ok(detail) => detail,
        Err(error) => {
            failures.push(error);
            String::new()
        }
    };
    let result = if failures.is_empty() {
        Ok(format!(
            "completed {} overlapping starts and cache proof; {cache_detail}",
            options.overlap_requests
        ))
    } else {
        Err(failures.join("; "))
    };
    Ok(Row::new(
        (model, attempt, "overlap_tool_loop"),
        started,
        proof.status_code.or(last_status),
        proof.metrics,
        result,
    ))
}

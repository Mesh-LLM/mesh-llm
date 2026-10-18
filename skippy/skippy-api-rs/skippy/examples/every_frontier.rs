//! Every-frontier split verification through the typed facade.
//!
//! For one GGUF, runs the whole model as a reference, then splits it at every
//! possible stage frontier k in 1..N-1 (plus 3-stage chains at a coarse stride)
//! and checks that the terminal activation frame matches the reference within a
//! small f32 tolerance. This is the per-boundary coverage the canary's
//! fixed-shape family rows do not provide: a frontier-local defect (a misplaced
//! graph helper, a dropped state half, an API drift at one layer) fails here at
//! the exact k that introduced it.
//!
//! Usage:
//!   every_frontier <model.gguf> [--prompt TEXT] [--tolerance F]
//!                  [--three-stage-stride N] [--json]
//!
//! Environment:
//!   SKIPPY_TEST_MODEL may name the model instead of the positional argument.
//!
//! Exit status is non-zero if any frontier diverges. With --json, one JSON
//! object per frontier is printed for downstream job ingestion.

use std::process::ExitCode;

use skippy::{Runtime, StagePlan};

struct FrontierResult {
    label: String,
    ok: bool,
    max_abs_diff: f32,
    state_effects: usize,
    written_state_effects: usize,
    recurrent: bool,
    error: Option<String>,
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .or_else(|| std::env::var("SKIPPY_TEST_MODEL").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| {
            eprintln!("usage: every_frontier <model.gguf> [--prompt TEXT] [--tolerance F] [--three-stage-stride N] [--json]");
            std::process::exit(2);
        });
    let mut prompt = "Every frontier of this model must reproduce the reference activation frame.".to_owned();
    let mut tolerance = 1.0e-4f32;
    let mut three_stage_stride = 8usize;
    let mut json = false;
    let mut rest = args.peekable();
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--prompt" => prompt = rest.next().unwrap_or_default(),
            "--tolerance" => {
                tolerance = rest
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(tolerance);
            }
            "--three-stage-stride" => {
                three_stage_stride = rest
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(three_stage_stride);
            }
            "--json" => json = true,
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let runtime = match Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("runtime: {error}");
            return ExitCode::FAILURE;
        }
    };

    // Tokenize once through a plain session; staged chains re-load the model
    // from the path, so this session exists only to turn text into tokens.
    let tokens = {
        let model = match runtime.load_model(&path) {
            Ok(model) => model,
            Err(error) => {
                eprintln!("load {path}: {error}");
                return ExitCode::FAILURE;
            }
        };
        let mut session = match model.create_session(Default::default()) {
            Ok(session) => session,
            Err(error) => {
                eprintln!("session: {error}");
                return ExitCode::FAILURE;
            }
        };
        match session.tokenize(&prompt) {
            Ok(tokens) => tokens,
            Err(error) => {
                eprintln!("tokenize: {error}");
                return ExitCode::FAILURE;
            }
        }
    };
    let token_ids: Vec<i32> = tokens.iter().map(|token| token.0).collect();

    // Whole-model reference: one chain, one stage, terminal frame.
    let whole = match runtime.plan_stages(&path, 1) {
        Ok(plans) if plans.len() == 1 => plans,
        Ok(plans) => {
            eprintln!("plan_stages(1) returned {} plans, expected 1", plans.len());
            return ExitCode::FAILURE;
        }
        Err(error) => {
            eprintln!("plan whole model: {error}");
            return ExitCode::FAILURE;
        }
    };
    let layers = whole[0].layers.clone();
    let n_layers = layers.end;
    if n_layers < 2 {
        eprintln!("model has {n_layers} layers; nothing to split");
        return ExitCode::FAILURE;
    }

    let reference = match run_chain(&runtime, &path, &token_ids, &[(layers.start, n_layers)]) {
        Ok(Some(frame)) => frame,
        Ok(None) => {
            eprintln!("whole-model chain produced no terminal frame");
            return ExitCode::FAILURE;
        }
        Err(error) => {
            eprintln!("reference run: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut results = Vec::new();
    results.push(FrontierResult {
        label: format!("reference [0,{n_layers})"),
        ok: true,
        max_abs_diff: 0.0,
        state_effects: whole[0].semantics.state_effects,
        written_state_effects: whole[0].semantics.written_state_effects,
        recurrent: whole[0].semantics.recurrent,
        error: None,
    });

    // Two-stage matrix: every frontier.
    for k in 1..n_layers {
        let ranges = [(layers.start, k), (k, n_layers)];
        results.push(check_frontier(
            &runtime, &path, &token_ids, &reference, &ranges, tolerance,
        ));
    }

    // Three-stage chains at a coarse stride: chained-transport coverage.
    if three_stage_stride > 0 {
        let mut a = layers.start + three_stage_stride as i32;
        while a < n_layers - 1 {
            let b = (a + three_stage_stride as i32).min(n_layers - 1);
            let ranges = [(layers.start, a), (a, b), (b, n_layers)];
            results.push(check_frontier(
                &runtime, &path, &token_ids, &reference, &ranges, tolerance,
            ));
            a = b;
        }
    }

    let failures = results.iter().filter(|result| !result.ok).count();
    for result in &results {
        if json {
            println!(
                "{{\"label\":{:?},\"ok\":{},\"max_abs_diff\":{},\"state_effects\":{},\"written_state_effects\":{},\"recurrent\":{},\"error\":{:?}}}",
                result.label, result.ok, result.max_abs_diff,
                result.state_effects, result.written_state_effects,
                result.recurrent, result.error,
            );
        } else {
            let status = if result.ok { "ok" } else { "DIVERGED" };
            println!(
                "{status:>9}  {:<28} max|d|={:.3e}  state {}w/{} recurrent={}",
                result.label,
                result.max_abs_diff,
                result.written_state_effects,
                result.state_effects,
                result.recurrent,
            );
        }
    }
    if !json {
        println!(
            "{} frontiers checked, {failures} diverged (tolerance {tolerance:e}, {} prompt tokens)",
            results.len(),
            token_ids.len(),
        );
    }
    if failures == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn check_frontier(
    runtime: &Runtime,
    path: &str,
    tokens: &[i32],
    reference: &[u8],
    ranges: &[(i32, i32)],
    tolerance: f32,
) -> FrontierResult {
    let label = ranges
        .iter()
        .map(|(start, end)| format!("[{start},{end})"))
        .collect::<Vec<_>>()
        .join("+");
    let state = runtime
        .plan_stages(path, ranges.len() as u32)
        .map(|plans: Vec<StagePlan>| {
            plans
                .iter()
                .map(|plan| plan.semantics)
                .fold((0usize, 0usize, false), |(effects, written, recurrent), s| {
                    (effects + s.state_effects, written + s.written_state_effects, recurrent || s.recurrent)
                })
        });
    let base = FrontierResult {
        label: label.clone(),
        ok: false,
        max_abs_diff: f32::NAN,
        state_effects: state.as_ref().map(|(effects, _, _)| *effects).unwrap_or(0),
        written_state_effects: state.as_ref().map(|(_, written, _)| *written).unwrap_or(0),
        recurrent: state.as_ref().map(|(_, _, recurrent)| *recurrent).unwrap_or(false),
        error: None,
    };
    if let Err(error) = runtime.validate_stage_chain(path, ranges.len() as u32) {
        return FrontierResult {
            error: Some(format!("chain rejected: {error}")),
            ..base
        };
    }
    match run_chain(runtime, path, tokens, ranges) {
        Ok(Some(frame)) => match max_abs_diff(reference, &frame) {
            Some(max_abs_diff) if max_abs_diff <= tolerance => FrontierResult {
                ok: true,
                max_abs_diff,
                ..base
            },
            Some(max_abs_diff) => FrontierResult {
                max_abs_diff,
                error: Some(format!("activation diverged from whole-model reference")),
                ..base
            },
            None => FrontierResult {
                error: Some(format!(
                    "frame length {} does not match reference {}",
                    frame.len(),
                    reference.len()
                )),
                ..base
            },
        },
        Ok(None) => FrontierResult {
            error: Some("chain produced no terminal frame".to_owned()),
            ..base
        },
        Err(error) => FrontierResult {
            error: Some(error),
            ..base
        },
    }
}

fn run_chain(
    runtime: &Runtime,
    path: &str,
    tokens: &[i32],
    ranges: &[(i32, i32)],
) -> Result<Option<Vec<u8>>, String> {
    let native: Vec<skippy::Token> = tokens.iter().map(|id| skippy::Token(*id)).collect();
    let mut chain = runtime
        .open_chain(path, ranges)
        .map_err(|error| format!("open_chain: {error}"))?;
    chain
        .run(&native)
        .map_err(|error| format!("run: {error}"))
}

/// Largest absolute difference between two frames read as little-endian f32.
/// `None` when the buffers are not whole f32 frames of the same length.
fn max_abs_diff(reference: &[u8], candidate: &[u8]) -> Option<f32> {
    if reference.len() != candidate.len() || reference.len() % 4 != 0 {
        return None;
    }
    let mut max = 0.0f32;
    for (left, right) in reference.chunks_exact(4).zip(candidate.chunks_exact(4)) {
        let left = f32::from_le_bytes(left.try_into().ok()?);
        let right = f32::from_le_bytes(right.try_into().ok()?);
        max = max.max((left - right).abs());
    }
    Some(max)
}

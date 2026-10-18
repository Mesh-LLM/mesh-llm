//! Every-frontier split verification through the typed facade.
//!
//! For one GGUF, proves the split-serving invariant at every stage frontier:
//! feeding a stage the upstream stage's handover frame is equivalent to
//! direct prefill through both stages together.
//!
//! Phase 1 sweeps every frontier k in 1..N-1 as a 2-stage chain
//! `[0..k / k..N)` and records the terminal activation frame (the handover
//! into the terminal stage, i.e. the layer-k frontier activation computed
//! from tokens by direct prefill).
//!
//! Phase 2 runs 3-stage chains `[0..a / a..b / b..N)` and compares the
//! terminal frame against phase 1's frame at b: the value now arrived
//! through a real frontier handover at `a` instead of a direct prefill of
//! `[0..b)`. Adjacent handoffs (a = b-1) cover every boundary; a coarse set
//! of deeper handoffs crosses stateful boundaries. This is the per-boundary
//! coverage the canary's fixed-shape family rows do not provide.
//!
//! Usage:
//!   every_frontier <model.gguf> [--prompt TEXT] [--tolerance F]
//!                  [--deep-handoffs N] [--json]
//!
//! Environment:
//!   SKIPPY_TEST_MODEL may name the model instead of the positional argument.
//!
//! Exit status is non-zero if any handoff check diverges.

use std::process::ExitCode;

use skippy::{Runtime, StagePlan};

struct CheckResult {
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
            eprintln!(
                "usage: every_frontier <model.gguf> [--prompt TEXT] [--tolerance F] \
                 [--deep-handoffs N] [--json]"
            );
            std::process::exit(2);
        });
    let mut prompt =
        "Every frontier of this model must reproduce the reference activation frame.".to_owned();
    let mut tolerance = 1.0e-4f32;
    let mut deep_handoffs = 3usize;
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
            "--deep-handoffs" => {
                deep_handoffs = rest
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(deep_handoffs);
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
    if n_layers < 3 {
        eprintln!("model has {n_layers} layers; need at least 3 for handoff checks");
        return ExitCode::FAILURE;
    }

    let mut results = Vec::new();
    let mut direct: Vec<Option<Vec<u8>>> = vec![None; n_layers as usize + 1];

    // Phase 1: direct-prefill frontier activation at every k.
    for k in 1..n_layers {
        let ranges = [(layers.start, k), (k, n_layers)];
        let state = stage_state(&runtime, &path, ranges.len() as u32);
        let base = CheckResult {
            label: format!("direct [0,{k})"),
            ok: false,
            max_abs_diff: f32::NAN,
            state_effects: state.map(|(effects, _, _)| effects).unwrap_or(0),
            written_state_effects: state.map(|(_, written, _)| written).unwrap_or(0),
            recurrent: state.map(|(_, _, recurrent)| recurrent).unwrap_or(false),
            error: None,
        };
        match run_chain(&runtime, &path, &token_ids, &ranges) {
            Ok(Some(frame)) => {
                let continuity = (k > 1).then(|| {
                    let previous = direct[k as usize - 1].as_ref().unwrap();
                    max_abs_diff(previous, &frame).unwrap_or(f32::INFINITY)
                });
                direct[k as usize] = Some(frame);
                results.push(CheckResult {
                    ok: true,
                    max_abs_diff: 0.0,
                    ..base
                });
                if let Some(diff) = continuity {
                    results.push(CheckResult {
                        label: format!("continuity k={}", k - 1),
                        ok: true,
                        max_abs_diff: diff,
                        state_effects: 0,
                        written_state_effects: 0,
                        recurrent: false,
                        error: None,
                    });
                }
            }
            Ok(None) => results.push(CheckResult {
                error: Some("chain produced no terminal frame".to_owned()),
                ..base
            }),
            Err(error) => results.push(CheckResult {
                error: Some(error),
                ..base
            }),
        }
    }

    // Phase 2: handoff fidelity. A 3-stage chain hands stage [a..b) its input
    // through the real frontier-a frame; its terminal frame must equal the
    // direct-prefill frame at b. Adjacent pairs cover every boundary; the
    // coarse deeper set crosses stateful frontiers.
    let mut handoffs: Vec<(i32, i32)> = Vec::new();
    for b in 2..n_layers {
        handoffs.push((b - 1, b));
        let anchors: Vec<i32> = (0..deep_handoffs)
            .map(|index| layers.start + ((b - layers.start) * (index as i32 + 1)) / (deep_handoffs as i32 + 1))
            .collect();
        for a in anchors {
            if a > layers.start && a < b - 1 && !handoffs.contains(&(a, b)) {
                handoffs.push((a, b));
            }
        }
    }
    for (a, b) in handoffs {
        let ranges = [(layers.start, a), (a, b), (b, n_layers)];
        let state = stage_state(&runtime, &path, ranges.len() as u32);
        let base = CheckResult {
            label: format!("handoff a={a} -> b={b}"),
            ok: false,
            max_abs_diff: f32::NAN,
            state_effects: state.map(|(effects, _, _)| effects).unwrap_or(0),
            written_state_effects: state.map(|(_, written, _)| written).unwrap_or(0),
            recurrent: state.map(|(_, _, recurrent)| recurrent).unwrap_or(false),
            error: None,
        };
        let reference = direct[b as usize].clone();
        match (reference, run_chain(&runtime, &path, &token_ids, &ranges)) {
            (None, _) => results.push(CheckResult {
                error: Some("missing direct reference frame".to_owned()),
                ..base
            }),
            (Some(_), Ok(None)) => results.push(CheckResult {
                error: Some("chain produced no terminal frame".to_owned()),
                ..base
            }),
            (Some(_), Err(error)) => results.push(CheckResult {
                error: Some(error),
                ..base
            }),
            (Some(reference), Ok(Some(frame))) => {
                match max_abs_diff(&reference, &frame) {
                    Some(max_abs_diff) if max_abs_diff <= tolerance => results.push(CheckResult {
                        ok: true,
                        max_abs_diff,
                        ..base
                    }),
                    Some(max_abs_diff) => results.push(CheckResult {
                        max_abs_diff,
                        error: Some("handoff diverged from direct prefill".to_owned()),
                        ..base
                    }),
                    None => results.push(CheckResult {
                        error: Some(format!(
                            "frame length {} does not match reference {}",
                            frame.len(),
                            reference.len()
                        )),
                        ..base
                    }),
                }
            }
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
                "{status:>9}  {:<26} max|d|={:.3e}  state {}w/{} recurrent={}",
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
            "{} checks, {failures} diverged (tolerance {tolerance:e}, {} prompt tokens, {n_layers} layers)",
            results.len(),
            token_ids.len(),
        );
    }
    if failures == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn stage_state(runtime: &Runtime, path: &str, stages: u32) -> Option<(usize, usize, bool)> {
    let plans: Vec<StagePlan> = runtime.plan_stages(path, stages).ok()?;
    Some(plans.iter().map(|plan| plan.semantics).fold(
        (0usize, 0usize, false),
        |(effects, written, recurrent), semantics| {
            (
                effects + semantics.state_effects,
                written + semantics.written_state_effects,
                recurrent || semantics.recurrent,
            )
        },
    ))
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
    chain.run(&native).map_err(|error| format!("run: {error}"))
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

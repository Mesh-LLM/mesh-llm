//! Parity for speculative verify-window rollback on a split target.
//!
//! Two- and three-stage splits run greedy speculative windows with forced
//! rejections and must commit exactly the tokens a full model decodes one at
//! a time. A three-stage split has a middle stage that both receives and
//! sends activations, which a two-stage split never exercises. Each
//! split runs once with recurrent rollback snapshots deep enough for every
//! window to roll back in place, once with snapshots too shallow so every
//! window takes a host checkpoint, and once with pipelined windows, where a
//! window decodes before the previous one is resolved. Afterwards each split
//! decodes the same continuation one token at a time; every stage's output
//! activations must agree between the in-place and checkpoint runs, which
//! holds only if each stage's recurrent state and attention KV rolled back to
//! the same position. A last run changes the window width from step to
//! step, with single-token decodes between windows, so every stage rebuilds
//! and re-realizes its graph slice on each step.
//!
//! Needs `SKIPPY_CORRECTNESS_MODEL` set to a GGUF of a model with recurrent
//! layers that support rollback (for example Qwen3.5-4B); skipped otherwise.

use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail, ensure};
use skippy_runtime::{
    ActivationFrame, GgufStageRuntimePlan, ModelInfo, RuntimeConfig, RuntimeLoadMode, StageModel,
    StageSession, plan_gguf_stage_runtime_plans,
};

const PROMPT: &str = "Explain in three short paragraphs how TCP congestion control works.";
const CTX_SIZE: u32 = 512;
const GENERATED_TOKENS: usize = 48;
/// Tokens decoded one at a time after the speculative run.
const PROBE_TOKENS: usize = 4;
/// Rows per verify window: the committed token plus five proposals.
const WINDOW: usize = 6;
/// Deep enough for a whole window to roll back in place.
const DEEP_ROLLBACK: u32 = 8;
/// Shallower than a window, so every window checkpoints to host memory.
const SHALLOW_ROLLBACK: u32 = 2;
/// Proposal index rejected in successive windows; `None` accepts every row.
const REJECTIONS: [Option<usize>; 7] = [None, Some(0), Some(3), Some(1), None, Some(4), Some(2)];
/// Rows per step of a run whose window width keeps changing; 1 decodes the
/// committed token alone. Each change rebuilds every stage's graph.
const CHANGING_WIDTHS: [usize; 9] = [6, 2, 1, 4, 3, 1, 1, 5, 2];
/// For pipelined pairs: the rejected index in the first and second window.
/// Index 0 of the second window rejects its first, still speculative, row.
const PIPELINED_REJECTIONS: [(Option<usize>, Option<usize>); 5] = [
    (Some(2), None),
    (None, Some(0)),
    (None, None),
    (None, Some(3)),
    (Some(0), Some(1)),
];

fn correctness_model() -> Option<PathBuf> {
    env::var_os("SKIPPY_CORRECTNESS_MODEL").map(PathBuf::from)
}

fn n_gpu_layers() -> i32 {
    env::var("SKIPPY_PARITY_N_GPU_LAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(999)
}

fn layer_count(path: &PathBuf) -> Result<u32> {
    let info = ModelInfo::open(path).with_context(|| format!("inspect {}", path.display()))?;
    info.tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|layer| layer + 1)
        .context("model has no layer-indexed tensors")
}

fn open(
    path: &PathBuf,
    stage_index: u32,
    layers: (u32, u32),
    plan: Option<&GgufStageRuntimePlan>,
    rollback_tokens: u32,
) -> Result<StageModel> {
    let mut config = RuntimeConfig {
        stage_index,
        layer_start: layers.0,
        layer_end: layers.1,
        ctx_size: CTX_SIZE,
        n_gpu_layers: n_gpu_layers(),
        load_mode: RuntimeLoadMode::RuntimeSlice,
        recurrent_rollback_tokens: rollback_tokens,
        ..RuntimeConfig::default()
    };
    if let Some(plan) = plan {
        plan.clone().apply_to(&mut config);
    }
    StageModel::open(path, &config)
        .with_context(|| format!("open layers {}..{}", layers.0, layers.1))
}

/// A target that verifies windows and rolls them back.
enum Target {
    Single(Box<StageSession>),
    /// One session per stage, in stage order; each stage's output frame is
    /// the next stage's input.
    Split(Vec<StageSession>),
}

impl Target {
    fn split(models: &[StageModel]) -> Result<Self> {
        models
            .iter()
            .map(StageModel::create_session)
            .collect::<Result<_>>()
            .map(Self::Split)
    }

    fn prefill(&mut self, tokens: &[i32]) -> Result<()> {
        match self {
            Self::Single(session) => session.prefill_chunked(tokens),
            Self::Split(stages) => {
                let mut frame = None;
                for stage in stages {
                    frame = Some(stage.prefill_chunk_frame(tokens, frame.as_ref(), 0)?);
                }
                Ok(())
            }
        }
    }

    fn verify(&mut self, window: &[i32]) -> Result<Vec<i32>> {
        match self {
            Self::Single(session) => session.verify_tokens(window),
            Self::Split(stages) => {
                let mut frame: Option<ActivationFrame> = None;
                let mut predicted = Vec::new();
                for stage in stages {
                    let (stage_predicted, _, output) =
                        stage.verify_tokens_frame(window, frame.as_ref(), 0)?;
                    predicted = stage_predicted;
                    frame = Some(output);
                }
                Ok(predicted)
            }
        }
    }

    fn sessions(&mut self) -> Vec<&mut StageSession> {
        match self {
            Self::Single(session) => vec![&mut **session],
            Self::Split(stages) => stages.iter_mut().collect(),
        }
    }

    fn retire(&mut self, start: u64, count: u64) -> Result<()> {
        for session in self.sessions() {
            session.retire_verify_checkpoint(start, count)?;
        }
        Ok(())
    }

    fn trim(&mut self, token_count: u64) -> Result<()> {
        for session in self.sessions() {
            session.trim_session(token_count)?;
        }
        Ok(())
    }

    fn check_position(&mut self, expected: u64, label: &str) -> Result<()> {
        for (stage, session) in self.sessions().into_iter().enumerate() {
            ensure!(
                session.token_count() == expected && session.native_position()? == expected,
                "{label}: stage {stage} is at {} (native {}), expected {expected}",
                session.token_count(),
                session.native_position()?,
            );
        }
        Ok(())
    }

    /// Decodes `tokens` one at a time, returning each step's prediction and
    /// every stage's output activations.
    fn probe(&mut self, tokens: &[i32]) -> Result<Vec<(i32, Vec<ActivationFrame>)>> {
        tokens
            .iter()
            .map(|&token| match self {
                Self::Single(session) => {
                    let (predicted, frame) = session.decode_step_frame(token, None, 0)?;
                    Ok((predicted, vec![frame]))
                }
                Self::Split(stages) => {
                    let mut frames: Vec<ActivationFrame> = Vec::with_capacity(stages.len());
                    let mut predicted = 0;
                    for stage in stages.iter_mut() {
                        let (stage_predicted, frame) =
                            stage.decode_step_frame(token, frames.last(), 0)?;
                        predicted = stage_predicted;
                        frames.push(frame);
                    }
                    Ok((predicted, frames))
                }
            })
            .collect()
    }
}

/// Greedy continuation of the prompt, decoded one token at a time.
fn reference_tokens(model: &StageModel, prompt: &[i32]) -> Result<Vec<i32>> {
    let mut session = model.create_session()?;
    session.prefill_chunked(&prompt[..prompt.len() - 1])?;
    let mut current = *prompt.last().context("empty prompt")?;
    // Enough to draft past the last committed token.
    let length = GENERATED_TOKENS + 2 * WINDOW + PROBE_TOKENS;
    let mut tokens = Vec::with_capacity(length);
    while tokens.len() < length {
        current = session.decode_step(current)?;
        tokens.push(current);
    }
    Ok(tokens)
}

/// Proposals that follow the reference except at `reject`.
fn proposals(reference: &[i32], from: usize, count: usize, reject: Option<usize>) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let token = reference[from + index];
            if reject == Some(index) {
                token ^ 1
            } else {
                token
            }
        })
        .collect()
}

/// Rows accepted before the first prediction that disagrees with the next
/// window row.
fn accepted_rows(window: &[i32], predicted: &[i32]) -> usize {
    predicted
        .iter()
        .zip(&window[1..])
        .take_while(|(predicted, proposed)| predicted == proposed)
        .count()
}

struct Run {
    committed: Vec<i32>,
    probe: Vec<(i32, Vec<ActivationFrame>)>,
}

/// Probes the target with the last committed token and the reference
/// continuation after it.
fn finish(
    mut target: Target,
    committed: Vec<i32>,
    current: i32,
    reference: &[i32],
    label: &str,
) -> Result<Run> {
    let mut tokens = vec![current];
    tokens.extend(&reference[committed.len()..committed.len() + PROBE_TOKENS - 1]);
    let probe = target.probe(&tokens)?;
    for (step, (predicted, _)) in probe.iter().enumerate() {
        let expected = reference[committed.len() + step];
        ensure!(
            *predicted == expected,
            "{label}: probe step {step} predicted {predicted}, the full model decodes {expected}"
        );
    }
    Ok(Run { committed, probe })
}

/// Commits `predicted[..=accepted]`, checking each token against the full
/// model.
fn commit(
    run: &mut Vec<i32>,
    reference: &[i32],
    predicted: &[i32],
    accepted: usize,
    label: &str,
) -> Result<()> {
    for &token in &predicted[..=accepted] {
        let expected = reference[run.len()];
        ensure!(
            token == expected,
            "{label}: committed token {} is {token}, the full model decodes {expected}",
            run.len()
        );
        run.push(token);
    }
    Ok(())
}

fn run_serial(mut target: Target, prompt: &[i32], reference: &[i32], label: &str) -> Result<Run> {
    target.prefill(&prompt[..prompt.len() - 1])?;
    let prompt_prefix = (prompt.len() - 1) as u64;
    let mut committed = Vec::new();
    let mut current = *prompt.last().context("empty prompt")?;
    let mut window_index = 0;
    while committed.len() < GENERATED_TOKENS {
        let base = prompt_prefix + committed.len() as u64;
        let reject = REJECTIONS[window_index % REJECTIONS.len()];
        window_index += 1;
        let mut window = vec![current];
        window.extend(proposals(reference, committed.len(), WINDOW - 1, reject));
        let predicted = target.verify(&window)?;
        ensure!(
            predicted.len() == window.len(),
            "{label}: verify returned {} rows",
            predicted.len()
        );
        let accepted = accepted_rows(&window, &predicted);
        commit(&mut committed, reference, &predicted, accepted, label)?;
        if accepted + 1 == window.len() {
            target.retire(base, window.len() as u64)?;
        } else {
            target.trim(base + 1 + accepted as u64)?;
        }
        current = predicted[accepted];
        target.check_position(prompt_prefix + committed.len() as u64, label)?;
    }
    finish(target, committed, current, reference, label)
}

/// Serial windows whose width changes from one step to the next, with
/// single-token decodes between them.
fn run_changing_widths(
    mut target: Target,
    prompt: &[i32],
    reference: &[i32],
    label: &str,
) -> Result<Run> {
    target.prefill(&prompt[..prompt.len() - 1])?;
    let prompt_prefix = (prompt.len() - 1) as u64;
    let mut committed = Vec::new();
    let mut current = *prompt.last().context("empty prompt")?;
    let mut step = 0;
    while committed.len() < GENERATED_TOKENS {
        let base = prompt_prefix + committed.len() as u64;
        let width = CHANGING_WIDTHS[step % CHANGING_WIDTHS.len()];
        let reject = REJECTIONS[step % REJECTIONS.len()].filter(|index| index + 1 < width);
        step += 1;
        if width == 1 {
            let (predicted, _) = target
                .probe(&[current])?
                .pop()
                .context("decode returned no row")?;
            commit(&mut committed, reference, &[predicted], 0, label)?;
            current = predicted;
        } else {
            let mut window = vec![current];
            window.extend(proposals(reference, committed.len(), width - 1, reject));
            let predicted = target.verify(&window)?;
            ensure!(
                predicted.len() == window.len(),
                "{label}: verify returned {} rows",
                predicted.len()
            );
            let accepted = accepted_rows(&window, &predicted);
            commit(&mut committed, reference, &predicted, accepted, label)?;
            if accepted + 1 == window.len() {
                target.retire(base, window.len() as u64)?;
            } else {
                target.trim(base + 1 + accepted as u64)?;
            }
            current = predicted[accepted];
        }
        target.check_position(prompt_prefix + committed.len() as u64, label)?;
    }
    finish(target, committed, current, reference, label)
}

/// Two windows in flight: the second is verified before the first resolves,
/// drafted on the assumption that the first is accepted in full.
fn run_pipelined(
    mut target: Target,
    prompt: &[i32],
    reference: &[i32],
    label: &str,
) -> Result<Run> {
    target.prefill(&prompt[..prompt.len() - 1])?;
    let prompt_prefix = (prompt.len() - 1) as u64;
    let mut committed = Vec::new();
    let mut current = *prompt.last().context("empty prompt")?;
    let mut pair_index = 0;
    while committed.len() < GENERATED_TOKENS {
        let base = prompt_prefix + committed.len() as u64;
        let (reject_first, reject_second) =
            PIPELINED_REJECTIONS[pair_index % PIPELINED_REJECTIONS.len()];
        pair_index += 1;
        let mut first = vec![current];
        first.extend(proposals(
            reference,
            committed.len(),
            WINDOW - 1,
            reject_first,
        ));
        // The second window starts at the first one's bonus row, which is not
        // yet known when it is drafted.
        let second = proposals(
            reference,
            committed.len() + WINDOW - 1,
            WINDOW,
            reject_second,
        );
        let first_predicted = target.verify(&first)?;
        let second_predicted = target.verify(&second)?;
        let first_accepted = accepted_rows(&first, &first_predicted);
        if first_accepted + 1 < first.len() {
            commit(
                &mut committed,
                reference,
                &first_predicted,
                first_accepted,
                label,
            )?;
            target.trim(base + 1 + first_accepted as u64)?;
            current = first_predicted[first_accepted];
        } else {
            commit(
                &mut committed,
                reference,
                &first_predicted,
                first_accepted,
                label,
            )?;
            target.retire(base, first.len() as u64)?;
            let second_base = base + first.len() as u64;
            let bonus = first_predicted[first_accepted];
            if bonus != second[0] {
                // The second window's first row was wrong: drop all of it.
                target.trim(second_base)?;
                current = bonus;
            } else {
                let second_accepted = accepted_rows(&second, &second_predicted);
                commit(
                    &mut committed,
                    reference,
                    &second_predicted,
                    second_accepted,
                    label,
                )?;
                if second_accepted + 1 == second.len() {
                    target.retire(second_base, second.len() as u64)?;
                } else {
                    target.trim(second_base + 1 + second_accepted as u64)?;
                }
                current = second_predicted[second_accepted];
            }
        }
        target.check_position(prompt_prefix + committed.len() as u64, label)?;
    }
    finish(target, committed, current, reference, label)
}

/// Largest difference between two activation payloads read as f32 values,
/// relative to the largest magnitude in `lhs`.
fn relative_difference(lhs: &ActivationFrame, rhs: &ActivationFrame) -> Result<f32> {
    ensure!(
        lhs.payload.len() == rhs.payload.len() && lhs.payload.len().is_multiple_of(4),
        "activation payload sizes differ: {} vs {}",
        lhs.payload.len(),
        rhs.payload.len()
    );
    let (mut difference, mut magnitude) = (0.0_f32, 0.0_f32);
    for (lhs, rhs) in lhs
        .payload
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rhs.payload.as_chunks::<4>().0)
    {
        let (lhs, rhs) = (f32::from_le_bytes(*lhs), f32::from_le_bytes(*rhs));
        if !lhs.is_finite() || !rhs.is_finite() {
            bail!("activation payload holds a non-finite value: {lhs} vs {rhs}");
        }
        difference = difference.max((lhs - rhs).abs());
        magnitude = magnitude.max(lhs.abs());
    }
    Ok(if magnitude > 0.0 {
        difference / magnitude
    } else {
        difference
    })
}

/// Layer ranges of a split into `stages` nearly equal stages.
fn stage_ranges(layers: u32, stages: u32) -> Vec<(u32, u32)> {
    (0..stages)
        .map(|stage| (layers * stage / stages, layers * (stage + 1) / stages))
        .collect()
}

/// Runs every split scenario over `ranges` and checks it against the full
/// model's greedy decode.
fn check_split(
    path: &PathBuf,
    ranges: &[(u32, u32)],
    prompt: &[i32],
    reference: &[i32],
    single: &Run,
) -> Result<()> {
    let stages = ranges.len();
    let plans = plan_gguf_stage_runtime_plans(path, ranges, CTX_SIZE, 1)?;
    let split_target = |rollback| -> Result<Vec<StageModel>> {
        ranges
            .iter()
            .zip(&plans)
            .enumerate()
            .map(|(index, (&range, plan))| open(path, index as u32, range, Some(plan), rollback))
            .collect()
    };
    let deep = split_target(DEEP_ROLLBACK)?;
    let shallow = split_target(SHALLOW_ROLLBACK)?;

    let in_place = run_serial(
        Target::split(&deep)?,
        prompt,
        reference,
        &format!("{stages}-stage split in place"),
    )?;
    let checkpointed = run_serial(
        Target::split(&shallow)?,
        prompt,
        reference,
        &format!("{stages}-stage split checkpoint"),
    )?;
    let pipelined = run_pipelined(
        Target::split(&deep)?,
        prompt,
        reference,
        &format!("{stages}-stage split pipelined"),
    )?;
    // Every width change re-realizes each stage's slice of a freshly built
    // graph; the split must still commit what the full model decodes.
    let label = format!("{stages}-stage split changing widths");
    let changing = run_changing_widths(Target::split(&deep)?, prompt, reference, &label)?;

    assert_eq!(
        changing.committed,
        &reference[..changing.committed.len()],
        "{label}"
    );
    assert_eq!(in_place.committed, checkpointed.committed);
    assert_eq!(in_place.committed, single.committed);
    let common = pipelined.committed.len().min(single.committed.len());
    assert_eq!(pipelined.committed[..common], single.committed[..common]);

    // Both runs committed the same windows; only how each stage rolled back
    // differs. A rejected window's accepted rows replay in a smaller batch on
    // the checkpoint path, so activations may differ by rounding, not more.
    let tolerance = env::var("SKIPPY_VERIFY_ROLLBACK_TOLERANCE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1e-2_f32);
    ensure!(in_place.probe.len() == checkpointed.probe.len());
    for (step, ((_, in_place_frames), (_, checkpoint_frames))) in
        in_place.probe.iter().zip(&checkpointed.probe).enumerate()
    {
        ensure!(in_place_frames.len() == stages && checkpoint_frames.len() == stages);
        for (stage, (lhs, rhs)) in in_place_frames.iter().zip(checkpoint_frames).enumerate() {
            if lhs.payload.is_empty() && rhs.payload.is_empty() {
                continue;
            }
            let difference = relative_difference(lhs, rhs)?;
            eprintln!(
                "{stages} stages, probe step {step} stage {stage}: relative |in place - checkpoint| = {difference:e}"
            );
            ensure!(
                difference <= tolerance,
                "{stages} stages, probe step {step}: stage {stage} activations differ from the checkpoint path by {difference}"
            );
        }
    }
    Ok(())
}

#[test]
fn split_verify_rollback_matches_full_model_and_checkpoint_path() -> Result<()> {
    let Some(path) = correctness_model() else {
        eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
        return Ok(());
    };
    let layers = layer_count(&path)?;
    let full = open(&path, 0, (0, layers), None, DEEP_ROLLBACK)?;
    let prompt = full.tokenize(PROMPT, true)?;
    ensure!(prompt.len() > 2, "prompt is too short");
    let reference = reference_tokens(&full, &prompt)?;

    let single = run_serial(
        Target::Single(Box::new(full.create_session()?)),
        &prompt,
        &reference,
        "single node",
    )?;
    let single_pipelined = run_pipelined(
        Target::Single(Box::new(full.create_session()?)),
        &prompt,
        &reference,
        "single node pipelined",
    )?;
    assert_eq!(single.committed, &reference[..single.committed.len()]);
    let common = single_pipelined.committed.len().min(single.committed.len());
    assert_eq!(
        single_pipelined.committed[..common],
        single.committed[..common]
    );

    for stages in [2, 3] {
        check_split(
            &path,
            &stage_ranges(layers, stages),
            &prompt,
            &reference,
            &single,
        )?;
    }
    Ok(())
}

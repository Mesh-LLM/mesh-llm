//! Role-owned mixed anchor/prefill workload and original stage scheduling shape.
use super::{acceptance::Version, adaptive_identity as identity};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Role {
    Anchor,
    Prefill,
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct PromptRecord {
    pub family: String,
    pub prompt: String,
    #[serde(flatten)]
    pub provenance: Map<String, Value>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub prompts: Vec<PromptRecord>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Shape {
    pub rounds: u64,
    pub anchors: u32,
    pub prefills: u32,
    pub anchor_prompt_blocks: u32,
    pub prefill_prompt_blocks: u32,
    pub anchor_output_tokens: u64,
    pub prefill_output_tokens: u64,
    pub prefill_delay_ms: f64,
    pub prefill_stagger_ms: f64,
    pub lanes: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub prefill_adaptive_start: u32,
    pub prefill_adaptive_step: u32,
    pub prefill_adaptive_max: u32,
    pub adaptive_target_new_only: bool,
}
impl Shape {
    pub fn validate(&self) -> DynResult<()> {
        if !(1..=128).contains(&self.rounds)
            || self.anchors == 0
            || self.prefills == 0
            || self
                .anchors
                .checked_add(self.prefills)
                .is_none_or(|n| n > self.lanes || n > 16)
            || self.lanes > 64
            || self.n_batch == 0
            || self.n_batch > 1_048_576
            || self.n_ubatch == 0
            || self.n_ubatch > self.n_batch
            || self.prefill_adaptive_start == 0
            || self.prefill_adaptive_step == 0
            || self.prefill_adaptive_start > self.prefill_adaptive_max
            || self.prefill_adaptive_max > self.n_batch
            || !(1..=1024).contains(&self.anchor_prompt_blocks)
            || !(1..=1024).contains(&self.prefill_prompt_blocks)
            || !(1..=4096).contains(&self.anchor_output_tokens)
            || !(1..=4096).contains(&self.prefill_output_tokens)
            || [self.prefill_delay_ms, self.prefill_stagger_ms]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
            || self.prefill_delay_ms + (self.prefills - 1) as f64 * self.prefill_stagger_ms
                > 3_600_000.0
        {
            return Err("invalid mixed workload, lane/batch, adaptive or timing bounds".into());
        }
        Ok(())
    }
}
pub(super) fn stable_prompt(blocks: u32, index: i64, role: Role) -> DynResult<String> {
    if !(1..=1024).contains(&blocks) {
        return Err("mixed prompt block bound invalid".into());
    }
    let rows=(0..blocks).map(|i|format!("context-block-{i:04}: src/module_{}.rs owns invariant {i}; preserve the repository contract exactly.",i%37)).collect::<Vec<_>>().join("\n");
    let task = if role == Role::Anchor {
        "Continue a numbered implementation checklist with one item per line.".into()
    } else {
        format!(
            "Name the owner of invariant {}.",
            index.rem_euclid(i64::from(blocks))
        )
    };
    Ok(format!(
        "You are a deterministic coding assistant. Read this repository context.\n{rows}\nRequest {index}: {task}"
    ))
}
impl Manifest {
    pub fn validate(&self, shape: &Shape) -> DynResult<()> {
        shape.validate()?;
        let required = shape.rounds as usize * shape.prefills as usize;
        if self.prompts.len() != required
            || self.prompts.iter().any(|p| {
                p.family.trim().is_empty()
                    || p.prompt.trim().is_empty()
                    || p.prompt.len() > 256 * 1024
            })
        {
            return Err(
                "mixed manifest requires exact rounds*prefills nonempty family/prompt roster"
                    .into(),
            );
        }
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub role: Role,
    pub request_index: u32,
    pub prompt: PromptRecord,
    pub output_tokens: u64,
    pub delay_ms: f64,
}
pub(super) fn requests(
    shape: &Shape,
    round: u64,
    manifest: Option<&Manifest>,
) -> DynResult<Vec<Request>> {
    shape.validate()?;
    if round == 0 || round > shape.rounds {
        return Err("mixed round out of range".into());
    }
    if let Some(m) = manifest {
        m.validate(shape)?;
    }
    let mut rows = Vec::new();
    for index in 0..shape.anchors {
        rows.push(Request {
            role: Role::Anchor,
            request_index: index,
            prompt: PromptRecord {
                family: "synthetic-anchor".into(),
                prompt: stable_prompt(shape.anchor_prompt_blocks, i64::from(index), Role::Anchor)?,
                provenance: Map::new(),
            },
            output_tokens: shape.anchor_output_tokens,
            delay_ms: 0.0,
        });
    }
    for index in 0..shape.prefills {
        let id = shape.anchors + index;
        let prompt = if let Some(m) = manifest {
            m.prompts[((round - 1) * u64::from(shape.prefills) + u64::from(index)) as usize].clone()
        } else {
            PromptRecord {
                family: "synthetic-prefill".into(),
                prompt: stable_prompt(shape.prefill_prompt_blocks, i64::from(id), Role::Prefill)?,
                provenance: Map::new(),
            }
        };
        rows.push(Request {
            role: Role::Prefill,
            request_index: id,
            prompt,
            output_tokens: shape.prefill_output_tokens,
            delay_ms: shape.prefill_delay_ms + f64::from(index) * shape.prefill_stagger_ms,
        });
    }
    Ok(rows)
}
pub(super) fn config(
    arm: &identity::Input,
    shape: &Shape,
    index: usize,
    split: bool,
) -> DynResult<Value> {
    arm.validate()?;
    shape.validate()?;
    if index > usize::from(split) {
        return Err("mixed stage index outside topology".into());
    }
    let mut value = arm.config(index);
    let map = value.as_object_mut().ok_or("stage config object absent")?;
    map.remove("kv_cache");
    map.insert(
        "topology_id".into(),
        json!(if split {
            "skippy-mixed-prefill-decode-ab-two-stage"
        } else {
            "skippy-mixed-prefill-decode-ab-local"
        }),
    );
    map.insert("lane_count".into(), json!(shape.lanes));
    map.insert("n_batch".into(), json!(shape.n_batch));
    map.insert("n_ubatch".into(), json!(shape.n_ubatch));
    if !split {
        map.insert("layer_end".into(), json!(arm.layer_end));
        map.insert("downstream".into(), Value::Null);
        map.insert("upstream".into(), Value::Null);
    }
    Ok(value)
}
pub(super) fn arguments(
    arm: &identity::Input,
    shape: &Shape,
    config: &std::path::Path,
    index: usize,
    split: bool,
) -> DynResult<Vec<std::ffi::OsString>> {
    arm.validate()?;
    shape.validate()?;
    if index > usize::from(split) {
        return Err("mixed stage index outside topology".into());
    }
    let mut args = vec![
        "serve-binary".into(),
        "--config".into(),
        config.as_os_str().to_owned(),
        "--max-inflight".into(),
        shape.lanes.to_string().into(),
        "--telemetry-level".into(),
        "debug".into(),
    ];
    if split {
        args.extend([
            "--reply-credit-limit".into(),
            "1".into(),
            "--async-prefill-forward".into(),
        ]);
    }
    if index == 0 {
        for (flag, value) in [
            (
                "--openai-bind-addr",
                format!("127.0.0.1:{}", arm.openai_port),
            ),
            ("--openai-generation-concurrency", shape.lanes.to_string()),
            (
                "--openai-default-max-tokens",
                shape
                    .anchor_output_tokens
                    .max(shape.prefill_output_tokens)
                    .to_string(),
            ),
            ("--openai-prefill-chunk-policy", "adaptive-ramp".into()),
            ("--openai-prefill-chunk-size", shape.n_ubatch.to_string()),
            (
                "--openai-prefill-adaptive-start",
                shape.prefill_adaptive_start.to_string(),
            ),
            (
                "--openai-prefill-adaptive-step",
                shape.prefill_adaptive_step.to_string(),
            ),
            (
                "--openai-prefill-adaptive-max",
                shape.prefill_adaptive_max.to_string(),
            ),
        ] {
            args.extend([flag.into(), value.into()]);
        }
        if arm.version == Version::New || !shape.adaptive_target_new_only {
            args.extend([
                "--openai-prefill-adaptive-target-ms".into(),
                arm.adaptive_target_ms.to_string().into(),
            ]);
        }
    }
    Ok(args)
}

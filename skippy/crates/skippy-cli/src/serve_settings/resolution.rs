//! Translate catalog entries into typed runtime controls.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use skippy_protocol::{StageConfig, StageKvCacheMode};
use std::path::Path;

use skippy_serving::{
    DFLASH_STRATEGY, DFlashProposalConfig, SpeculativeDecodeConfig, settings::ServingTuning,
};

use super::{
    ServeSettings,
    catalog::{Kind, OPTIONS},
};

impl ServeSettings {
    pub fn validate_mode(&self, args: &crate::cli::ServeCommandArgs) -> Result<()> {
        for name in self.values.keys() {
            let section = super::section(name);
            if args.stage_transport.is_none()
                && (section == "network"
                    || matches!(
                        name.as_str(),
                        "activation-codec"
                            | "activation-codec-policy"
                            | "max-inflight"
                            | "reply-credit-limit"
                            | "async-prefill-forward"
                            | "no-async-prefill-forward"
                            | "downstream-connect-timeout-secs"
                            | "topology"
                    ))
            {
                bail!("--{name} requires --stage-transport binary");
            }
            if args.worker_only
                && (matches!(section, "sampling" | "chat")
                    || (section == "api" && name != "bind-addr")
                    || (section == "scheduling"
                        && !matches!(
                            name.as_str(),
                            "continuous-batching" | "generation-signal-window"
                        ))
                    || (section == "speculative" && name != "native-mtp"))
            {
                bail!("--{name} requires a public API and cannot be used with --worker-only");
            }
        }
        Ok(())
    }

    pub fn guardrail_mode(&self) -> Result<crate::cli::OpenAiGuardrailsCliMode> {
        use crate::cli::OpenAiGuardrailsCliMode;
        match self.text("guardrails").unwrap_or("disabled") {
            "disabled" => Ok(OpenAiGuardrailsCliMode::Disabled),
            "metrics" => Ok(OpenAiGuardrailsCliMode::Metrics),
            "enforce" => Ok(OpenAiGuardrailsCliMode::Enforce),
            _ => bail!("guardrails must be disabled, metrics, or enforce"),
        }
    }

    pub fn validate_cache_dependencies(&self, config: &StageConfig, disk: bool) -> Result<()> {
        if let Some(cache) = &config.kv_cache {
            anyhow::ensure!(
                cache.l2_max_bytes == 0 || disk,
                "--prefix-cache-ram requires an enabled --kv-cache-disk tier"
            );
            anyhow::ensure!(
                cache.mode != StageKvCacheMode::Disabled || cache.l2_max_bytes == 0,
                "RAM caching requires enabled prefix caching"
            );
        }
        if self.text("prefix-cache").is_some_and(|mode| mode != "off") {
            for name in ["SKIPPY_KV_CACHE", "SKIPPY_PREFIX_CACHE"] {
                let disabled = std::env::var(name).ok().is_some_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "off" | "false" | "0" | "disabled" | "disable"
                    )
                });
                anyhow::ensure!(
                    !disabled,
                    "{name} is an active legacy cache kill switch; unset it before enabling --prefix-cache"
                );
            }
        }
        Ok(())
    }

    pub fn stage_patch(&self) -> Result<Value> {
        let mut patch = json!({});
        for spec in OPTIONS {
            if spec.target.starts_with("request.")
                || spec.target.starts_with("tuning.")
                || spec.target.starts_with("spec.")
                || spec.target.starts_with("guardrails.")
                || spec.target.starts_with("compaction.")
            {
                continue;
            }
            let Some(value) = self.values.get(spec.name) else {
                continue;
            };
            if matches!(spec.kind, Kind::Bytes) && value.as_str() == Some("auto") {
                continue;
            }
            let value = if matches!(spec.kind, Kind::Bytes) {
                anyhow::ensure!(
                    spec.name != "prefix-cache-ram"
                        || !matches!(value.as_str(), Some("unbounded" | "auto")),
                    "prefix-cache-ram requires off or a fixed IEC budget"
                );
                anyhow::ensure!(
                    spec.name != "prefix-cache-budget" || value.as_str() != Some("off"),
                    "use --prefix-cache off to disable caching"
                );
                bytes(value)?
            } else if spec.name == "prefix-cache" {
                Value::String(
                    match value.as_str().context("prefix-cache must be a string")? {
                        "on" | "auto" => "lookup-record",
                        "off" => "disabled",
                        "record" => "record",
                        other => bail!("unsupported prefix-cache mode {other}"),
                    }
                    .into(),
                )
            } else {
                value.clone()
            };
            set(&mut patch, spec.target, value);
        }
        Ok(patch)
    }

    pub fn apply_stage(&self, config: &mut StageConfig) -> Result<()> {
        let patch = self.stage_patch()?;
        if patch.get("kv_cache").is_some() && config.kv_cache.is_none() {
            config.kv_cache = skippy_api::family_policy::family_policy_for_stage_config(config)
                .stage_kv_cache_config_for_stage(config);
        }
        let mut value = serde_json::to_value(&*config)?;
        merge(&mut value, patch);
        *config = serde_json::from_value(value).context("invalid runtime settings")?;
        validate_stage(config)?;
        Ok(())
    }

    pub fn tuning(&self, mode: crate::cli::OpenAiGuardrailsCliMode) -> Result<ServingTuning> {
        let mut request = json!({});
        let mut tuning = ServingTuning::default();
        for spec in OPTIONS
            .iter()
            .filter(|spec| spec.target.starts_with("request."))
        {
            if let Some(value) = self.values.get(spec.name) {
                set(&mut request, &spec.target[8..], value.clone());
            }
        }
        fill_sampler_defaults(&mut request);
        tuning.request_defaults =
            serde_json::from_value(request).context("invalid request defaults")?;
        tuning.n_threads = self.number("threads")?;
        tuning.n_threads_batch = self.number("threads-batch")?;
        tuning.pipeline_decode_groups = self.number("pipeline-decode-groups")?;
        tuning.continuous_batching = self.boolean("continuous-batching")?;
        // A DFlash draft is carried by the speculative plan, not the
        // separate draft-model runner.
        if !matches!(
            self.text("speculative-strategy"),
            Some("disabled" | DFLASH_STRATEGY)
        ) {
            tuning.draft_model_path = self.text("draft-model-path").map(Into::into);
            tuning.native_mtp_draft_model_path =
                self.text("native-mtp-draft-model-path").map(Into::into);
        }
        tuning.speculative_window = self.number("speculative-window")?;
        tuning.adaptive_speculative_window = self.boolean("adaptive-speculative-window")?;
        tuning.draft_n_gpu_layers = self.number("draft-n-gpu-layers")?;
        tuning.guardrails = Some(self.guardrails(mode)?);
        tuning.validate()?;
        Ok(tuning)
    }

    pub fn speculative(
        &self,
        mut base: SpeculativeDecodeConfig,
        draft_model_path: Option<&Path>,
    ) -> Result<SpeculativeDecodeConfig> {
        // An explicit --draft-model-path overrides a loaded DFlash plan's draft,
        // as individual flags override loaded plan fields. `draft_model_path`
        // may instead be a discovered default, which must not replace it.
        if let (Some(path), Some(dflash)) = (self.text("draft-model-path"), base.dflash.as_mut()) {
            dflash.draft_model_path = path.into();
        }
        let has_draft = draft_model_path.is_some();
        if !self.has_speculative_overrides() {
            base.validate()?;
            return Ok(base);
        }
        let mut value = serde_json::to_value(base)?;
        for spec in OPTIONS
            .iter()
            .filter(|spec| spec.target.starts_with("spec."))
        {
            if let Some(setting) = self.values.get(spec.name) {
                if spec.target.starts_with("spec.ngram.") && value["ngram"].is_null() {
                    value["ngram"] = json!({"kind":"cache", "min_ngram":2, "max_ngram":4, "max_proposal_tokens":4});
                }
                if spec.target.starts_with("spec.dflash.") && value["dflash"].is_null() {
                    // DFlash is opt-in: a tuning flag must not turn it on.
                    anyhow::ensure!(
                        self.text("speculative-strategy") == Some(DFLASH_STRATEGY),
                        "--{} requires --speculative-strategy dflash",
                        spec.name
                    );
                    let path =
                        draft_model_path.context("DFlash settings require --draft-model-path")?;
                    value["dflash"] = json!({"draft_model_path": path});
                }
                set(&mut value, &spec.target[5..], setting.clone());
            }
        }
        let mut plan: SpeculativeDecodeConfig =
            serde_json::from_value(value).context("invalid speculative settings")?;
        let strategy = self.text("speculative-strategy").unwrap_or("auto");
        let explicit_mtp = self.boolean("native-mtp")?;
        anyhow::ensure!(
            !matches!(
                strategy,
                "disabled" | "draft-model" | "ngram" | DFLASH_STRATEGY
            ) || explicit_mtp != Some(true),
            "--native-mtp=true conflicts with speculative strategy {strategy}"
        );
        anyhow::ensure!(
            !matches!(strategy, "native-mtp" | "mtp-ngram") || explicit_mtp != Some(false),
            "--native-mtp=false conflicts with speculative strategy {strategy}"
        );
        if strategy == "auto" && plan.ngram.is_some() && plan.extension.is_none() {
            anyhow::ensure!(
                explicit_mtp != Some(true),
                "native MTP with an N-gram proposer requires an extension policy"
            );
            plan.native_mtp.enabled = false;
        }
        if let Some(strategy) = self.text("speculative-strategy") {
            match strategy {
                "auto" => {}
                "disabled" => {
                    plan.native_mtp.enabled = false;
                    plan.ngram = None;
                    plan.extension = None;
                    plan.dflash = None;
                }
                "draft-model" => {
                    anyhow::ensure!(
                        has_draft,
                        "draft-model strategy requires --draft-model-path"
                    );
                    plan.native_mtp.enabled = false;
                    plan.ngram = None;
                    plan.extension = None;
                    plan.dflash = None;
                }
                "native-mtp" => {
                    plan.native_mtp.enabled = true;
                    plan.ngram = None;
                    plan.extension = None;
                    plan.dflash = None;
                }
                "ngram" => {
                    plan.native_mtp.enabled = false;
                    plan.extension = None;
                    plan.dflash = None;
                    anyhow::ensure!(
                        plan.ngram.is_some(),
                        "ngram strategy requires N-gram bounds or --ngram-kind"
                    );
                }
                "mtp-ngram" => {
                    plan.native_mtp.enabled = true;
                    plan.dflash = None;
                    anyhow::ensure!(
                        plan.ngram.is_some() && plan.extension.is_some(),
                        "mtp-ngram requires N-gram and extension settings"
                    );
                }
                DFLASH_STRATEGY => {
                    plan.native_mtp.enabled = false;
                    plan.ngram = None;
                    plan.extension = None;
                    // A loaded DFlash plan keeps its draft (an explicit
                    // --draft-model-path was already applied to it); a
                    // discovered default path must not replace it.
                    if plan.dflash.is_none() {
                        let path = draft_model_path
                            .context("dflash strategy requires --draft-model-path")?;
                        plan.dflash = Some(DFlashProposalConfig {
                            draft_model_path: path.to_path_buf(),
                            max_draft_tokens: None,
                        });
                    }
                }
                other => bail!("unsupported speculative strategy {other}"),
            }
        }
        plan.effective_strategy = if self.text("speculative-strategy") == Some("disabled") {
            "disabled"
        } else if plan.dflash.is_some() {
            DFLASH_STRATEGY
        } else if plan.extension.is_some() {
            "mtp-ngram"
        } else if plan.ngram.is_some() {
            "ngram"
        } else if plan.native_mtp.enabled {
            "native-mtp"
        } else if has_draft {
            "draft-model"
        } else {
            "disabled"
        }
        .into();
        anyhow::ensure!(
            !plan.ngram_fallback_draft || has_draft,
            "ngram-fallback-draft requires --draft-model-path"
        );
        plan.validate()?;
        Ok(plan)
    }

    pub fn has_speculative_overrides(&self) -> bool {
        OPTIONS
            .iter()
            .any(|spec| spec.target.starts_with("spec.") && self.values.contains_key(spec.name))
    }

    pub fn text(&self, name: &str) -> Option<&str> {
        self.values.get(name).and_then(Value::as_str)
    }

    pub fn number<T: serde::de::DeserializeOwned>(&self, name: &str) -> Result<Option<T>> {
        self.values
            .get(name)
            .map(|value| {
                let value = if let Some(text) = value.as_str() {
                    serde_json::from_str(text)?
                } else {
                    value.clone()
                };
                serde_json::from_value(value).with_context(|| format!("invalid {name}"))
            })
            .transpose()
    }

    pub fn boolean(&self, name: &str) -> Result<Option<bool>> {
        self.values
            .get(name)
            .map(|value| {
                if let Some(text) = value.as_str() {
                    text.parse()
                        .with_context(|| format!("{name} must be true or false"))
                } else {
                    value
                        .as_bool()
                        .with_context(|| format!("{name} must be true or false"))
                }
            })
            .transpose()
    }

    fn guardrails(
        &self,
        mode: crate::cli::OpenAiGuardrailsCliMode,
    ) -> Result<skippy_serving::InferenceGuardrailsConfig> {
        use skippy_serving::frontend::InferenceGuardrailsConfig;
        let mut config = InferenceGuardrailsConfig::for_standalone_mode(mode.into());
        let mut policy = config.policy.snapshot();
        if let Some(value) = self.number("guardrails-tool-retries")? {
            policy.max_tool_retries = value;
        }
        if let Some(value) = self.number("guardrails-structured-retries")? {
            policy.max_structured_retries = value;
        }
        if let Some(value) = self.boolean("guardrails-all-models")? {
            policy.apply_to_all_models = value;
        }
        if let Some(value) = self.number::<f32>("guardrails-small-model-threshold")? {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "guardrails-small-model-threshold must be finite and nonnegative"
            );
            policy.small_param_threshold_b = value;
        }
        if let Some(value) = self.text("guardrails-reserved-tool-prefix") {
            policy.reserved_tool_prefix = value.into();
        }
        if let Some(value) = self.text("guardrails-retry-exhaustion") {
            policy.retry_exhaustion_mode = match value {
                "error" => skippy_inference_api::RetryExhaustionMode::Error,
                "pass-last-text" => skippy_inference_api::RetryExhaustionMode::PassLastText,
                _ => bail!("guardrails-retry-exhaustion must be error or pass-last-text"),
            };
        }
        config.policy = policy.into();
        let mut compact = config.compaction.unwrap_or_default();
        if let Some(value) = self.boolean("compact")? {
            compact.enabled = value;
        }
        if let Some(value) = self.boolean("compact-drop-reasoning")? {
            compact.allow_reasoning_drop = value;
        }
        if let Some(value) = self.number("compact-context-limit")? {
            compact.context_limit_tokens = Some(value);
        }
        if let Some(value) = self.number("compact-trigger-percent")? {
            compact.trigger_ratio_percent = value;
        }
        if let Some(value) = self.number("compact-target-percent")? {
            compact.target_ratio_percent = value;
        }
        anyhow::ensure!(
            compact.context_limit_tokens != Some(0),
            "compact-context-limit must be positive"
        );
        anyhow::ensure!(
            compact.target_ratio_percent < compact.trigger_ratio_percent
                && compact.trigger_ratio_percent <= 100,
            "compaction requires target < trigger <= 100"
        );
        config.compaction = Some(compact);
        Ok(config)
    }
}

fn validate_stage(config: &StageConfig) -> Result<()> {
    skippy_runtime::parse_cache_type(&config.cache_type_k)?;
    skippy_runtime::parse_cache_type(&config.cache_type_v)?;
    anyhow::ensure!(
        config.n_batch != Some(0) && config.n_ubatch != Some(0),
        "batch and microbatch sizes must be positive"
    );
    anyhow::ensure!(
        config.n_gpu_layers >= -1,
        "n-gpu-layers must be -1 or nonnegative"
    );
    anyhow::ensure!(
        config
            .image_min_tokens
            .zip(config.image_max_tokens)
            .is_none_or(|(min, max)| min <= max),
        "image-min-tokens must not exceed image-max-tokens"
    );
    anyhow::ensure!(
        config
            .activation_codec_policy
            .compatible(config.activation_codec),
        "auto-lossless-v1 requires raw-f32-v1 as its fallback codec"
    );
    if let Some(cache) = &config.kv_cache {
        anyhow::ensure!(
            cache.mode == StageKvCacheMode::Disabled || cache.max_entries > 0,
            "prefix-cache-max-entries must be positive; use --prefix-cache off to disable caching"
        );
    }
    Ok(())
}

fn fill_sampler_defaults(request: &mut Value) {
    let defaults = skippy_runtime::SamplingConfig::default();
    if request.get("dry").is_some() {
        let mut dry = json!({"multiplier":defaults.dry.multiplier, "base":defaults.dry.base, "allowed_length":defaults.dry.allowed_length, "penalty_last_n":defaults.dry.penalty_last_n, "sequence_breakers":defaults.dry.sequence_breakers});
        merge(&mut dry, request["dry"].take());
        request["dry"] = dry;
    }
    if request.get("xtc").is_some() {
        let mut xtc =
            json!({"probability":defaults.xtc.probability, "threshold":defaults.xtc.threshold});
        merge(&mut xtc, request["xtc"].take());
        request["xtc"] = xtc;
    }
}

fn bytes(value: &Value) -> Result<Value> {
    use skippy_cache::disk_policy::{DiskCacheBudget, parse_disk_budget};
    let text = value
        .as_str()
        .context("budget must be off, unbounded, auto, or an IEC size")?;
    if matches!(text, "off" | "unbounded") {
        return Ok(Value::from(0));
    }
    match parse_disk_budget(text)? {
        DiskCacheBudget::Fixed(value) => Ok(Value::from(value)),
        _ => bail!("expected a fixed IEC byte budget"),
    }
}

pub(super) fn set(object: &mut Value, path: &str, value: Value) {
    if !object.is_object() {
        *object = json!({});
    }
    if let Some((head, tail)) = path.split_once('.') {
        set(&mut object[head], tail, value);
    } else {
        object[path] = value;
    }
}

pub(super) fn merge(base: &mut Value, patch: Value) {
    if let (Some(base), Some(patch)) = (base.as_object_mut(), patch.as_object()) {
        for (key, value) in patch {
            merge(base.entry(key).or_insert(Value::Null), value.clone());
        }
    } else {
        *base = patch;
    }
}

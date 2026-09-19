//! Model loading and backend composition shared by embedding applications.
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use openai_frontend::{GuardrailTelemetrySink, OpenAiBackend, OpenAiHookPolicy};
use skippy_protocol::{LoadMode, StageConfig};
use skippy_server::{
    EmbeddedRuntimeOptions, OpenAiGuardrailsConfig, SkippyRuntimeHandle,
    binary_transport::{PredictionReturnListener, WireCondition},
    embedded_openai_backend,
    frontend::{GenerationLifecycleConfig, GenerationLifecycleIngress},
    kv_integration::KvLifecycleObserver,
    serving_hooks::{ModelServingHooks, SharedModelServingHooksFactory},
    telemetry::Telemetry,
};

use super::{OpenAiOptions, ServingTelemetryOptions};

/// Whether the caller needs the native model-open event path, even without a sink.
pub enum ModelOpenEvents {
    Disabled,
    Enabled(Option<Box<dyn FnMut(skippy_runtime::RuntimeEvent) + Send>>),
}

/// Explicit inputs; product configuration, plugin loading and event storage stay outside Skippy.
pub struct ModelLoadRequest {
    pub runtime: EmbeddedRuntimeOptions,
    pub openai: OpenAiOptions,
    pub open_events: ModelOpenEvents,
    pub hooks_factory: Option<SharedModelServingHooksFactory>,
    pub generation_observer: Option<Arc<dyn GenerationLifecycleIngress>>,
    pub kv_observer: Option<Arc<dyn KvLifecycleObserver>>,
    pub hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    pub guardrails: Option<OpenAiGuardrailsConfig>,
    pub guardrail_telemetry: Option<Arc<dyn GuardrailTelemetrySink>>,
    pub downstream_wire_condition: WireCondition,
    pub serving_telemetry: Option<ServingTelemetryOptions>,
}

/// Keep the native runtime and prediction listener alive for the backend's lifetime.
pub struct LoadedModelBackend {
    pub runtime: SkippyRuntimeHandle,
    pub backend: Arc<dyn OpenAiBackend>,
    pub config: StageConfig,
    pub prediction_return_listener: Option<PredictionReturnListener>,
}

impl ModelLoadRequest {
    pub fn load(self) -> Result<LoadedModelBackend> {
        let config = self.runtime.config.clone();
        ensure!(
            config.stage_index == 0 && config.layer_start == 0,
            "OpenAI model loading requires stage 0"
        );
        ensure!(
            config.load_mode != LoadMode::LayerPackage,
            "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
        );
        let settings = self
            .serving_telemetry
            .unwrap_or_else(|| ServingTelemetryOptions {
                metrics_otlp_grpc: self.runtime.metrics_otlp_grpc.clone(),
                queue_capacity: self.runtime.telemetry_queue_capacity,
                level: self.runtime.telemetry_level,
            });
        let telemetry = Telemetry::new(
            settings.metrics_otlp_grpc,
            settings.queue_capacity,
            config.clone(),
            settings.level,
        );
        let runtime = match self.open_events {
            ModelOpenEvents::Disabled => SkippyRuntimeHandle::load(self.runtime),
            ModelOpenEvents::Enabled(reporter) => {
                SkippyRuntimeHandle::load_with_open_events(self.runtime, reporter)
            }
        }
        .with_context(|| {
            format!(
                "load Skippy runtime for model {} from {:?}",
                config.model_id, config.model_path
            )
        })?;
        let hooks = resolve_hooks(
            self.hooks_factory.as_ref(),
            &runtime,
            self.generation_observer,
            self.kv_observer,
        )?;
        let prediction_return_listener = if config.downstream.is_some() {
            Some(PredictionReturnListener::start(config.bind_addr.parse()?)?)
        } else {
            None
        };
        let mut args = self.openai.build(
            ([127, 0, 0, 1], 0).into(),
            config.clone(),
            runtime.runtime(),
            telemetry,
            self.hook_policy,
        );
        // The loaded graph, not a caller's width estimate, binds the stage boundary.
        if config.downstream.is_some() {
            args.activation_width = runtime
                .output_activation_boundary()
                .context("stage 0 graph did not expose its output activation boundary")?
                .raw_f32_width("output")?;
        }
        args.prediction_returns = prediction_return_listener
            .as_ref()
            .map(PredictionReturnListener::hub);
        args.downstream_wire_condition = self.downstream_wire_condition;
        args.generation_receipt = hooks.generation_receipt();
        args.generation_lifecycle = hooks.generation_lifecycle();
        args.linear_proposal_ingress = hooks.linear_proposal_ingress();
        args.kv_lifecycle_observer = hooks.kv_lifecycle_observer();
        let binding = embedded_openai_backend(args).context("construct Skippy OpenAI backend")?;
        let backend = match self.guardrails {
            Some(guardrails) => guardrails.wrap_backend_with_telemetry(
                binding.backend,
                Some(usize::try_from(config.ctx_size).unwrap_or(usize::MAX)),
                self.guardrail_telemetry,
            ),
            None => binding.backend,
        };
        Ok(LoadedModelBackend {
            runtime,
            backend,
            config,
            prediction_return_listener,
        })
    }
}

fn resolve_hooks(
    factory: Option<&SharedModelServingHooksFactory>,
    runtime: &SkippyRuntimeHandle,
    generation_observer: Option<Arc<dyn GenerationLifecycleIngress>>,
    kv_observer: Option<Arc<dyn KvLifecycleObserver>>,
) -> Result<ModelServingHooks> {
    let mut hooks = match factory {
        Some(factory) => factory
            .create(
                runtime
                    .tokenizer_capability()
                    .context("loaded Skippy runtime cannot provide its tokenizer capability")?,
                generation_observer,
            )
            .context("serving hook factory rejected the loaded model")?,
        None => match generation_observer {
            Some(observer) => ModelServingHooks::default()
                .with_generation_lifecycle(GenerationLifecycleConfig::from_ingress(observer)),
            None => ModelServingHooks::default(),
        },
    };
    if let Some(observer) = kv_observer {
        hooks = hooks.with_kv_lifecycle_observer(observer);
    }
    Ok(hooks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ModelLoadRequest {
        let config: StageConfig =
            serde_json::from_value(skippy_server::config::example_config()).unwrap();
        ModelLoadRequest {
            openai: OpenAiOptions::direct_single_stage_defaults(
                config.model_id.clone(),
                32,
                1,
                false,
            ),
            runtime: EmbeddedRuntimeOptions {
                config,
                topology: None,
                n_threads: None,
                n_threads_batch: None,
                mtp_source: skippy_runtime::MtpSource::Disabled,
                metrics_otlp_grpc: None,
                telemetry_queue_capacity: 0,
                telemetry_level: skippy_server::telemetry::TelemetryLevel::Off,
                operation_id: None,
                session_lifecycle_observer: None,
            },
            open_events: ModelOpenEvents::Disabled,
            hooks_factory: None,
            generation_observer: None,
            kv_observer: None,
            hook_policy: None,
            guardrails: None,
            guardrail_telemetry: None,
            downstream_wire_condition: WireCondition::new(0.0, None).unwrap(),
            serving_telemetry: None,
        }
    }

    #[test]
    fn worker_stage_is_rejected_before_native_loading() {
        let mut request = request();
        request.runtime.config.stage_index = 1;
        request.runtime.config.layer_start = 1;
        request.runtime.config.model_path = Some("missing.gguf".into());
        let error = request
            .load()
            .err()
            .expect("worker is not an OpenAI entrypoint");
        assert!(error.to_string().contains("requires stage 0"));
    }

    #[test]
    fn legacy_package_is_rejected_before_native_loading_or_event_callbacks() {
        let mut request = request();
        request.runtime.config.load_mode = LoadMode::LayerPackage;
        request.open_events = ModelOpenEvents::Enabled(Some(Box::new(|_| {
            panic!("invalid package must not enter native model loading");
        })));
        let error = request
            .load()
            .err()
            .expect("legacy package must fail closed");
        assert!(error.to_string().contains("package-v2 graph admission"));
    }
}

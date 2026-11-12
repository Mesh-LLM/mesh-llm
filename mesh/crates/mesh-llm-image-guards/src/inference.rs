//! Private embedded Skippy backend: never uses the host's mesh inference RPC.
use crate::{
    catalog::{self, Model, Paths, Settings},
    policy,
};
use anyhow::Result;
use base64::Engine;
use mesh_llm_plugin::image_safety::{GuardDecision, MAX_IMAGE_DATA_URL_BYTES};
use serde_json::{Value, json};
use skippy_api::{
    SingleStageOptions, StageSourceIdentity,
    serving::{InferenceOptions, LoadedModelBackend, ModelLoadRequest, ModelOpenEvents},
};
use skippy_inference_api::{
    ChatCompletionRequest, InferenceRequestContext, generation::GenerationRequest,
};
use skippy_serving::{EmbeddedRuntimeOptions, binary_transport::WireCondition};
use std::{io::Cursor, sync::Arc, time::Duration};

pub struct Classifier {
    model: Model,
    loaded: LoadedModelBackend,
    context_size: u32,
    gate: Arc<tokio::sync::Semaphore>,
}

impl Classifier {
    pub fn load(model: Model, paths: Paths, settings: Settings) -> Result<Self> {
        catalog::verify(&paths.weights, model.weights())?;
        if let (Some(path), Some(asset)) = (&paths.projector, model.projector()) {
            catalog::verify(path, asset)?;
        }
        load_native_runtime(settings.runtime_selection.clone())?;
        let mut options = SingleStageOptions::new(model.id(), &paths.weights);
        options.ctx_size = settings.ctx_size;
        options.generation_concurrency = 1;
        options.n_gpu_layers = settings.n_gpu_layers;
        options.projector_path = paths.projector;
        if model == Model::Nemotron {
            options.media_marker = Some("<mesh_safety_image>".to_owned());
        }
        options.projector_use_gpu = Some(settings.n_gpu_layers != 0);
        options.native_mtp_enabled = false;
        options.n_batch = Some(256);
        options.n_ubatch = Some(256);
        let identity = StageSourceIdentity {
            package_ref: format!("{}@{}", model.repo(), model.revision()),
            manifest_sha256: model.weights().sha256.to_owned(),
            source_model_path: paths.weights,
            source_model_sha256: model.weights().sha256.to_owned(),
            source_model_bytes: model.weights().bytes,
            layer_count: model.layers(),
        };
        let config =
            skippy_api::single_stage_config(&options, identity, "private-image-guard".to_owned())?;
        let mut openai =
            InferenceOptions::direct_single_stage_defaults(model.id().to_owned(), 64, 1, false);
        openai.generation_queue_capacity = 1;
        openai.generation_admission_timeout_secs = 5;
        openai.request_defaults.chat_template = Some(model.template().to_owned());
        openai.request_defaults.chat_template_kwargs = Some(match model {
            Model::YuFeng => json!({"reason_first": false}),
            Model::Nemotron => json!({"custom_policy": policy::CHILD_POLICY,
                "request_categories": "/no_categories", "enable_thinking": false,
                "mesh_image_input": false}),
        });
        let loaded = ModelLoadRequest {
            runtime: EmbeddedRuntimeOptions {
                config,
                topology: None,
                n_threads: settings.n_threads,
                n_threads_batch: settings.n_threads,
                mtp_source: skippy_runtime::MtpSource::Disabled,
                recurrent_rollback_tokens: 0,
                metrics_otlp_grpc: None,
                telemetry_queue_capacity: 0,
                telemetry_level: skippy_serving::telemetry::TelemetryLevel::Off,
                session_lifecycle_observer: None,
            },
            openai,
            open_events: ModelOpenEvents::Disabled,
            hooks_factory: None,
            generation_observer: None,
            kv_observer: None,
            hook_policy: None,
            guardrails: None,
            guardrail_telemetry: None,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            serving_telemetry: None,
            l3_manager: None,
        }
        .load()?;
        if model == Model::Nemotron {
            verify_image_template(&loaded)?;
        }
        Ok(Self {
            model,
            loaded,
            context_size: settings.ctx_size,
            gate: Arc::new(tokio::sync::Semaphore::new(1)),
        })
    }

    pub async fn text(&self, text: String) -> GuardDecision {
        let Ok(permit) = self.gate.clone().try_acquire_owned() else {
            return GuardDecision::Unavailable;
        };
        self.classify(json!(text), false, permit).await
    }

    pub async fn image(&self, data_url: String) -> GuardDecision {
        if self.model != Model::Nemotron || data_url.len() > MAX_IMAGE_DATA_URL_BYTES {
            return GuardDecision::Unavailable;
        }
        let Ok(permit) = self.gate.clone().try_acquire_owned() else {
            return GuardDecision::Unavailable;
        };
        tokio::time::timeout(Duration::from_secs(120), async {
            // A cancelled RPC leaves its owned execution permit with the
            // blocking decoder until that bounded decoder has finished.
            let decoded = tokio::task::spawn_blocking(move || {
                let valid = validate_image(&data_url).is_ok();
                (data_url, permit, valid)
            })
            .await;
            let Ok((data_url, permit, true)) = decoded else {
                return GuardDecision::Unavailable;
            };
            self.classify(json!([
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Classify this image under the fixed child-safety policy."}
            ]), true, permit).await
        })
        .await
        .unwrap_or(GuardDecision::Unavailable)
    }

    async fn classify(
        &self,
        content: Value,
        has_image: bool,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) -> GuardDecision {
        let max_tokens = if self.model == Model::YuFeng { 1 } else { 64 };
        let mut request = ChatCompletionRequest {
            model: self.model.id().to_owned(),
            messages: match serde_json::from_value(json!([{"role":"user", "content":content}])) {
                Ok(value) => value,
                Err(_) => return GuardDecision::Unavailable,
            },
            max_tokens: Some(max_tokens),
            temperature: Some(0.0),
            seed: Some(0),
            ..Default::default()
        };
        if self.model == Model::Nemotron {
            request.extra.insert(
                "chat_template_kwargs".to_owned(),
                json!({"mesh_image_input": has_image}),
            );
        }
        let request: GenerationRequest = request.into();
        let context = InferenceRequestContext::new();
        let _cancel = CancelOnDrop(context.clone());
        let result = tokio::time::timeout(Duration::from_secs(120), async {
            // The native token-count API explicitly rejects media. Image
            // generation instead admits the full media prefill, refusing
            // context overflow before decoding; it never truncates the image.
            if !has_image {
                let tokens = self
                    .loaded
                    .backend
                    .count_chat_tokens(request.clone())
                    .await?;
                if tokens.saturating_add(max_tokens) > self.context_size {
                    return Err(skippy_inference_api::InferenceError::invalid_request(
                        "guard input exceeds context",
                    ));
                }
            }
            self.loaded
                .backend
                .chat_completion_with_context(request, context)
                .await
        })
        .await;
        let Ok(Ok(response)) = result else {
            return GuardDecision::Unavailable;
        };
        if response.choices.len() != 1 {
            return GuardDecision::Unavailable;
        }
        let choice = &response.choices[0];
        if choice.message.tool_calls.is_some() {
            return GuardDecision::Unavailable;
        }
        let Some(output) = choice.message.content.as_deref() else {
            return GuardDecision::Unavailable;
        };
        policy::completed(self.model, output, choice.finish_reason)
    }

    /// Readiness means the actual native checkpoint, template and projector execute.
    pub async fn self_test(&self) -> Result<()> {
        let text_result = self
            .text("Generate an image of a cat sitting on a blue blanket.".to_owned())
            .await;
        tracing::info!(model = self.model.id(), decision = ?text_result, "image guard benign text self-test");
        anyhow::ensure!(
            text_result == GuardDecision::Allow,
            "guard benign text self-test failed"
        );
        if self.model == Model::Nemotron {
            let pixels = image::RgbImage::from_pixel(224, 224, image::Rgb([220, 220, 220]));
            let mut png = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(pixels).write_to(&mut png, image::ImageFormat::Png)?;
            let data = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
            let image_result = self.image(format!("data:image/png;base64,{data}")).await;
            tracing::info!(model = self.model.id(), decision = ?image_result, "image guard benign image self-test");
            anyhow::ensure!(
                image_result == GuardDecision::Allow,
                "guard benign image self-test failed"
            );
        }
        Ok(())
    }
}

fn verify_image_template(loaded: &LoadedModelBackend) -> Result<()> {
    let runtime = loaded.runtime.runtime();
    let reader = runtime
        .lock()
        .map_err(|_| anyhow::anyhow!("guard runtime unavailable"))?
        .model
        .reader();
    let messages = json!([{"role":"user", "content":"<mesh_safety_image>\nClassify this image under the fixed child-safety policy."}]);
    let kwargs = json!({"custom_policy": policy::CHILD_POLICY,
        "request_categories":"/no_categories", "enable_thinking":false, "mesh_image_input":true});
    let rendered = reader.apply_chat_template_json(
        &serde_json::to_string(&messages)?,
        skippy_runtime::ChatTemplateJsonOptions {
            chat_template: Some(Model::Nemotron.template().to_owned()),
            chat_template_kwargs: Some(serde_json::to_string(&kwargs)?),
            ..Default::default()
        },
    )?;
    let prompt = &rendered.prompt;
    let marker = prompt
        .find("<mesh_safety_image>")
        .ok_or_else(|| anyhow::anyhow!("image marker missing from guard template"))?;
    let policy = prompt
        .find(policy::CHILD_POLICY)
        .ok_or_else(|| anyhow::anyhow!("fixed policy missing from guard template"))?;
    anyhow::ensure!(
        marker < policy && prompt.matches("<mesh_safety_image>").count() == 1,
        "guard image template does not preserve trained image-first layout"
    );
    Ok(())
}

fn load_native_runtime(selection: Option<String>) -> Result<()> {
    let release = skippy_runtime_install::runtime_release_version();
    let bundle_dirs = skippy_runtime_install::discover_native_runtime_bundle_dirs(&[], release)?;
    let cache = skippy_runtime_install::default_native_runtime_cache()?;
    // A child process must load its own verified libraries; the host process's
    // already-loaded ABI symbols do not cross the plugin process boundary.
    skippy_api::native_runtime::load_local_native_runtime(
        &skippy_api::native_runtime::NativeRuntimeOptions {
            bundle_dirs,
            cache_dir: Some(cache.root().to_path_buf()),
            release: Some(release.to_owned()),
            selection,
        },
    )
}

struct CancelOnDrop(InferenceRequestContext);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub fn validate_image(data_url: &str) -> Result<()> {
    anyhow::ensure!(
        data_url.len() <= MAX_IMAGE_DATA_URL_BYTES,
        "image exceeds guard frame budget"
    );
    let (header, encoded) = data_url
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("invalid image data URL"))?;
    let expected = match header {
        "data:image/png;base64" => image::ImageFormat::Png,
        "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
        "data:image/webp;base64" => image::ImageFormat::WebP,
        _ => anyhow::bail!("unsupported image data URL"),
    };
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    anyhow::ensure!(
        image::guess_format(&bytes)? == expected,
        "image MIME mismatch"
    );
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), expected);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(4096);
    limits.max_image_height = Some(4096);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    let decoded = reader.decode()?;
    if decoded.color().has_alpha() {
        anyhow::ensure!(
            decoded.to_rgba8().pixels().all(|pixel| pixel[3] == 255),
            "transparent images are not supported by image screening"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn images_are_inline_bounded_and_decodable() {
        for value in [
            "https://example.com/cat.png",
            "file:///tmp/cat.png",
            "data:image/png;base64,AAAA",
        ] {
            assert!(validate_image(value).is_err());
        }
        assert!(validate_image(&"x".repeat(MAX_IMAGE_DATA_URL_BYTES + 1)).is_err());
    }

    #[test]
    fn transparency_cannot_change_the_displayed_classifier_view() {
        let data_url = |alpha| {
            let pixels = image::RgbaImage::from_pixel(1, 1, image::Rgba([20, 30, 40, alpha]));
            let mut png = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(pixels)
                .write_to(&mut png, image::ImageFormat::Png)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png.into_inner())
            )
        };
        assert!(validate_image(&data_url(255)).is_ok());
        assert!(validate_image(&data_url(0)).is_err());
        assert!(validate_image(&data_url(127)).is_err());
    }
}

//! Render-only discovery of the reasoning controls a loaded chat template reacts to.
//!
//! The probe renders one fixed chat through the *selected* chat template several
//! times, changing only the client-visible reasoning ask, and compares the
//! rendered prompts. It never runs a forward pass and never tokenizes the result.
//!
//! Every render goes through the same normalization and default resolution a real
//! request does (see [`crate::frontend::thinking_probe_options`]), so the summary
//! describes the effective request path rather than raw renderer kwargs. In
//! particular an effort is probed as thinking-on-plus-effort, which is what a
//! client's `reasoning_effort` actually resolves to.
//!
//! What the summary can and cannot say:
//!
//! * `enabled: true` means turning thinking on versus off changed the rendered
//!   prompt. It is not evidence that the model obeyed the control.
//! * `enabled: false` is inconclusive: the template may be indifferent, or a
//!   render may have failed. A control this probe could not observe is never
//!   advertised.
//! * `efforts` lists only the values that changed the prompt relative to a plain
//!   thinking-on render, and only the bounded set that was tried. Every other
//!   effort value, budget semantic, or control name stays unreported.
//!
//! The summary belongs to a (artifact, selected template, renderer) triple, so a
//! request- or server-level template override needs its own probe. The wire shape
//! is [`skippy_inference_api::thinking::ThinkingControls`], which rides on the
//! model object served by `/v1/models`; this module keeps the prompt hashes,
//! cases, and fingerprint for the diagnostic line only.

use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::thinking_probe_options;
use crate::runtime_state::RuntimeState;
use skippy_inference_api::ReasoningConfig;
use skippy_inference_api::ReasoningEffort;
use skippy_inference_api::thinking::ThinkingControls;
use skippy_runtime::ChatTemplateJsonOptions;
use skippy_runtime::StageModelReader;
use std::sync::Arc;
use std::sync::Mutex;

/// The chat every probe render uses.
///
/// Deliberately minimal: it must not depend on a system message, tools, or media
/// that a template may not support, so that the only variable between renders is
/// the reasoning ask under test.
const PROBE_MESSAGES_JSON: &str = r#"[{"role":"user","content":"Answer 2 + 2."}]"#;

/// Effort values the probe exercises, in the order they are reported.
///
/// Bounded on purpose: an untested value stays unreported rather than being
/// inferred from a neighbour. `xhigh` is included because it is the canonical
/// value on the Qwen3.8 model card; a template whose default already selects it
/// will still omit it from `efforts`, since it then matches a plain thinking-on
/// render.
const PROBE_EFFORTS: [ReasoningEffort; 4] = [
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::Xhigh,
];

/// Renders a chat template. Implemented by the native runtime in production and
/// by an in-memory template in tests.
pub trait ChatTemplateProbeRenderer {
    fn render(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<String, String>;
}

/// Inputs the probe needs before it can render anything.
pub struct ThinkingProbeInputs<'a> {
    /// Deployment request defaults: the probe resolves every case through them,
    /// exactly as serving would.
    pub defaults: &'a EmbeddedOpenAiRequestDefaults,
    pub model_id: &'a str,
    /// Artifact identity (source hash or package ref) from the stage config.
    pub artifact: Option<&'a str>,
    /// The selected server/request template override, if any.
    pub template_override: Option<&'a str>,
    /// Native renderer identity; see [`ThinkingProbeFingerprint::renderer`].
    pub renderer: &'a str,
}

/// Identity the probe ran against, for the diagnostic line only.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ThinkingProbeFingerprint {
    model_id: String,
    /// Artifact identity the runtime knows (source hash or package ref), when it has one.
    artifact: Option<String>,
    /// `embedded` for the artifact's own template, or `sha256:<hex>` of the
    /// template override the server selected.
    template: String,
    /// Native renderer identity, so a template-engine change re-keys the probe.
    renderer: String,
}

impl ThinkingProbeInputs<'_> {
    fn fingerprint(&self) -> ThinkingProbeFingerprint {
        ThinkingProbeFingerprint {
            model_id: self.model_id.to_string(),
            artifact: self.artifact.map(str::to_string),
            template: match self.template_override {
                Some(template) => format!("sha256:{}", sha256_hex(template.as_bytes())),
                None => "embedded".to_string(),
            },
            renderer: self.renderer.to_string(),
        }
    }
}

/// One client-visible reasoning ask the probe renders, and the name it reports under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeCase {
    /// The client asks for nothing, so the deployment default decides.
    Default,
    /// `reasoning.enabled = false`.
    Off,
    /// `reasoning.enabled = true`.
    On,
    /// `reasoning_effort = <value>`.
    Effort(ReasoningEffort),
}

impl ProbeCase {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Off => "off",
            Self::On => "on",
            Self::Effort(effort) => effort.as_str(),
        }
    }

    /// The client-visible reasoning ask this case sends into the effective path.
    fn ask(self) -> (Option<ReasoningConfig>, Option<ReasoningEffort>) {
        match self {
            Self::Default => (None, None),
            Self::Off => (Some(reasoning_enabled(false)), None),
            Self::On => (Some(reasoning_enabled(true)), None),
            Self::Effort(effort) => (None, Some(effort)),
        }
    }
}

fn reasoning_enabled(enabled: bool) -> ReasoningConfig {
    ReasoningConfig {
        enabled: Some(enabled),
        effort: None,
        max_tokens: None,
        exclude: None,
        extra: Default::default(),
    }
}

fn probe_cases() -> Vec<ProbeCase> {
    let mut cases = vec![ProbeCase::Default, ProbeCase::Off, ProbeCase::On];
    cases.extend(PROBE_EFFORTS.into_iter().map(ProbeCase::Effort));
    cases
}

/// What one render produced: the prompt hash, or why there is none.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeRender {
    Rendered {
        prompt_sha256: String,
        prompt_bytes: usize,
    },
    Error(String),
}

/// The full probe result: the diagnostic detail plus the client-facing summary.
pub struct ThinkingProbeReport {
    controls: ThinkingControls,
    fingerprint: ThinkingProbeFingerprint,
    observations: Vec<(ProbeCase, ProbeRender)>,
}

impl ThinkingProbeReport {
    /// The client-facing summary that rides `/v1/models`.
    pub fn controls(&self) -> &ThinkingControls {
        &self.controls
    }
}

/// Identity of the native chat-template renderer.
///
/// The template engine belongs to the patched native runtime, so its ABI version
/// is the field that re-keys a probe when the renderer changes.
pub fn native_renderer_identity() -> String {
    format!(
        "skippy-abi-{}.{}.{}",
        skippy_runtime::ABI_VERSION_MAJOR,
        skippy_runtime::ABI_VERSION_MINOR,
        skippy_runtime::ABI_VERSION_PATCH
    )
}

/// Renders every probe case and summarizes which controls the template reacted to.
pub fn run_thinking_probe(
    renderer: &impl ChatTemplateProbeRenderer,
    inputs: &ThinkingProbeInputs<'_>,
) -> ThinkingProbeReport {
    let observations = probe_cases()
        .into_iter()
        .map(|case| (case, render_case(renderer, inputs, case)))
        .collect::<Vec<_>>();
    ThinkingProbeReport {
        controls: summarize_controls(&observations),
        fingerprint: inputs.fingerprint(),
        observations,
    }
}

/// Turns the rendered prompts into the client-facing control summary.
///
/// Only positive, observed effects are advertised: a case that failed to render,
/// or that matched its reference render, contributes nothing.
fn summarize_controls(observations: &[(ProbeCase, ProbeRender)]) -> ThinkingControls {
    let hash = |case: ProbeCase| match find_render(observations, case) {
        Some(ProbeRender::Rendered { prompt_sha256, .. }) => Some(prompt_sha256.as_str()),
        _ => None,
    };
    let on = hash(ProbeCase::On);
    let off = hash(ProbeCase::Off);
    let enabled = matches!((on, off), (Some(on), Some(off)) if on != off);
    let efforts = on
        .map(|on| {
            PROBE_EFFORTS
                .into_iter()
                .filter(|effort| hash(ProbeCase::Effort(*effort)).is_some_and(|hash| hash != on))
                .map(|effort| effort.as_str().to_string())
                .collect()
        })
        .unwrap_or_default();
    ThinkingControls { enabled, efforts }
}

fn find_render(observations: &[(ProbeCase, ProbeRender)], case: ProbeCase) -> Option<&ProbeRender> {
    observations
        .iter()
        .find(|(candidate, _)| *candidate == case)
        .map(|(_, outcome)| outcome)
}

fn render_case(
    renderer: &impl ChatTemplateProbeRenderer,
    inputs: &ThinkingProbeInputs<'_>,
    case: ProbeCase,
) -> ProbeRender {
    let (reasoning, effort) = case.ask();
    let options = match thinking_probe_options(
        inputs.defaults,
        reasoning.as_ref(),
        effort,
        inputs.template_override,
    ) {
        Ok(options) => options,
        Err(error) => return ProbeRender::Error(format!("{error:?}")),
    };
    match renderer.render(PROBE_MESSAGES_JSON, options) {
        Ok(prompt) => ProbeRender::Rendered {
            prompt_sha256: sha256_hex(prompt.as_bytes()),
            prompt_bytes: prompt.len(),
        },
        Err(message) => ProbeRender::Error(message),
    }
}

/// Reports what the probe observed through the serving diagnostic channel.
///
/// The summary is public; the hashes and fingerprint exist only here, so a
/// deployment log can explain a surprising summary without putting probe detail
/// on `/v1/models`.
pub fn emit_probe_status(report: &ThinkingProbeReport) -> std::io::Result<()> {
    let cases = report
        .observations
        .iter()
        .map(|(case, outcome)| match outcome {
            ProbeRender::Rendered { prompt_sha256, .. } => format!(
                "{}:{}",
                case.name(),
                prompt_sha256.get(..12).unwrap_or(prompt_sha256)
            ),
            ProbeRender::Error(message) => format!("{}:error({message})", case.name()),
        })
        .collect::<Vec<_>>()
        .join(" ");
    skippy_events::diagnostics::emit(skippy_events::diagnostics::ServingDiagnostic::Status {
        message: format!(
            "skippy thinking probe: model_id={} template={} renderer={} enabled={} efforts=[{}] cases: {cases}",
            report.fingerprint.model_id,
            report.fingerprint.template,
            report.fingerprint.renderer,
            report.controls.enabled,
            report.controls.efforts.join(","),
        ),
    })
}

/// Runs the probe against an already-loaded runtime.
///
/// Returns `None` when the runtime lock is poisoned. A renderer error is recorded
/// on the case, not propagated, so one unsupported control cannot hide the rest.
pub fn probe_loaded_model(
    runtime: &Arc<Mutex<RuntimeState>>,
    inputs: &ThinkingProbeInputs<'_>,
) -> Option<ThinkingProbeReport> {
    let reader = runtime.lock().ok()?.model.reader();
    Some(run_thinking_probe(
        &NativeChatTemplateProbe { reader: &reader },
        inputs,
    ))
}

struct NativeChatTemplateProbe<'a> {
    reader: &'a StageModelReader,
}

impl ChatTemplateProbeRenderer for NativeChatTemplateProbe<'_> {
    fn render(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<String, String> {
        self.reader
            .apply_chat_template_json(messages_json, options)
            .map(|result| result.prompt)
            .map_err(|error| error.to_string())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::EmbeddedReasoningBudget;
    use std::sync::Mutex;

    /// Parses the serialized `chat_template_kwargs` a render received.
    fn kwargs(options: &ChatTemplateJsonOptions) -> serde_json::Value {
        options
            .chat_template_kwargs
            .as_deref()
            .and_then(|kwargs| serde_json::from_str(kwargs).ok())
            .unwrap_or(serde_json::Value::Null)
    }

    /// A template that renders both controls, exactly like a chat template would.
    struct ControlAwareTemplate;

    impl ChatTemplateProbeRenderer for ControlAwareTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            let kwargs = kwargs(&options);
            let effort = kwargs
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("-");
            Ok(format!(
                "<|thinking={}|><|effort={effort}|>{messages_json}",
                options.enable_thinking.unwrap_or(false)
            ))
        }
    }

    /// A template that ignores every reasoning control.
    struct ControlBlindTemplate;

    impl ChatTemplateProbeRenderer for ControlBlindTemplate {
        fn render(
            &self,
            messages_json: &str,
            _options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            Ok(messages_json.to_string())
        }
    }

    /// A template that only understands `reasoning_effort`, so the toggle does nothing.
    struct EffortOnlyTemplate;

    impl ChatTemplateProbeRenderer for EffortOnlyTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            let kwargs = kwargs(&options);
            let effort = kwargs
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("-");
            Ok(format!("<|effort={effort}|>{messages_json}"))
        }
    }

    /// A template that renders every effort except `medium`, which it maps to the
    /// plain thinking-on render.
    struct PartialEffortTemplate;

    impl ChatTemplateProbeRenderer for PartialEffortTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            let kwargs = kwargs(&options);
            let effort = kwargs
                .get("reasoning_effort")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("-");
            match effort {
                "medium" | "-" => Ok(format!("<|thinking=on|>{messages_json}")),
                other => Ok(format!("<|effort={other}|>{messages_json}")),
            }
        }
    }

    /// A template that rejects turning thinking on, as an unknown-variable error.
    struct ThinkingRejectingTemplate;

    impl ChatTemplateProbeRenderer for ThinkingRejectingTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            if options.enable_thinking == Some(true) {
                return Err("enable_thinking is not a known variable".to_string());
            }
            Ok(messages_json.to_string())
        }
    }

    fn defaults() -> EmbeddedOpenAiRequestDefaults {
        EmbeddedOpenAiRequestDefaults::default()
    }

    fn inputs(defaults: &EmbeddedOpenAiRequestDefaults) -> ThinkingProbeInputs<'_> {
        ThinkingProbeInputs {
            defaults,
            model_id: "test/model",
            artifact: Some("abc123"),
            template_override: None,
            renderer: "skippy-abi-0.1.66",
        }
    }

    #[test]
    fn a_control_blind_template_offers_no_controls() {
        let defaults = defaults();
        let report = run_thinking_probe(&ControlBlindTemplate, &inputs(&defaults));
        assert_eq!(
            *report.controls(),
            ThinkingControls {
                enabled: false,
                efforts: vec![],
            }
        );
    }

    #[test]
    fn a_control_aware_template_offers_the_toggle_and_every_distinguished_effort() {
        let defaults = defaults();
        let report = run_thinking_probe(&ControlAwareTemplate, &inputs(&defaults));
        assert_eq!(
            *report.controls(),
            ThinkingControls {
                enabled: true,
                efforts: vec![
                    "low".to_string(),
                    "medium".to_string(),
                    "high".to_string(),
                    "xhigh".to_string(),
                ],
            }
        );
    }

    #[test]
    fn an_effort_only_template_offers_effort_without_a_toggle() {
        let defaults = defaults();
        let report = run_thinking_probe(&EffortOnlyTemplate, &inputs(&defaults));
        assert!(!report.controls().enabled);
        assert_eq!(
            report.controls().efforts,
            vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "xhigh".to_string()
            ]
        );
    }

    #[test]
    fn an_effort_that_renders_like_a_plain_thinking_on_request_is_not_offered() {
        let defaults = defaults();
        let report = run_thinking_probe(&PartialEffortTemplate, &inputs(&defaults));
        assert_eq!(report.controls().efforts, vec!["low", "high", "xhigh"]);
    }

    #[test]
    fn a_control_that_cannot_render_is_not_advertised() {
        let defaults = defaults();
        let report = run_thinking_probe(&ThinkingRejectingTemplate, &inputs(&defaults));
        assert_eq!(
            *report.controls(),
            ThinkingControls {
                enabled: false,
                efforts: vec![],
            }
        );
        assert!(report.observations.iter().any(|(_, outcome)| matches!(
            outcome,
            ProbeRender::Error(message) if message.contains("enable_thinking")
        )));
    }

    #[test]
    fn the_effort_cases_use_the_effective_client_request_path() {
        // The gap this guards: a raw probe would send `reasoning_effort` with the
        // toggle unset, but a client's `reasoning_effort: low` resolves to
        // thinking-on *and* the effort kwarg.
        struct RecordingTemplate(Mutex<Vec<ChatTemplateJsonOptions>>);

        impl ChatTemplateProbeRenderer for RecordingTemplate {
            fn render(
                &self,
                _messages_json: &str,
                options: ChatTemplateJsonOptions,
            ) -> Result<String, String> {
                self.0
                    .lock()
                    .expect("recording lock poisoned")
                    .push(options);
                Ok("render".to_string())
            }
        }

        let defaults = EmbeddedOpenAiRequestDefaults {
            chat_template_kwargs: Some(serde_json::json!({ "deployment": "x" })),
            reasoning_budget: Some(EmbeddedReasoningBudget::Tokens(1024)),
            ..EmbeddedOpenAiRequestDefaults::default()
        };
        let recorder = RecordingTemplate(Mutex::new(Vec::new()));
        run_thinking_probe(&recorder, &inputs(&defaults));
        let recorded = recorder.0.lock().expect("recording lock poisoned").clone();

        let effort_case = &recorded[3];
        assert_eq!(
            effort_case.enable_thinking,
            Some(true),
            "an effort ask must also turn thinking on"
        );
        let effort_kwargs = kwargs(effort_case);
        assert_eq!(effort_kwargs["reasoning_effort"], "low");
        // Deployment defaults are part of the effective path for every case.
        assert_eq!(effort_kwargs["deployment"], "x");
        assert_eq!(effort_kwargs["thinking_budget"], 1024);
        // A deployment budget is part of the effective path: it turns a silent
        // request's thinking on, exactly as serving would.
        assert_eq!(recorded[0].enable_thinking, Some(true));

        // With no deployment reasoning configuration, the built-in default is off.
        let plain = EmbeddedOpenAiRequestDefaults::default();
        let recorder = RecordingTemplate(Mutex::new(Vec::new()));
        run_thinking_probe(&recorder, &inputs(&plain));
        let plain_recorded = recorder.0.lock().expect("recording lock poisoned").clone();
        assert_eq!(plain_recorded[0].enable_thinking, Some(false));
    }

    #[test]
    fn the_fingerprint_rekeys_on_the_selected_template_and_the_renderer() {
        let defaults = defaults();
        let embedded = run_thinking_probe(&ControlBlindTemplate, &inputs(&defaults));
        assert_eq!(embedded.fingerprint.template, "embedded");

        let overridden = run_thinking_probe(
            &ControlBlindTemplate,
            &ThinkingProbeInputs {
                template_override: Some("{% for m in messages %}{{ m }}{% endfor %}"),
                renderer: "skippy-abi-0.1.67",
                ..inputs(&defaults)
            },
        );
        assert!(overridden.fingerprint.template.starts_with("sha256:"));
        assert_ne!(
            embedded.fingerprint.template,
            overridden.fingerprint.template
        );
        assert_ne!(
            embedded.fingerprint.renderer,
            overridden.fingerprint.renderer
        );
    }

    #[test]
    fn the_native_renderer_identity_names_the_abi_version() {
        let identity = native_renderer_identity();
        assert!(identity.starts_with("skippy-abi-"), "{identity}");
        assert!(identity.ends_with(&format!(
            "{}.{}.{}",
            skippy_runtime::ABI_VERSION_MAJOR,
            skippy_runtime::ABI_VERSION_MINOR,
            skippy_runtime::ABI_VERSION_PATCH
        )));
    }

    #[test]
    fn the_probe_renders_the_same_chat_for_every_case() {
        // The only variable between renders must be the reasoning ask.
        struct RecordingTemplate(Mutex<Vec<String>>);

        impl ChatTemplateProbeRenderer for RecordingTemplate {
            fn render(
                &self,
                messages_json: &str,
                _options: ChatTemplateJsonOptions,
            ) -> Result<String, String> {
                self.0
                    .lock()
                    .expect("recording lock poisoned")
                    .push(messages_json.to_string());
                Ok(messages_json.to_string())
            }
        }

        let defaults = defaults();
        let recorder = RecordingTemplate(Mutex::new(Vec::new()));
        let report = run_thinking_probe(&recorder, &inputs(&defaults));
        let seen = recorder.0.lock().expect("recording lock poisoned").clone();
        assert_eq!(seen.len(), report.observations.len());
        assert!(seen.iter().all(|chat| chat == PROBE_MESSAGES_JSON));
    }
}

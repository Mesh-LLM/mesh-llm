//! Render-only discovery of the reasoning controls a loaded chat template reacts to.
//!
//! The probe renders one fixed chat through the *selected* chat template several
//! times, changing only the reasoning-control inputs, and compares the rendered
//! prompts. It never runs a forward pass and never tokenizes the result.
//!
//! What the report can and cannot say:
//!
//! * `changes_prompt` means the tested input changed the rendered prompt. It is
//!   not evidence that the model obeyed the control.
//! * `same_prompt` is inconclusive: another layer may apply the control outside
//!   the template.
//! * Only the inputs named in `cases` were tried. Every other effort value,
//!   budget semantic, or control name stays unknown, and any value the model
//!   template accepts dynamically is invisible here.
//!
//! The report is a per-(artifact, selected template, renderer) observation, so a
//! request- or server-level template override needs its own probe. The wire shape
//! is [`skippy_inference_api::thinking::ThinkingControls`], which rides on the
//! model object served by `/v1/models`.

use crate::runtime_state::RuntimeState;
use skippy_inference_api::thinking::{
    ControlObedience, ThinkingControlCase, ThinkingControlDifference, ThinkingControls,
    ThinkingControlsFingerprint, ThinkingEffect, ThinkingEvidence, ThinkingRenderOutcome,
};
use skippy_runtime::ChatReasoningFormat;
use skippy_runtime::ChatTemplateJsonOptions;
use skippy_runtime::StageModelReader;
use std::sync::Arc;
use std::sync::Mutex;

/// The chat every probe render uses.
///
/// Deliberately minimal: it must not depend on a system message, tools, or media
/// that a template may not support, so that the only variable between renders is
/// the reasoning control under test.
const PROBE_MESSAGES_JSON: &str = r#"[{"role":"user","content":"Answer 2 + 2."}]"#;

/// Effort values the probe exercises. Bounded on purpose: an untested value stays
/// unknown rather than being inferred from a neighbour.
pub const PROBE_EFFORTS: [&str; 3] = ["low", "medium", "high"];

const UNTESTED_CONTROLS: &str = "every effort value outside low/medium/high, budget semantics, and control names not listed in cases";

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
    pub model_id: &'a str,
    /// Artifact identity (source hash or package ref) from the stage config.
    pub artifact: Option<&'a str>,
    /// The selected server/request template override, if any.
    pub template_override: Option<&'a str>,
    pub use_jinja: bool,
    /// Native renderer identity; see [`ThinkingControlsFingerprint::renderer`].
    pub renderer: &'a str,
}

impl ThinkingProbeInputs<'_> {
    fn fingerprint(&self) -> ThinkingControlsFingerprint {
        ThinkingControlsFingerprint {
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

/// Renders every probe case and records which inputs changed the prompt.
pub fn run_thinking_probe(
    renderer: &impl ChatTemplateProbeRenderer,
    inputs: &ThinkingProbeInputs<'_>,
) -> ThinkingControls {
    let cases = probe_cases()
        .into_iter()
        .map(|case| {
            let outcome = render_case(renderer, inputs, &case);
            ThinkingControlCase { outcome, ..case }
        })
        .collect::<Vec<_>>();
    let differences = probe_differences(&cases);
    ThinkingControls {
        fingerprint: inputs.fingerprint(),
        evidence: ThinkingEvidence::RenderedPromptOnly,
        model_obeys_control: ControlObedience::Unknown,
        untested: UNTESTED_CONTROLS.to_string(),
        cases,
        differences,
    }
}

/// Reports what the probe observed through the serving diagnostic channel.
///
/// The report is also returned to the caller; this line exists so the observation
/// is visible in a deployment log, independently of any consumer.
pub fn emit_probe_status(report: &ThinkingControls) -> std::io::Result<()> {
    let observed = report
        .observations()
        .map(|(case, effect)| format!("{case}={}", effect_name(effect)))
        .collect::<Vec<_>>()
        .join(" ");
    skippy_events::diagnostics::emit(skippy_events::diagnostics::ServingDiagnostic::Status {
        message: format!(
            "skippy thinking probe: model_id={} template={} renderer={} evidence=rendered_prompt_only {observed}",
            report.fingerprint.model_id, report.fingerprint.template, report.fingerprint.renderer,
        ),
    })
}

fn effect_name(effect: ThinkingEffect) -> &'static str {
    match effect {
        ThinkingEffect::ChangesPrompt => "changes_prompt",
        ThinkingEffect::SamePrompt => "same_prompt",
        ThinkingEffect::Unavailable => "unavailable",
    }
}

/// Runs the probe against an already-loaded runtime.
///
/// Returns `None` when the runtime lock is poisoned. A renderer error is recorded
/// on the case, not propagated, so one unsupported control cannot hide the rest.
pub fn probe_loaded_model(
    runtime: &Arc<Mutex<RuntimeState>>,
    inputs: &ThinkingProbeInputs<'_>,
) -> Option<ThinkingControls> {
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

fn probe_cases() -> Vec<ThinkingControlCase> {
    let unrendered = || ThinkingRenderOutcome::Error {
        message: "not rendered".to_string(),
    };
    let mut cases = vec![
        ThinkingControlCase {
            name: ThinkingControls::BASELINE_CASE.to_string(),
            enable_thinking: None,
            reasoning_effort: None,
            outcome: unrendered(),
        },
        ThinkingControlCase {
            name: "off".to_string(),
            enable_thinking: Some(false),
            reasoning_effort: None,
            outcome: unrendered(),
        },
        ThinkingControlCase {
            name: "on".to_string(),
            enable_thinking: Some(true),
            reasoning_effort: None,
            outcome: unrendered(),
        },
    ];
    cases.extend(
        PROBE_EFFORTS
            .iter()
            .copied()
            .map(|effort| ThinkingControlCase {
                name: effort.to_string(),
                enable_thinking: None,
                reasoning_effort: Some(effort.to_string()),
                outcome: unrendered(),
            }),
    );
    cases
}

fn render_case(
    renderer: &impl ChatTemplateProbeRenderer,
    inputs: &ThinkingProbeInputs<'_>,
    case: &ThinkingControlCase,
) -> ThinkingRenderOutcome {
    let chat_template_kwargs = case
        .reasoning_effort
        .as_deref()
        .map(|effort| serde_json::json!({ "reasoning_effort": effort }).to_string());
    let options = ChatTemplateJsonOptions {
        add_assistant: true,
        enable_thinking: case.enable_thinking,
        // Constant across cases: the parser name must not become a second variable.
        reasoning_format: Some(ChatReasoningFormat::Auto),
        chat_template_kwargs,
        tools_json: None,
        tool_choice_json: None,
        parallel_tool_calls: true,
        chat_template: inputs.template_override.map(str::to_string),
        use_jinja: inputs.use_jinja,
        grammar: None,
        json_schema: None,
        skip_chat_parsing: false,
    };
    match renderer.render(PROBE_MESSAGES_JSON, options) {
        Ok(prompt) => ThinkingRenderOutcome::Rendered {
            prompt_sha256: sha256_hex(prompt.as_bytes()),
            prompt_bytes: prompt.len(),
        },
        Err(message) => ThinkingRenderOutcome::Error { message },
    }
}

fn probe_differences(cases: &[ThinkingControlCase]) -> Vec<ThinkingControlDifference> {
    let mut pairs = cases
        .iter()
        .filter(|case| case.name != ThinkingControls::BASELINE_CASE)
        .map(|case| {
            (
                case.name.clone(),
                ThinkingControls::BASELINE_CASE.to_string(),
            )
        })
        .collect::<Vec<_>>();
    // The off/on pair shows whether the control is directional; the effort pairs
    // show whether the template distinguishes effort values at all.
    pairs.extend(
        [
            ("off", "on"),
            ("low", "medium"),
            ("low", "high"),
            ("medium", "high"),
        ]
        .into_iter()
        .map(|(left, right)| (left.to_string(), right.to_string())),
    );
    pairs
        .into_iter()
        .map(|(left, right)| ThinkingControlDifference {
            effect: effect_between(cases, &left, &right),
            left,
            right,
        })
        .collect()
}

fn effect_between(cases: &[ThinkingControlCase], left: &str, right: &str) -> ThinkingEffect {
    let (Some(left), Some(right)) = (find_case(cases, left), find_case(cases, right)) else {
        return ThinkingEffect::Unavailable;
    };
    match (&left.outcome, &right.outcome) {
        (
            ThinkingRenderOutcome::Rendered {
                prompt_sha256: left,
                ..
            },
            ThinkingRenderOutcome::Rendered {
                prompt_sha256: right,
                ..
            },
        ) if left == right => ThinkingEffect::SamePrompt,
        (ThinkingRenderOutcome::Rendered { .. }, ThinkingRenderOutcome::Rendered { .. }) => {
            ThinkingEffect::ChangesPrompt
        }
        _ => ThinkingEffect::Unavailable,
    }
}

fn find_case<'a>(cases: &'a [ThinkingControlCase], name: &str) -> Option<&'a ThinkingControlCase> {
    cases.iter().find(|case| case.name == name)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A template that renders `enable_thinking` and `reasoning_effort` into the
    /// prompt exactly like a chat template would.
    struct ControlAwareTemplate;

    impl ChatTemplateProbeRenderer for ControlAwareTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            let thinking = match options.enable_thinking {
                Some(value) => format!("<|thinking={value}|>"),
                None => "<|thinking=template-default|>".to_string(),
            };
            let effort = options
                .chat_template_kwargs
                .as_deref()
                .map(|kwargs| format!("<|effort={kwargs}|>"))
                .unwrap_or_default();
            Ok(format!("{thinking}{effort}{messages_json}"))
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

    /// A template that only understands `reasoning_effort`.
    struct EffortOnlyTemplate;

    impl ChatTemplateProbeRenderer for EffortOnlyTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            let effort = options
                .chat_template_kwargs
                .as_deref()
                .map(|kwargs| format!("<|effort={kwargs}|>"))
                .unwrap_or_default();
            Ok(format!("{effort}{messages_json}"))
        }
    }

    /// A template that errors on the `enable_thinking` control.
    struct ThinkingRejectingTemplate;

    impl ChatTemplateProbeRenderer for ThinkingRejectingTemplate {
        fn render(
            &self,
            messages_json: &str,
            options: ChatTemplateJsonOptions,
        ) -> Result<String, String> {
            if options.enable_thinking.is_some() {
                return Err("enable_thinking is not a known variable".to_string());
            }
            Ok(messages_json.to_string())
        }
    }

    fn inputs() -> ThinkingProbeInputs<'static> {
        ThinkingProbeInputs {
            model_id: "test/model",
            artifact: Some("abc123"),
            template_override: None,
            use_jinja: true,
            renderer: "skippy-abi-0.1.66",
        }
    }

    fn effect(report: &ThinkingControls, left: &str, right: &str) -> ThinkingEffect {
        report
            .differences
            .iter()
            .find(|difference| difference.left == left && difference.right == right)
            .unwrap_or_else(|| panic!("no difference recorded for {left}/{right}"))
            .effect
    }

    #[test]
    fn a_control_blind_template_reports_every_tested_input_unchanged() {
        let report = run_thinking_probe(&ControlBlindTemplate, &inputs());
        assert_eq!(report.cases.len(), 6);
        for case in &report.cases {
            assert!(
                matches!(case.outcome, ThinkingRenderOutcome::Rendered { .. }),
                "case {} did not render",
                case.name
            );
        }
        for (name, effect) in report.observations() {
            assert_eq!(effect, ThinkingEffect::SamePrompt, "case {name}");
        }
        assert_eq!(effect(&report, "off", "on"), ThinkingEffect::SamePrompt);
    }

    #[test]
    fn a_control_aware_template_reports_the_directional_changes() {
        let report = run_thinking_probe(&ControlAwareTemplate, &inputs());
        assert_eq!(
            effect(&report, "off", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::ChangesPrompt
        );
        assert_eq!(
            effect(&report, "on", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::ChangesPrompt
        );
        assert_eq!(effect(&report, "off", "on"), ThinkingEffect::ChangesPrompt);
        assert_eq!(
            effect(&report, "low", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::ChangesPrompt
        );
    }

    #[test]
    fn an_effort_only_template_distinguishes_effort_values_from_the_baseline() {
        let report = run_thinking_probe(&EffortOnlyTemplate, &inputs());
        assert_eq!(
            effect(&report, "off", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::SamePrompt
        );
        assert_eq!(
            effect(&report, "on", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::SamePrompt
        );
        for effort in PROBE_EFFORTS {
            assert_eq!(
                effect(&report, effort, ThinkingControls::BASELINE_CASE),
                ThinkingEffect::ChangesPrompt,
                "effort={effort}"
            );
        }
        assert_eq!(
            effect(&report, "low", "high"),
            ThinkingEffect::ChangesPrompt
        );
    }

    #[test]
    fn a_rejected_control_is_unavailable_rather_than_the_same_prompt() {
        let report = run_thinking_probe(&ThinkingRejectingTemplate, &inputs());
        assert_eq!(
            effect(&report, "off", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::Unavailable
        );
        assert_eq!(effect(&report, "off", "on"), ThinkingEffect::Unavailable);
        // The effort cases still rendered, so they keep a real observation.
        assert_eq!(
            effect(&report, "low", ThinkingControls::BASELINE_CASE),
            ThinkingEffect::SamePrompt
        );
        assert!(report.cases.iter().any(|case| matches!(
            &case.outcome,
            ThinkingRenderOutcome::Error { message } if message.contains("enable_thinking")
        )));
    }

    #[test]
    fn the_report_never_claims_more_than_a_render_can_show() {
        let report = run_thinking_probe(&ControlAwareTemplate, &inputs());
        assert_eq!(report.evidence, ThinkingEvidence::RenderedPromptOnly);
        assert_eq!(report.model_obeys_control, ControlObedience::Unknown);
        assert!(!report.untested.is_empty());
    }

    #[test]
    fn the_fingerprint_rekeys_on_the_selected_template_and_the_renderer() {
        let embedded = run_thinking_probe(&ControlBlindTemplate, &inputs());
        assert_eq!(embedded.fingerprint.template, "embedded");

        let overridden = run_thinking_probe(
            &ControlBlindTemplate,
            &ThinkingProbeInputs {
                template_override: Some("{% for m in messages %}{{ m }}{% endfor %}"),
                renderer: "skippy-abi-0.1.67",
                ..inputs()
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
        // The only variable between renders must be the reasoning control.
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

        let recorder = RecordingTemplate(Mutex::new(Vec::new()));
        let report = run_thinking_probe(&recorder, &inputs());
        let seen = recorder.0.lock().expect("recording lock poisoned").clone();
        assert_eq!(seen.len(), report.cases.len());
        assert!(seen.iter().all(|chat| chat == PROBE_MESSAGES_JSON));
    }
}

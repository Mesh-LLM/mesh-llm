use mesh_llm_config::{
    ConfigApplyMode, ConfigConditionValue, ConfigControlAvailabilitySource, ConfigControlBehavior,
    ConfigDisabledWritePolicy, ConfigOptionsSource, ConfigPath, ConfigRestartScope,
    ConfigSettingOwner, ConfigSettingSchema, ConfigSupportState, ConfigValueSchema,
    ConfigVisibility,
};

pub(crate) fn runtime_source_setting(
    path: &str,
    source: ConfigOptionsSource,
) -> ConfigSettingSchema {
    ConfigSettingSchema {
        path: ConfigPath::parse_rendered(path).expect("test path should parse"),
        alias_policy: Default::default(),
        owner: ConfigSettingOwner::BuiltIn,
        value_schema: ConfigValueSchema::String,
        support: ConfigSupportState::Supported,
        control_surfaces: Vec::new(),
        apply_mode: ConfigApplyMode::StaticOnLoad,
        restart_scope: ConfigRestartScope::None,
        visibility: ConfigVisibility::User,
        constraints: Vec::new(),
        description: None,
        presentation: None,
        control_behavior: Some(ConfigControlBehavior {
            options_source: Some(source),
            ..ConfigControlBehavior::default()
        }),
    }
}

pub(crate) struct RuntimeOptionSpec<'a> {
    source: ConfigOptionsSource,
    value: &'a str,
    label: &'a str,
    note: Option<&'a str>,
    disabled: bool,
    reason: Option<&'a str>,
}

impl<'a> RuntimeOptionSpec<'a> {
    pub(crate) fn enabled(source: ConfigOptionsSource, value: &'a str, label: &'a str) -> Self {
        Self {
            source,
            value,
            label,
            note: None,
            disabled: false,
            reason: None,
        }
    }

    pub(crate) fn with_note(self, note: &'a str) -> Self {
        Self {
            note: Some(note),
            ..self
        }
    }

    pub(crate) fn disabled_with_reason(self, reason: &'a str) -> Self {
        Self {
            disabled: true,
            reason: Some(reason),
            ..self
        }
    }
}

pub(crate) fn runtime_option(
    spec: RuntimeOptionSpec<'_>,
) -> crate::api::routes::runtime::ConfigControlOption {
    crate::api::routes::runtime::ConfigControlOption {
        value: ConfigConditionValue::String(spec.value.to_string()),
        label: Some(spec.label.to_string()),
        note: spec.note.map(str::to_string),
        disabled: spec.disabled,
        reason: spec.reason.map(str::to_string),
        source: spec.source,
    }
}

#[test]
fn runtime_config_control_state_builder_enables_two_gpu_choices_with_stable_labels_and_values() {
    let setting =
        runtime_source_setting("defaults.hardware.device", ConfigOptionsSource::RuntimeGpus);
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            gpus: crate::api::routes::runtime_control_state::RuntimeOptionsState::Options(vec![
                runtime_option(
                    RuntimeOptionSpec::enabled(
                        ConfigOptionsSource::RuntimeGpus,
                        "CUDA0",
                        "NVIDIA A100 (CUDA0)",
                    )
                    .with_note("80.0 GiB VRAM"),
                ),
                runtime_option(
                    RuntimeOptionSpec::enabled(
                        ConfigOptionsSource::RuntimeGpus,
                        "CUDA1",
                        "NVIDIA H100 (CUDA1)",
                    )
                    .with_note("80.0 GiB VRAM"),
                ),
            ]),
            ..Default::default()
        },
    );

    let entry = payload
        .settings
        .get("defaults.hardware.device")
        .expect("gpu overlay should exist");
    let options = entry.options.as_ref().expect("gpu options should exist");

    assert!(entry.enabled);
    assert_eq!(entry.source, ConfigControlAvailabilitySource::Runtime);
    assert_eq!(
        entry.write_policy,
        ConfigDisabledWritePolicy::PreserveExisting
    );
    assert_eq!(options.len(), 2);
    assert_eq!(
        options[0].value,
        ConfigConditionValue::String("CUDA0".to_string())
    );
    assert_eq!(options[0].label.as_deref(), Some("NVIDIA A100 (CUDA0)"));
    assert_eq!(
        options[1].value,
        ConfigConditionValue::String("CUDA1".to_string())
    );
    assert_eq!(options[1].label.as_deref(), Some("NVIDIA H100 (CUDA1)"));
}

#[test]
fn runtime_config_control_state_builder_populates_native_backend_choices() {
    let setting = runtime_source_setting(
        "plugin.demo.settings.runtime_kind",
        ConfigOptionsSource::RuntimeNativeBackends,
    );
    let payload = crate::api::routes::runtime_control_state::build_runtime_control_state_payload(
        [&setting],
        &crate::api::routes::runtime_control_state::RuntimeControlStateSources {
            native_backends:
                crate::api::routes::runtime_control_state::RuntimeOptionsState::Options(vec![
                    runtime_option(RuntimeOptionSpec::enabled(
                        ConfigOptionsSource::RuntimeNativeBackends,
                        "cpu",
                        "CPU",
                    )),
                    runtime_option(RuntimeOptionSpec::enabled(
                        ConfigOptionsSource::RuntimeNativeBackends,
                        "metal",
                        "Metal",
                    )),
                ]),
            ..Default::default()
        },
    );

    let options = payload
        .settings
        .get("plugin.demo.settings.runtime_kind")
        .and_then(|entry| entry.options.as_ref())
        .expect("backend options should exist");
    assert_eq!(options.len(), 2);
    assert_eq!(
        options[0].source,
        ConfigOptionsSource::RuntimeNativeBackends
    );
    assert_eq!(
        options[1].value,
        ConfigConditionValue::String("metal".to_string())
    );
}

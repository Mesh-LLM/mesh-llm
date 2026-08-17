use super::*;

pub(super) fn set_numeric(
    setting: &mut ConfigSettingSchema,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    unit: Option<&str>,
) {
    let numeric = numeric_control_mut(setting);
    numeric.min = min;
    numeric.max = max;
    numeric.step = step;
    numeric.unit = unit.map(str::to_string);
}

pub(super) fn set_text_format(setting: &mut ConfigSettingSchema, text_format: ConfigTextFormat) {
    control_behavior_mut(setting).text_format = Some(text_format);
}

pub(super) fn set_static_options(setting: &mut ConfigSettingSchema) {
    control_behavior_mut(setting).options_source = Some(ConfigOptionsSource::Static);
}

pub(super) fn set_runtime_gpu_options(setting: &mut ConfigSettingSchema) {
    control_behavior_mut(setting).options_source = Some(ConfigOptionsSource::RuntimeGpus);
}

pub(super) fn set_static_unavailable(setting: &mut ConfigSettingSchema, reason: &str) {
    control_behavior_mut(setting).availability = Some(ConfigControlAvailability {
        enabled: false,
        reason: Some(reason.to_string()),
        note: None,
        source: ConfigControlAvailabilitySource::Static,
    });
}

pub(super) fn set_static_unavailable_with_note(
    setting: &mut ConfigSettingSchema,
    reason: &str,
    note: &str,
) {
    control_behavior_mut(setting).availability = Some(ConfigControlAvailability {
        enabled: false,
        reason: Some(reason.to_string()),
        note: Some(note.to_string()),
        source: ConfigControlAvailabilitySource::Static,
    });
}

pub(super) fn set_write_policy(
    setting: &mut ConfigSettingSchema,
    policy: ConfigDisabledWritePolicy,
) {
    control_behavior_mut(setting).write_policy = Some(policy);
}

pub(super) fn push_enable_when(
    setting: &mut ConfigSettingSchema,
    condition: ConfigControlCondition,
) {
    control_behavior_mut(setting).enable_when.push(condition);
}

pub(super) fn push_dependency_disable(
    setting: &mut ConfigSettingSchema,
    condition: ConfigControlCondition,
    reason: String,
) {
    push_disable(
        setting,
        condition,
        reason,
        ConfigDisabledWritePolicy::OmitWhenDisabled,
    );
}

pub(super) fn push_disable(
    setting: &mut ConfigSettingSchema,
    condition: ConfigControlCondition,
    reason: String,
    write_policy: ConfigDisabledWritePolicy,
) {
    control_behavior_mut(setting)
        .disable_when
        .push(ConfigConditionalDisable {
            condition,
            reason,
            note: None,
            write_policy,
        });
}

pub(super) fn push_constraint(setting: &mut ConfigSettingSchema, constraint: ConfigConstraint) {
    setting.constraints.push(constraint);
}

pub(super) fn push_non_empty_constraint(setting: &mut ConfigSettingSchema) {
    push_constraint(setting, ConfigConstraint::NonEmpty);
}

pub(super) fn push_requires_constraint(setting: &mut ConfigSettingSchema, sibling_path: &str) {
    push_constraint(
        setting,
        ConfigConstraint::Requires {
            path: schema_path(sibling_path),
        },
    );
}

pub(super) fn push_range_constraint(
    setting: &mut ConfigSettingSchema,
    min: Option<String>,
    max: Option<String>,
) {
    push_constraint(setting, ConfigConstraint::Range { min, max });
}

pub(super) fn push_allowed_pattern_constraint(
    setting: &mut ConfigSettingSchema,
    pattern: impl AsRef<str>,
) {
    push_constraint(
        setting,
        ConfigConstraint::AllowedPattern {
            pattern: pattern.as_ref().to_string(),
        },
    );
}

pub(super) fn equals_string_condition(path: &str, value: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Equals,
        values: vec![ConfigConditionValue::String(value.to_string())],
    }
}

pub(super) fn equals_bool_condition(path: &str, value: bool) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Equals,
        values: vec![ConfigConditionValue::Bool(value)],
    }
}

pub(super) fn in_condition(
    path: &str,
    values: impl IntoIterator<Item = ConfigConditionValue>,
) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::In,
        values: values.into_iter().collect(),
    }
}

pub(super) fn not_in_condition(
    path: &str,
    values: impl IntoIterator<Item = ConfigConditionValue>,
) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::NotIn,
        values: values.into_iter().collect(),
    }
}

pub(super) fn present_condition(path: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Present,
        values: Vec::new(),
    }
}

pub(super) fn absent_condition(path: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Absent,
        values: Vec::new(),
    }
}

pub(super) fn falsy_condition(path: &str) -> ConfigControlCondition {
    ConfigControlCondition {
        path: schema_path(path),
        operator: ConfigConditionOperator::Falsy,
        values: Vec::new(),
    }
}

pub(super) fn setting_path(setting: &ConfigSettingSchema) -> String {
    setting.path.render()
}

fn control_behavior_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigControlBehavior {
    setting
        .control_behavior
        .get_or_insert_with(ConfigControlBehavior::default)
}

fn numeric_control_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigNumericControl {
    control_behavior_mut(setting)
        .numeric
        .get_or_insert_with(ConfigNumericControl::default)
}

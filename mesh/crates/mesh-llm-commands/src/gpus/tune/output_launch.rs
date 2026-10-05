use super::*;

pub(crate) fn build_launch_preview(
    prepared: &crate::gpus::tune_apply::PreparedTunePlan,
    settings: &[TuneRenderedSetting],
    status: TuneTargetStatus,
) -> Option<TuneLaunchPreview> {
    if !matches!(status, TuneTargetStatus::Ready | TuneTargetStatus::Written) {
        return None;
    }
    let mut argv = vec![
        "mesh-llm".to_string(),
        "serve".to_string(),
        "--model".to_string(),
        prepared.target.resolved_path.display().to_string(),
    ];
    if let Some(ctx_size) = setting_context_size(settings) {
        argv.push("--ctx-size".to_string());
        argv.push(ctx_size.to_string());
    }
    if let Some(device) = setting_device(settings) {
        argv.push("--device".to_string());
        argv.push(device);
    }
    Some(TuneLaunchPreview {
        shell: argv
            .iter()
            .map(|arg| shell_quote(arg))
            .collect::<Vec<_>>()
            .join(" "),
        config_settings: settings
            .iter()
            .filter(|setting| {
                matches!(
                    setting.status,
                    TuneRenderedSettingStatus::Applied | TuneRenderedSettingStatus::Preserved
                )
            })
            .filter_map(|setting| {
                let value = setting.value.clone()?;
                Some(TuneLaunchSetting {
                    config_path: setting.config_path.clone(),
                    field: setting.field,
                    value,
                })
            })
            .collect(),
        report_only: settings
            .iter()
            .filter(|setting| setting.status == TuneRenderedSettingStatus::ReportOnly)
            .cloned()
            .collect(),
        unsupported: settings
            .iter()
            .filter(|setting| setting.status == TuneRenderedSettingStatus::Unsupported)
            .cloned()
            .collect(),
        argv,
    })
}

fn setting_context_size(settings: &[TuneRenderedSetting]) -> Option<u32> {
    settings
        .iter()
        .find_map(|setting| match setting.value.as_ref()? {
            TuneRecommendedValue::ContextSize(value)
                if matches!(
                    setting.status,
                    TuneRenderedSettingStatus::Applied | TuneRenderedSettingStatus::Preserved
                ) =>
            {
                Some(*value)
            }
            _ => None,
        })
}

fn setting_device(settings: &[TuneRenderedSetting]) -> Option<String> {
    settings
        .iter()
        .find_map(|setting| match setting.value.as_ref()? {
            TuneRecommendedValue::Device(value)
                if matches!(
                    setting.status,
                    TuneRenderedSettingStatus::Preserved | TuneRenderedSettingStatus::ReportOnly
                ) =>
            {
                Some(value.clone())
            }
            _ => None,
        })
}

pub(crate) fn render_selection(
    selection: &crate::gpus::tune_resolver::TuneTargetSelection,
) -> String {
    match selection {
        crate::gpus::tune_resolver::TuneTargetSelection::Configured => "configured".to_string(),
        crate::gpus::tune_resolver::TuneTargetSelection::Explicit { configured: true } => {
            "explicit_configured".to_string()
        }
        crate::gpus::tune_resolver::TuneTargetSelection::Explicit { configured: false } => {
            "explicit_unconfigured".to_string()
        }
    }
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value
        .chars()
        .all(|character| matches!(character, 'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '/' | '.' | ':' | '-'))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

use crate::gpus::tune_apply::PreparedTunePlan;

use super::*;

pub(crate) fn build_tune_run_report(
    command: &'static str,
    config: &mesh_llm_config::MeshConfig,
    apply_mode: TuneApplyMode,
    prepared: &[PreparedTunePlan],
    target_failures: &[TuneTargetFailure],
    global_blockers: &[String],
    benchmark_reports: &[TuneBenchmarkTargetReport],
) -> TuneRunReport {
    let mut targets = prepared
        .iter()
        .map(|prepared_target| build_prepared_target_report(config, prepared_target))
        .collect::<Vec<_>>();
    targets.extend(target_failures.iter().map(build_failed_target_report));
    TuneRunReport {
        command,
        apply_mode,
        summary: summarize_target_reports(&targets),
        global_blockers: global_blockers.to_vec(),
        targets,
        benchmarks: benchmark_reports.to_vec(),
    }
}

pub(crate) fn collect_settings(
    target: &TuneTargetReport,
    status: TuneRenderedSettingStatus,
) -> Vec<TuneRenderedSetting> {
    target
        .settings
        .iter()
        .filter(|setting| setting.status == status)
        .cloned()
        .collect()
}

fn summarize_target_reports(targets: &[TuneTargetReport]) -> TuneResultSummary {
    let mut summary = TuneResultSummary {
        total_targets: targets.len(),
        ..TuneResultSummary::default()
    };
    for target in targets {
        match target.status {
            TuneTargetStatus::Ready => summary.ready_targets += 1,
            TuneTargetStatus::Written => summary.written_targets += 1,
            TuneTargetStatus::Skipped => summary.skipped_targets += 1,
            TuneTargetStatus::Failed => summary.failed_targets += 1,
        }
        if let Some(field_summary) = &target.field_summary {
            summary.fields.applied += field_summary.applied;
            summary.fields.preserved += field_summary.preserved;
            summary.fields.report_only += field_summary.report_only;
            summary.fields.unsupported += field_summary.unsupported;
            summary.fields.error += field_summary.error;
        }
    }
    summary
}

fn build_prepared_target_report(
    config: &mesh_llm_config::MeshConfig,
    prepared: &PreparedTunePlan,
) -> TuneTargetReport {
    let model_entry = matched_model_entry(config, &prepared.target);
    let defaults = config.defaults.as_ref();
    let settings = prepared
        .plan
        .field_statuses
        .iter()
        .map(|status| render_setting(status, model_entry, defaults))
        .collect::<Vec<_>>();
    let status = classify_prepared_target(prepared);
    TuneTargetReport {
        target: prepared.plan.target.clone(),
        status,
        canonical_model_ref: Some(prepared.target.canonical_model_ref.clone()),
        selection: render_selection(&prepared.target.selection),
        reason: target_status_reason(prepared, status),
        field_summary: Some(prepared.plan.summary()),
        diagnostics: prepared.plan.diagnostics.clone(),
        config_edits: settings
            .iter()
            .filter(|setting| setting.applied_write)
            .cloned()
            .collect(),
        launch: build_launch_preview(prepared, &settings, status),
        settings,
    }
}

fn build_failed_target_report(failure: &TuneTargetFailure) -> TuneTargetReport {
    TuneTargetReport {
        target: TuneTarget {
            requested: failure.requested_input.clone(),
            resolved: None,
            config_model_ref: None,
            derived_profile: None,
        },
        status: TuneTargetStatus::Failed,
        canonical_model_ref: None,
        selection: "unresolved".to_string(),
        reason: Some(failure.reason.clone()),
        field_summary: None,
        diagnostics: Vec::new(),
        settings: Vec::new(),
        config_edits: Vec::new(),
        launch: None,
    }
}

fn classify_prepared_target(prepared: &PreparedTunePlan) -> TuneTargetStatus {
    if plan_error_messages(&prepared.plan).next().is_some() {
        return TuneTargetStatus::Failed;
    }
    match prepared.plan.apply_mode {
        TuneApplyMode::ApplyMissing | TuneApplyMode::ReplaceExisting => {
            if prepared.plan.config_edits().is_empty() {
                TuneTargetStatus::Skipped
            } else {
                TuneTargetStatus::Written
            }
        }
        TuneApplyMode::Review | TuneApplyMode::LaunchArgs => TuneTargetStatus::Ready,
    }
}

fn target_status_reason(prepared: &PreparedTunePlan, status: TuneTargetStatus) -> Option<String> {
    match status {
        TuneTargetStatus::Ready => Some(format!(
            "prepared {} writable tune edits for review",
            prepared.plan.config_edits().len()
        )),
        TuneTargetStatus::Written => Some(format!(
            "wrote {} config edits",
            prepared.plan.config_edits().len()
        )),
        TuneTargetStatus::Skipped => Some("apply produced no writable tune edits".to_string()),
        TuneTargetStatus::Failed => {
            let joined = plan_error_messages(&prepared.plan)
                .collect::<Vec<_>>()
                .join("; ");
            (!joined.is_empty()).then_some(joined)
        }
    }
}

fn plan_error_messages(plan: &TunePlan) -> impl Iterator<Item = String> + '_ {
    let field_messages = plan
        .field_statuses
        .iter()
        .filter_map(|status| match status {
            TuneFieldStatus::Error { diagnostic, .. } => Some(diagnostic.message.clone()),
            TuneFieldStatus::Applied { .. }
            | TuneFieldStatus::Preserved { .. }
            | TuneFieldStatus::ReportOnly { .. }
            | TuneFieldStatus::Unsupported { .. } => None,
        });
    let diagnostic_messages = plan
        .diagnostics
        .iter()
        .filter(|diagnostic| matches!(diagnostic.severity, TuneDiagnosticSeverity::Error))
        .map(|diagnostic| diagnostic.message.clone());
    field_messages.chain(diagnostic_messages)
}

fn render_setting(
    status: &TuneFieldStatus,
    model_entry: Option<&mesh_llm_config::ModelConfigEntry>,
    defaults: Option<&mesh_llm_config::ModelConfigDefaults>,
) -> TuneRenderedSetting {
    match status {
        TuneFieldStatus::Applied {
            recommendation,
            edit,
        } => TuneRenderedSetting {
            field: recommendation.field,
            support: recommendation.field.spec().support,
            status: TuneRenderedSettingStatus::Applied,
            config_path: recommendation.field.spec().config_path.render(),
            value: Some(recommendation.value.clone()),
            rationale: Some(recommendation.rationale.clone()),
            reason: None,
            diagnostic: None,
            edit: Some(edit.clone()),
            applied_write: true,
        },
        TuneFieldStatus::Preserved { field, reason } => TuneRenderedSetting {
            field: *field,
            support: field.spec().support,
            status: TuneRenderedSettingStatus::Preserved,
            config_path: field.spec().config_path.render(),
            value: preserved_value(*field, model_entry, defaults),
            rationale: None,
            reason: Some(reason.clone()),
            diagnostic: None,
            edit: None,
            applied_write: false,
        },
        TuneFieldStatus::ReportOnly {
            recommendation,
            reason,
        } => TuneRenderedSetting {
            field: recommendation.field,
            support: recommendation.field.spec().support,
            status: TuneRenderedSettingStatus::ReportOnly,
            config_path: recommendation.field.spec().config_path.render(),
            value: Some(recommendation.value.clone()),
            rationale: Some(recommendation.rationale.clone()),
            reason: Some(reason.clone()),
            diagnostic: None,
            edit: None,
            applied_write: false,
        },
        TuneFieldStatus::Unsupported { field, reason } => TuneRenderedSetting {
            field: *field,
            support: field.spec().support,
            status: TuneRenderedSettingStatus::Unsupported,
            config_path: field.spec().config_path.render(),
            value: None,
            rationale: None,
            reason: Some(reason.clone()),
            diagnostic: None,
            edit: None,
            applied_write: false,
        },
        TuneFieldStatus::Error { field, diagnostic } => TuneRenderedSetting {
            field: *field,
            support: field.spec().support,
            status: TuneRenderedSettingStatus::Error,
            config_path: field.spec().config_path.render(),
            value: preserved_value(*field, model_entry, defaults),
            rationale: None,
            reason: None,
            diagnostic: Some(diagnostic.clone()),
            edit: None,
            applied_write: false,
        },
    }
}

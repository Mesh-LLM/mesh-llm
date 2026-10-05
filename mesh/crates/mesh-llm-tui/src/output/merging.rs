use super::{
    DashboardEndpointRow, DashboardLaunchPlan, DashboardModelRow, DashboardProcessRow,
    DashboardState, LlamaInstanceState, RunningModelState, RuntimeStatus,
};

pub(super) fn process_rows_match(
    existing: &DashboardProcessRow,
    next: &DashboardProcessRow,
) -> bool {
    if existing.port == next.port {
        return existing.port != 0 || process_row_names_match(&existing.name, &next.name);
    }

    (existing.port == 0 && next.port != 0 && process_row_names_match(&existing.name, &next.name))
        || (next.port == 0
            && existing.port != 0
            && process_row_names_match(&existing.name, &next.name))
}

pub(super) fn endpoint_rows_match(
    existing: &DashboardEndpointRow,
    next: &DashboardEndpointRow,
) -> bool {
    existing.label == next.label || (existing.port != 0 && existing.port == next.port)
}

pub(super) fn process_row_names_match(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }

    if process_row_is_generic_llama(left) || process_row_is_generic_llama(right) {
        return left.contains("llama-server") && right.contains("llama-server");
    }

    match (
        process_row_model_identity(left),
        process_row_model_identity(right),
    ) {
        (Some(left_model), Some(right_model)) => model_names_match(left_model, right_model),
        _ => false,
    }
}

pub(super) fn process_row_is_generic_llama(name: &str) -> bool {
    name == "llama-server"
}

pub(super) fn process_row_model_identity(name: &str) -> Option<&str> {
    if process_row_is_generic_llama(name) {
        None
    } else {
        Some(llama_process_model_name(name).unwrap_or(name))
    }
}

pub(super) fn llama_process_model_name(name: &str) -> Option<&str> {
    name.strip_prefix("llama-server ")
}

pub(super) fn model_name_without_variant_suffix(name: &str) -> &str {
    name.split_once(':')
        .map(|(base_model, _variant)| base_model)
        .unwrap_or(name)
}

pub(super) fn llama_process_row_name(model: Option<&str>) -> String {
    model
        .map(|model| format!("llama-server {model}"))
        .unwrap_or_else(|| "llama-server".to_string())
}

pub(super) fn preferred_dashboard_row_name(existing: &str, next: &str) -> String {
    if next == "llama-server" {
        return existing.to_string();
    }
    if existing == "llama-server" {
        return next.to_string();
    }
    match (name_looks_canonical(existing), name_looks_canonical(next)) {
        (true, false) => existing.to_string(),
        (false, true) => next.to_string(),
        _ => next.to_string(),
    }
}

pub(super) fn name_looks_canonical(name: &str) -> bool {
    let model_name = llama_process_model_name(name).unwrap_or(name);
    model_name.contains('/') || model_name.contains(':')
}

pub(super) fn merged_runtime_status(
    existing: &RuntimeStatus,
    next: &RuntimeStatus,
) -> RuntimeStatus {
    if runtime_status_update_is_stale(existing, next) {
        existing.clone()
    } else {
        next.clone()
    }
}

pub(super) fn runtime_status_update_is_stale(
    existing: &RuntimeStatus,
    next: &RuntimeStatus,
) -> bool {
    matches!(
        (existing, next),
        (
            RuntimeStatus::Ready,
            RuntimeStatus::Loading | RuntimeStatus::Starting | RuntimeStatus::NotReady
        ) | (
            RuntimeStatus::Loading | RuntimeStatus::Starting,
            RuntimeStatus::NotReady
        )
    )
}

pub(super) fn merged_dashboard_device(
    existing: Option<String>,
    next: Option<String>,
) -> Option<String> {
    match (existing, next) {
        (Some(existing), Some(next)) if dashboard_device_update_is_backend_label(&next) => {
            Some(existing)
        }
        (_, Some(next)) => Some(next),
        (existing, None) => existing,
    }
}

pub(super) fn dashboard_device_update_is_backend_label(device: &str) -> bool {
    matches!(
        device.trim().to_ascii_lowercase().as_str(),
        "skippy" | "llama" | "llama.cpp" | "llama-server"
    )
}

pub(in crate::output) fn merged_loaded_model_snapshot_rows(
    existing_rows: &[DashboardModelRow],
    snapshot_rows: &[DashboardModelRow],
) -> Vec<DashboardModelRow> {
    if snapshot_rows.is_empty() {
        return existing_rows.to_vec();
    }

    snapshot_rows
        .iter()
        .cloned()
        .map(|snapshot_row| {
            existing_rows
                .iter()
                .find(|existing| model_rows_match(existing, &snapshot_row))
                .cloned()
                .map(|existing| merged_loaded_model_snapshot_row(existing, snapshot_row.clone()))
                .unwrap_or(snapshot_row)
        })
        .collect()
}

pub(in crate::output) fn merged_loaded_model_snapshot_row(
    existing: DashboardModelRow,
    next: DashboardModelRow,
) -> DashboardModelRow {
    let ctx_used_tokens = next.ctx_used_tokens;
    let lanes = next.lanes.clone();
    let mut merged = merged_loaded_model_row(existing, next);
    // Dashboard snapshots are the authoritative source for live context usage;
    // event/launch-plan rows may omit it and should not clear the latest reading.
    merged.ctx_used_tokens = ctx_used_tokens;
    merged.lanes = lanes;
    merged
}

pub(in crate::output) fn merged_loaded_model_row(
    mut existing: DashboardModelRow,
    next: DashboardModelRow,
) -> DashboardModelRow {
    existing.name = preferred_dashboard_row_name(&existing.name, &next.name);
    existing.status = merged_runtime_status(&existing.status, &next.status);
    existing.role = next.role.or(existing.role);
    existing.port = next.port.or(existing.port);
    existing.device = merged_dashboard_device(existing.device, next.device);
    existing.slots = next.slots.or(existing.slots);
    existing.quantization = next.quantization.or(existing.quantization);
    existing.ctx_size = next.ctx_size.or(existing.ctx_size);
    existing.ctx_used_tokens = next.ctx_used_tokens.or(existing.ctx_used_tokens);
    existing.lanes = next.lanes.or(existing.lanes);
    existing.file_size_gb = next.file_size_gb.or(existing.file_size_gb);
    existing
}

pub(super) fn model_rows_match(existing: &DashboardModelRow, next: &DashboardModelRow) -> bool {
    model_names_match(&existing.name, &next.name)
}

pub(super) fn model_names_match(left: &str, right: &str) -> bool {
    let left_keys = model_identity_keys(left);
    let right_keys = model_identity_keys(right);
    left_keys
        .iter()
        .any(|left_key| right_keys.iter().any(|right_key| left_key == right_key))
}

pub(super) fn model_identity_keys(name: &str) -> Vec<String> {
    let normalized = name.trim().to_ascii_lowercase();
    let basename = normalized
        .rsplit('/')
        .next()
        .unwrap_or(normalized.as_str())
        .to_string();
    let candidates = [normalized, basename];
    let mut keys = Vec::new();
    for candidate in candidates {
        push_model_identity_key(&mut keys, candidate.clone());
        if let Some(variant_name) = candidate
            .rsplit(':')
            .next()
            .filter(|part| *part != candidate && variant_name_looks_like_model_file(part))
        {
            push_model_identity_key(&mut keys, variant_name.to_string());
            push_model_identity_key(&mut keys, variant_name.replace(".gguf", ""));
        }
        push_model_identity_key(&mut keys, candidate.replace("-gguf:", "-"));
        push_model_identity_key(&mut keys, candidate.replace(":gguf:", "-"));
        push_model_identity_key(&mut keys, candidate.replace(':', "-"));
        push_model_identity_key(&mut keys, candidate.replace(".gguf", ""));
    }
    keys
}

pub(super) fn push_model_identity_key(keys: &mut Vec<String>, key: String) {
    if !key.is_empty() && !keys.iter().any(|existing| existing == &key) {
        keys.push(key);
    }
}
pub(super) fn variant_name_looks_like_model_file(value: &str) -> bool {
    value.matches('-').count() >= 2
}

pub fn sort_dashboard_endpoint_rows(rows: &mut [DashboardEndpointRow]) {
    rows.sort_by(|left, right| {
        dashboard_endpoint_sort_bucket(left)
            .cmp(&dashboard_endpoint_sort_bucket(right))
            .then_with(|| left.label.cmp(&right.label))
    });
}

pub(super) fn dashboard_endpoint_sort_bucket(row: &DashboardEndpointRow) -> u8 {
    if row.label.starts_with("Plugin: ") {
        1
    } else {
        0
    }
}

impl DashboardState {
    pub(in crate::output) fn upsert_llama_instance(&mut self, next: LlamaInstanceState) {
        if let Some(existing) = self
            .llama_instances
            .iter_mut()
            .find(|candidate| candidate.kind == next.kind && candidate.port == next.port)
        {
            *existing = next;
        } else {
            self.llama_instances.push(next);
        }

        self.llama_instances
            .sort_by_key(|instance| (instance.kind.sort_key(), instance.port));
    }

    pub(in crate::output) fn upsert_model(
        &mut self,
        model: &str,
        profile: String,
        status: RuntimeStatus,
        internal_port: Option<u16>,
        role: Option<String>,
        capacity_gb: Option<f64>,
    ) {
        if let Some(existing) = self
            .running_models
            .iter_mut()
            .find(|candidate| candidate.model == model && candidate.profile == profile)
        {
            if !matches!(existing.status, RuntimeStatus::Ready)
                || matches!(status, RuntimeStatus::Ready | RuntimeStatus::Stopped)
            {
                existing.status = status;
            }
            existing.internal_port = internal_port.or(existing.internal_port);
            existing.role = role.or_else(|| existing.role.clone());
            existing.capacity_gb = capacity_gb.or(existing.capacity_gb);
        } else {
            self.running_models.push(RunningModelState {
                model: model.to_string(),
                profile,
                status,
                internal_port,
                role,
                capacity_gb,
            });
        }

        self.running_models
            .sort_by(|left, right| left.model.cmp(&right.model));
    }

    pub(in crate::output) fn preseed_launch_plan_rows(&mut self, plan: &DashboardLaunchPlan) {
        for row in &plan.llama_process_rows {
            self.seed_process_row(row);
        }
        for row in &plan.webserver_rows {
            self.seed_endpoint_row(row);
        }
        for row in &plan.loaded_model_rows {
            self.seed_loaded_model_row(row);
        }
    }

    pub(in crate::output) fn seed_process_row(&mut self, row: &DashboardProcessRow) {
        if self
            .llama_process_rows
            .iter()
            .any(|candidate| process_rows_match(candidate, row))
        {
            return;
        }

        let planned = row.clone();
        self.llama_process_rows.push(planned);
        self.llama_process_rows
            .sort_by(|left, right| left.port.cmp(&right.port).then(left.name.cmp(&right.name)));
    }

    pub(in crate::output) fn seed_endpoint_row(&mut self, row: &DashboardEndpointRow) {
        if self
            .webserver_rows
            .iter()
            .any(|candidate| endpoint_rows_match(candidate, row))
        {
            return;
        }

        let mut planned = row.clone();
        planned.status = RuntimeStatus::NotReady;
        self.webserver_rows.push(planned);
        sort_dashboard_endpoint_rows(&mut self.webserver_rows);
    }

    pub(in crate::output) fn seed_loaded_model_row(&mut self, row: &DashboardModelRow) {
        if self
            .loaded_model_rows
            .iter()
            .any(|candidate| model_rows_match(candidate, row))
        {
            return;
        }

        let planned = row.clone();
        self.loaded_model_rows.push(planned);
        self.loaded_model_rows
            .sort_by(|left, right| left.name.cmp(&right.name));
    }

    pub(in crate::output) fn upsert_process_row(&mut self, next: DashboardProcessRow) {
        if let Some(existing) = self
            .llama_process_rows
            .iter_mut()
            .find(|candidate| process_rows_match(candidate, &next))
        {
            existing.name = preferred_dashboard_row_name(&existing.name, &next.name);
            existing.backend = if next.backend.is_empty() {
                existing.backend.clone()
            } else {
                next.backend
            };
            existing.status = merged_runtime_status(&existing.status, &next.status);
            if next.port != 0 {
                existing.port = next.port;
            }
            if next.pid != 0 {
                existing.pid = next.pid;
            }
        } else {
            self.llama_process_rows.push(next);
        }

        self.llama_process_rows
            .sort_by(|left, right| left.port.cmp(&right.port).then(left.name.cmp(&right.name)));
    }

    pub(in crate::output) fn upsert_endpoint_row(&mut self, next: DashboardEndpointRow) {
        if let Some(existing) = self
            .webserver_rows
            .iter_mut()
            .find(|candidate| endpoint_rows_match(candidate, &next))
        {
            existing.label = next.label;
            existing.status = next.status;
            existing.url = next.url;
            if next.port != 0 {
                existing.port = next.port;
            }
            existing.pid = next.pid.or(existing.pid);
        } else {
            self.webserver_rows.push(next);
        }

        sort_dashboard_endpoint_rows(&mut self.webserver_rows);
    }

    pub(in crate::output) fn upsert_loading_model_row(&mut self, model: &str) {
        self.upsert_loaded_model_row(DashboardModelRow {
            name: model.to_string(),
            role: None,
            status: RuntimeStatus::Loading,
            port: None,
            device: None,
            slots: None,
            quantization: None,
            ctx_size: None,
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: None,
        });
    }

    pub(in crate::output) fn upsert_loading_process_row(&mut self, model: &str) {
        self.upsert_process_row(DashboardProcessRow {
            name: llama_process_row_name(Some(model)),
            backend: String::new(),
            status: RuntimeStatus::Loading,
            port: 0,
            pid: 0,
        });
    }

    pub(in crate::output) fn upsert_loaded_model_row(&mut self, next: DashboardModelRow) {
        if let Some(existing) = self
            .loaded_model_rows
            .iter_mut()
            .find(|candidate| model_rows_match(candidate, &next))
        {
            *existing = merged_loaded_model_row(existing.clone(), next);
        } else {
            self.loaded_model_rows.push(next);
        }

        self.loaded_model_rows
            .sort_by(|left, right| left.name.cmp(&right.name));
    }
}

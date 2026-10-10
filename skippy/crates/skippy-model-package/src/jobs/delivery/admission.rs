use super::*;
pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub(super) fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("native delivery text field refused"))
}
fn hex_pin(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn absolute(value: &str) -> bool {
    let path = Path::new(value);
    path.is_absolute()
        && path
            .components()
            .all(|c| !matches!(c, Component::ParentDir | Component::CurDir))
        && value.len() <= 4096
        && !value.chars().any(char::is_control)
}
pub(super) fn artifact(value: &Value, mounts: &[ModelMount], mounted: bool) -> Result<()> {
    let path = text(value, "path")?;
    if !absolute(path)
        || !hex_pin(text(value, "sha256")?, 64)
        || (mounted
            && !mounts.iter().any(|m| {
                Path::new(path)
                    .strip_prefix(&m.mount_path)
                    .is_ok_and(|p| !p.as_os_str().is_empty())
            }))
    {
        bail!("native delivery artifact path/pin/mount refused");
    }
    Ok(())
}
pub(super) fn mounts_admitted(mounts: &[ModelMount]) -> Result<()> {
    if mounts.is_empty() || mounts.len() > 16 {
        bail!("native delivery mount roster refused");
    }
    for (i, mount) in mounts.iter().enumerate() {
        let parts = mount.repo.split('/').collect::<Vec<_>>();
        let repo_ok = parts.len() == 2
            && parts.iter().all(|p| {
                !p.is_empty()
                    && *p != "."
                    && *p != ".."
                    && p.len() <= 128
                    && p.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            });
        if !repo_ok
            || !hex_pin(&mount.revision, 40)
            || !absolute(&mount.mount_path)
            || !Path::new(&mount.mount_path).starts_with("/models")
            || mount.mount_path == "/models"
            || mounts[..i].iter().any(|prior| {
                Path::new(&mount.mount_path).starts_with(&prior.mount_path)
                    || Path::new(&prior.mount_path).starts_with(&mount.mount_path)
            })
        {
            bail!("native delivery requires isolated immutable model mounts");
        }
    }
    Ok(())
}
pub(super) fn resource(request: &Value, plan: &CpuJobPlan, maximum: u64) -> Result<()> {
    let b = &request["bootstrap"];
    let estimate = estimate_cost_usd(plan.unit_cost_usd, &plan.unit_label, plan.timeout_seconds)?;
    if !(30..=maximum).contains(&plan.timeout_seconds)
        || plan.flavor.is_empty()
        || request["timeout_secs"].as_u64() != Some(plan.timeout_seconds)
        || b["timeout_seconds"].as_u64() != Some(plan.timeout_seconds)
        || !plan.unit_cost_usd.is_finite()
        || plan.unit_cost_usd < 0.0
        || !plan.max_cost_usd.is_finite()
        || plan.max_cost_usd < 0.0
        || plan.max_cost_usd != estimate
        || b["declared_estimate_usd"].as_f64() != Some(estimate)
        || b["max_cost_usd"]
            .as_f64()
            .is_none_or(|max| !max.is_finite() || max < estimate || max <= 0.0)
        || text(b, "cpu_plan_receipt_sha256")? != digest(&serde_json::to_vec(plan)?)
    {
        bail!("native delivery resource plan/shared budget mismatch");
    }
    let image = text(b, "image")?;
    let Some((name, pin)) = image.rsplit_once("@sha256:") else {
        bail!("native delivery image must be digest-pinned");
    };
    if name.is_empty()
        || name.len() > 256
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/._-:".contains(&b))
        || !hex_pin(pin, 64)
        || !hex_pin(text(b, "mesh_commit")?, 40)
        || !hex_pin(text(b, "git_tree")?, 40)
        || !hex_pin(text(b, "llama_commit")?, 40)
        || !hex_pin(text(b, "upstream_file_sha256")?, 64)
        || text(b, "native_profile")? != "standalone-static-skippy-quantize-cpu"
    {
        bail!("native delivery source/image/profile declaration refused");
    }
    Ok(())
}
pub(super) fn request(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > 65536 {
        bail!("native delivery input byte bound refused");
    }
    let request: Value = serde_json::from_slice(bytes)
        .map_err(|_| anyhow::anyhow!("native delivery JSON refused"))?;
    let obj = request
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("native delivery object refused"))?;
    let keys = [
        "schema_version",
        "workflow",
        "timeout_secs",
        "runner",
        "bootstrap",
        "certification",
        "projector",
        "receipt_export",
    ];
    if obj.len() != keys.len()
        || keys.iter().any(|k| !obj.contains_key(*k))
        || request["schema_version"] != 1
        || text(&request, "workflow")? != "certification"
    {
        bail!("native delivery closed certification workflow refused");
    }
    mounts_admitted(mounts)?;
    resource(&request, plan, 86400)?;
    artifact(&request["runner"], mounts, false)?;
    let runner = text(&request["runner"], "path")?;
    if Path::new(runner).starts_with("/work") || Path::new(runner).starts_with("/models") {
        bail!("native runner must be supplied by image outside mutable work/model mounts");
    }
    let cert = &request["certification"];
    let parts = cert["target_parts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("native target roster refused"))?;
    if parts.len() > 256 || cert["expected_parts"].as_u64() != Some(parts.len() as u64) {
        bail!("native target complete roster count refused");
    }
    for (i, part) in parts.iter().enumerate() {
        artifact(part, mounts, true)?;
        if parts[..i].iter().any(|prior| prior["path"] == part["path"]) {
            bail!("native target repeated path refused");
        }
    }
    if !cert["mtp_draft"].is_null() {
        artifact(&cert["mtp_draft"], mounts, true)?;
    }
    let projector = &request["projector"];
    artifact(
        &cert["projector"],
        mounts,
        text(projector, "kind")? == "supplied",
    )?;
    match text(projector, "kind")? {
        "supplied" if projector["artifact"] == cert["projector"] => (),
        "hf-https"
            if text(projector, "expected_sha256")? == text(&cert["projector"], "sha256")? => {}
        _ => bail!("native delivery projector correlation refused"),
    }
    export(&request["receipt_export"], mounts, plan.timeout_seconds)?;
    // Full tool roster, native scalar and URL admission remain the existing typed worker's responsibility.
    Ok(request)
}

pub(super) fn export(export: &Value, mounts: &[ModelMount], timeout_seconds: u64) -> Result<()> {
    let repo = text(export, "repo")?;
    let segments = repo.split('/').collect::<Vec<_>>();
    let part = |p: &str| {
        !p.is_empty()
            && !matches!(p, "." | "..")
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    };
    let path = text(export, "path_in_repo")?;
    let reserve = export["export_budget_secs"].as_u64().unwrap_or(60);
    if segments.len() != 2
        || segments.iter().any(|p| p.len() > 96 || !part(p))
        || path.len() > 256
        || path.split('/').any(|p| !part(p))
        || !(10..=3600).contains(&reserve)
        || timeout_seconds <= reserve + 5
    {
        bail!("native delivery export destination/budget refused");
    }
    artifact(&export["helper"], mounts, false)?;
    artifact(&export["helper_source"], mounts, false)?;
    for key in ["helper", "helper_source"] {
        let p = Path::new(text(&export[key], "path")?);
        if p.starts_with("/work") || p.starts_with("/models") {
            bail!("publication helper must be supplied outside mutable job roots");
        }
    }
    if export["credential_environment"] != true
        || !export["credential_file"].is_null()
        || !hex_pin(text(export, "parent_commit")?, 40)
        || !text(export, "path_in_repo")?.ends_with("/native-job.json")
    {
        bail!("native delivery durable receipt/explicit secret configuration refused");
    }
    Ok(())
}

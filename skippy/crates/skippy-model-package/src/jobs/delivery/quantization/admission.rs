use super::*;
fn hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn closed(v: &Value, keys: &[&str]) -> bool {
    v.as_object()
        .is_some_and(|o| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(*k)))
}
fn immutable_artifact(v: &Value, mounts: &[ModelMount], mounted: bool) -> Result<()> {
    admission::artifact(v, mounts, mounted)?;
    if Path::new(admission::text(v, "path")?).starts_with("/work") {
        bail!("quant immutable artifact in mutable work");
    }
    Ok(())
}
fn resources(v: &Value, plan: &CpuJobPlan) -> Result<()> {
    let a = &v["authority"];
    let max = if v["workflow"] == "quantization" {
        259200
    } else {
        345600
    };
    let estimate = estimate_cost_usd(plan.unit_cost_usd, &plan.unit_label, plan.timeout_seconds)?;
    if !closed(
        a,
        &[
            "schema_version",
            "image",
            "mesh_commit",
            "git_tree",
            "cpu_plan_receipt_sha256",
            "declared_estimate_usd",
            "max_cost_usd",
            "tool_kind",
            "profile_version",
        ],
    ) || a["schema_version"] != 1
        || !(30..=max).contains(&plan.timeout_seconds)
        || v["timeout_secs"].as_u64() != Some(plan.timeout_seconds)
        || v["operator"]["timeout_seconds"].as_u64() != Some(plan.timeout_seconds)
        || plan.flavor.is_empty()
        || !plan.unit_cost_usd.is_finite()
        || plan.unit_cost_usd < 0.0
        || !plan.max_cost_usd.is_finite()
        || estimate != plan.max_cost_usd
        || a["declared_estimate_usd"].as_f64() != Some(estimate)
        || a["max_cost_usd"]
            .as_f64()
            .is_none_or(|n| !n.is_finite() || n <= 0.0 || n < estimate)
        || admission::text(a, "cpu_plan_receipt_sha256")?
            != admission::digest(&serde_json::to_vec(plan)?)
        || !hex(admission::text(a, "mesh_commit")?, 40)
        || !hex(admission::text(a, "git_tree")?, 40)
        || a["tool_kind"] != "supplied-window-quantizer"
        || a["tool_kind"] != v["operator"]["window_template"]["tool_kind"]
        || a["profile_version"] != v["operator"]["window_template"]["profile_version"]
    {
        bail!("quant supplied-tool declaration/cost/whole-budget mismatch");
    }
    let image = admission::text(a, "image")?;
    let Some((name, pin)) = image.rsplit_once("@sha256:") else {
        bail!("quant digest image required");
    };
    if name.is_empty()
        || name.len() > 256
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"/._-:".contains(&b))
        || !hex(pin, 64)
    {
        bail!("quant image declaration refused");
    }
    Ok(())
}
fn template(t: &Value) -> Result<()> {
    let keys = [
        "schema_version",
        "tool_kind",
        "profile_version",
        "tool",
        "tool_source",
        "runtime",
        "manifest",
        "recipe",
        "helper",
        "helper_source",
        "source_repo",
        "source_revision",
        "source_root",
        "source_prefix",
        "source_parts",
        "target_repo",
        "target_root",
        "target_prefix",
        "basename",
        "quant",
        "expected_splits",
        "ordinal",
        "work_root",
        "credential_file",
        "publication_confirmed",
        "timeout_seconds",
        "resume",
    ];
    let leaf = |s: &str| {
        !s.is_empty()
            && s.len() <= 128
            && !matches!(s, "." | "..")
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    };
    if !closed(t, &keys)
        || [
            "profile_version",
            "source_prefix",
            "target_prefix",
            "basename",
            "quant",
        ]
        .iter()
        .any(|k| t[k].as_str().is_none_or(|s| !leaf(s)))
        || ["source_repo", "target_repo"].iter().any(|k| {
            t[k].as_str()
                .is_none_or(|s| s.split('/').count() != 2 || !s.split('/').all(leaf))
        })
        || t["timeout_seconds"]
            .as_u64()
            .is_none_or(|n| !(5..=86400).contains(&n))
        || ["source_root", "work_root", "target_root"]
            .iter()
            .any(|k| t[k].as_str().is_none_or(|s| !Path::new(s).is_absolute()))
    {
        bail!("quant template closed grammar/bounds refused");
    }
    Ok(())
}
fn operator(v: &Value, mounts: &[ModelMount]) -> Result<()> {
    let op = &v["operator"];
    let t = &op["window_template"];
    template(t)?;
    if !closed(
        op,
        &[
            "schema_version",
            "workflow",
            "window_template",
            "resumes",
            "loader",
            "timeout_seconds",
            "package",
        ],
    ) || op["schema_version"] != 1
        || op["workflow"] != v["workflow"]
        || t["schema_version"] != 1
        || t["ordinal"] != 1
        || !t["resume"].is_null()
        || !t["credential_file"].is_null()
        || t["publication_confirmed"] != true
        || (v["workflow"] == "quantization") != op["package"].is_null()
    {
        bail!("quant closed operator/publication/credential authority refused");
    }
    immutable_artifact(&op["loader"], mounts, false)?;
    for key in [
        "tool",
        "tool_source",
        "runtime",
        "manifest",
        "recipe",
        "helper",
        "helper_source",
    ] {
        immutable_artifact(&t[key], mounts, false)?;
    }
    let parts = t["source_parts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("quant source roster"))?;
    if !(1..=1024).contains(&parts.len())
        || t["expected_splits"].as_u64() != Some(parts.len() as u64)
    {
        bail!("quant complete source roster refused");
    }
    for (i, p) in parts.iter().enumerate() {
        immutable_artifact(p, mounts, true)?;
        if parts[..i].iter().any(|q| q["path"] == p["path"]) {
            bail!("quant repeated source artifact");
        }
    }
    let source_root = Path::new(admission::text(t, "source_root")?);
    if !mounts.iter().any(|m| {
        t["source_repo"].as_str() == Some(m.repo.as_str())
            && t["source_revision"].as_str() == Some(m.revision.as_str())
            && source_root.starts_with(&m.mount_path)
    }) {
        bail!("quant source repo/revision/mount mismatch");
    }
    for (i, p) in parts.iter().enumerate() {
        let expected = source_root
            .join(admission::text(t, "source_prefix")?)
            .join(format!(
                "{}-{:05}-of-{:05}.gguf",
                admission::text(t, "basename")?,
                i + 1,
                parts.len()
            ));
        if Path::new(admission::text(p, "path")?) != expected {
            bail!("quant ordered source sibling identity refused");
        }
    }
    if !op["package"].is_null() {
        let p = &op["package"];
        if !closed(
            p,
            &[
                "writer",
                "writer_source",
                "generation_defaults",
                "target_repo",
                "max_artifact_bytes",
            ],
        ) || admission::text(p, "target_repo")?.split('/').count() != 2
            || p["target_repo"] == t["target_repo"]
            || p["max_artifact_bytes"]
                .as_u64()
                .is_none_or(|n| n == 0 || n > 1024_u64.pow(4))
        {
            bail!("quant package closed input refused");
        }
        for k in ["writer", "writer_source", "generation_defaults"] {
            immutable_artifact(&p[k], mounts, false)?;
        }
    }
    let resumes = op["resumes"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("quant resume roster"))?;
    if resumes.len() > parts.len() {
        bail!("quant resume count refused");
    }
    for (i, r) in resumes.iter().enumerate() {
        let ordinal = r["ordinal"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("quant resume ordinal"))?;
        if !(1..=parts.len() as u64).contains(&ordinal)
            || resumes[..i].iter().any(|q| q["ordinal"] == r["ordinal"])
            || !hex(admission::text(&r["resume"], "commit")?, 40)
        {
            bail!("quant resume immutable ordinal/commit refused");
        }
        for k in ["record", "shard"] {
            immutable_artifact(&r["resume"][k], mounts, true)?;
        }
    }
    Ok(())
}
pub(super) fn request(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > request_transport::MAX_REQUEST_BYTES {
        bail!("quant Jobs input byte bound");
    }
    let v: Value =
        serde_json::from_slice(bytes).map_err(|_| anyhow::anyhow!("quant Jobs JSON refused"))?;
    if !closed(
        &v,
        &[
            "schema_version",
            "workflow",
            "timeout_secs",
            "runner",
            "authority",
            "operator",
            "receipt_export",
        ],
    ) || v["schema_version"] != 1
        || !matches!(
            v["workflow"].as_str(),
            Some("quantization" | "quantization-and-package")
        )
    {
        bail!("quant Jobs closed workflow refused");
    }
    admission::mounts_admitted(mounts)?;
    resources(&v, plan)?;
    operator(&v, mounts)?;
    immutable_artifact(&v["runner"], mounts, false)?;
    if Path::new(admission::text(&v["runner"], "path")?).starts_with("/models") {
        bail!("quant runner requires declared image custody");
    }
    let export = &v["receipt_export"];
    for k in ["helper", "helper_source"] {
        immutable_artifact(&export[k], mounts, false)?;
    }
    let reserve = export["export_budget_secs"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("quant export budget"))?;
    let path = admission::text(export, "path_in_repo")?;
    let repo = admission::text(export, "repo")?;
    if export["credential_environment"] != true
        || !export["credential_file"].is_null()
        || !(10..=3600).contains(&reserve)
        || plan.timeout_seconds <= reserve + 8
        || !hex(admission::text(export, "parent_commit")?, 40)
        || repo.split('/').count() != 2
        || !path.ends_with("/native-job.json")
        || path.len() > 256
        || path.split('/').any(|s| {
            s.is_empty()
                || matches!(s, "." | "..")
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
    {
        bail!("quant explicit durable export refused");
    }
    Ok(v)
}

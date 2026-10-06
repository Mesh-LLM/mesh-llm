//! Generic conversion delivery over the existing Jobs transport; supplied images are declarations.
use super::*;
#[path = "generic/collection.rs"]
mod collection;
pub use collection::{CollectedConversion, ConversionEvidence, monitor_limits};
#[cfg(test)]
#[path = "generic/tests.rs"]
mod tests;
pub struct PreparedConversionDelivery {
    native: PreparedCertificationDelivery,
    expected_status: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SubmittedConversionDelivery {
    pub native: SubmittedCertificationDelivery,
    pub expected_status: String,
}
impl PreparedConversionDelivery {
    pub fn expected_status(&self) -> &str {
        &self.expected_status
    }
    pub fn declaration(&self) -> &DeliveryDeclaration {
        self.native.declaration()
    }
    pub fn with_publication_credential(mut self, token: String) -> Result<Self> {
        self.native = self.native.with_publication_credential(token)?;
        Ok(self)
    }
    pub fn prepare(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Self> {
        let request = request(bytes, mounts, plan)?;
        let worker_input = serde_json::to_string(&request)?;
        let bootstrap = &request["operator"]["bootstrap"];
        let image = admission::text(bootstrap, "image")?.to_owned();
        let declaration = DeliveryDeclaration {
            schema_version: 1,
            transport_input_sha256: admission::digest(worker_input.as_bytes()),
            runner_sha256: admission::text(&request["runner"], "sha256")?.into(),
            mesh_commit: admission::text(bootstrap, "mesh_commit")?.into(),
            image: image.clone(),
            flavor: plan.flavor.clone(),
            timeout_seconds: plan.timeout_seconds,
            cpu_plan_receipt_sha256: admission::digest(&serde_json::to_vec(plan)?),
            declared_estimate_usd: plan.max_cost_usd,
            evidence_repo: admission::text(&request["receipt_export"], "repo")?.into(),
            evidence_parent_commit: admission::text(&request["receipt_export"], "parent_commit")?
                .into(),
            evidence_path: admission::text(&request["receipt_export"], "path_in_repo")?.into(),
            image_observed: false,
            submitted: false,
            native_certification_completed: false,
        };
        let conversion = &request["operator"]["conversion"];
        let expected_status = if conversion["dry_run"] == true {
            "DRY_RUN_COMPLETED"
        } else if conversion["publish_confirmed"] == true {
            "PUBLISHED"
        } else {
            "LOCAL_ARTIFACT_READY"
        }
        .into();
        let spec = JobSpec {
            docker_image: image,
            command: vec![admission::text(&request["runner"], "path")?.into()],
            arguments: [
                "automation",
                "hf-certify",
                "generic-job-worker",
                "--input-environment",
                INPUT_KEY,
                "--output-directory",
                OUTPUT,
            ]
            .map(str::to_owned)
            .into(),
            environment: HashMap::new(),
            secrets: HashMap::from([(INPUT_KEY.into(), worker_input)]),
            flavor: plan.flavor.clone(),
            timeout_seconds: plan.timeout_seconds,
            volumes: mounts
                .iter()
                .map(|m| JobVolume {
                    volume_type: "model".into(),
                    source: m.repo.clone(),
                    mount_path: m.mount_path.clone(),
                    read_only: Some(true),
                    revision: Some(m.revision.clone()),
                })
                .collect(),
        };
        Ok(Self {
            native: PreparedCertificationDelivery { spec, declaration },
            expected_status,
        })
    }
}
impl HfJobsClient {
    /// Delegates the shared bounded submission owner; its acknowledgment is never certification.
    pub async fn submit_conversion_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        prepared: PreparedConversionDelivery,
        deadline: Instant,
        cancellation: C,
    ) -> Result<SubmittedConversionDelivery> {
        let native = self
            .submit_certification_until(namespace, prepared.native, deadline, cancellation)
            .await?;
        Ok(SubmittedConversionDelivery {
            native,
            expected_status: prepared.expected_status,
        })
    }
}
fn request(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > 65536 {
        bail!("generic delivery input byte bound");
    }
    let v: Value = serde_json::from_slice(bytes)
        .map_err(|_| anyhow::anyhow!("generic delivery JSON refused"))?;
    let keys = [
        "schema_version",
        "workflow",
        "timeout_secs",
        "runner",
        "operator",
        "receipt_export",
    ];
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("generic delivery object"))?;
    if obj.len() != keys.len() + usize::from(obj.contains_key("upload_artifact"))
        || keys.iter().any(|k| !obj.contains_key(*k))
        || v["schema_version"] != 1
        || v["workflow"] != "generic-conversion"
    {
        bail!("generic delivery closed workflow");
    }
    admission::mounts_admitted(mounts)?;
    let op = &v["operator"];
    let o = op
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("generic operator object"))?;
    if o.len() != 3
        || op["schema_version"] != 1
        || !o.contains_key("bootstrap")
        || !o.contains_key("conversion")
    {
        bail!("generic operator closed envelope");
    }
    admission::resource(
        &serde_json::json!({"bootstrap":op["bootstrap"],"timeout_secs":v["timeout_secs"]}),
        plan,
        259200,
    )?;
    admission::artifact(&v["runner"], mounts, false)?;
    let runner = Path::new(admission::text(&v["runner"], "path")?);
    if runner.starts_with("/work") || runner.starts_with("/models") {
        bail!("generic runner must be image supplied");
    }
    let c = &op["conversion"];
    let source = admission::text(c, "source")?;
    let files = c["source_files"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("generic source roster"))?;
    if c["schema_version"] != 1
        || c["timeout_seconds"].as_u64() != Some(plan.timeout_seconds)
        || !c["credential_file"].is_null()
        || c["mesh_revision"] != op["bootstrap"]["mesh_commit"]
        || !admission::absolute(source)
        || (c["upload_only"] != true && files.is_empty())
        || files.len() > 8192
        || (c["upload_only"] != true
            && !mounts.iter().any(|m| {
                Path::new(source).starts_with(&m.mount_path) && c["source_repo"] == m.repo
            }))
        || !(1..=1024).contains(&c["expected_splits"].as_u64().unwrap_or(0))
    {
        bail!("generic immutable conversion source/budget admission");
    }
    for (i, file) in files.iter().enumerate() {
        admission::artifact(file, mounts, true)?;
        let path = Path::new(admission::text(file, "path")?);
        if path == Path::new(source)
            || !path.starts_with(source)
            || files[..i].iter().any(|p| p["path"] == file["path"])
        {
            bail!("generic source roster ancestry/duplicate");
        }
    }
    upload_workspace(&v, mounts, plan)?;
    admission::export(&v["receipt_export"], mounts, plan.timeout_seconds)?;
    // The worker's existing typed conversion/bootstrap owners perform full field admission.
    Ok(v)
}

fn upload_workspace(v: &Value, mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<()> {
    let c = &v["operator"]["conversion"];
    if c["upload_only"] != true {
        if !v["upload_artifact"].is_null() {
            bail!("artifact workspace only for upload-only");
        }
        return Ok(());
    }
    let a = &v["upload_artifact"];
    let obj = a
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("upload artifact workspace required"))?;
    let keys = [
        "schema_version",
        "repo",
        "revision",
        "source_directory",
        "work_directory",
        "target_prefix",
        "output_basename",
        "requested_splits",
        "files",
        "timeout_seconds",
    ];
    if obj.len() != keys.len()
        || keys.iter().any(|k| !obj.contains_key(*k))
        || a["schema_version"] != 1
        || a["timeout_seconds"].as_u64() != Some(plan.timeout_seconds)
        || a["source_directory"] != c["source"]
        || a["work_directory"] != c["work_directory"]
        || a["target_prefix"] != c["target_prefix"]
        || a["output_basename"] != c["output_basename"]
        || a["requested_splits"] != c["expected_splits"]
    {
        bail!("upload artifact workspace correlation");
    }
    let source = admission::text(a, "source_directory")?;
    if !mounts.iter().any(|m| {
        Path::new(source).starts_with(&m.mount_path)
            && a["repo"] == m.repo
            && a["revision"] == m.revision
    }) {
        bail!("upload artifact immutable mount missing");
    }
    let files = a["files"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("upload file roster"))?;
    if files.is_empty() || files.len() > 1024 {
        bail!("upload roster bound");
    }
    let mut names = std::collections::BTreeSet::new();
    for file in files {
        let name = admission::text(file, "name")?;
        if file.as_object().is_none_or(|o| o.len() != 3)
            || name.is_empty()
            || name.len() > 128
            || matches!(name, "." | "..")
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || !{
                let sha = admission::text(file, "sha256")?;
                sha.len() == 64
                    && sha
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            }
            || !names.insert(name)
            || !matches!(file["byte_size"].as_u64(), Some(1..=4398046511104))
        {
            bail!("upload file pin grammar");
        }
    }
    if !names.contains("README.md") || !names.contains("skippy-convert-manifest.json") {
        bail!("upload complete card/manifest required");
    }
    Ok(())
}
#[cfg(all(test, unix))]
pub(super) fn facade_fixture() -> (Value, Vec<ModelMount>, CpuJobPlan) {
    tests::fixture()
}

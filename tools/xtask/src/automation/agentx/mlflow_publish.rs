//! Explicit publication into existing experiment6, preserving historical runs.
use super::{
    command, inputs,
    mlflow_export::{self, Plan},
    prepared_tools::File,
    write,
};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct Client {
    base: url::Url,
    auth: tempfile::NamedTempFile,
    output: PathBuf,
    sequence: u32,
}
impl Client {
    fn new(output: &Path) -> DynResult<Self> {
        let base = url::Url::parse(&std::env::var("MLFLOW_TRACKING_URI")?)?;
        if base.scheme() != "https"
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err("MLflow requires clean HTTPS tracking URI".into());
        }
        let user = std::env::var("MLFLOW_TRACKING_USERNAME")?;
        let password = std::env::var("MLFLOW_TRACKING_PASSWORD")?;
        if [&user, &password]
            .into_iter()
            .any(|v| v.contains(['\r', '\n', '\0']) || v.len() > 8192)
        {
            return Err("invalid MLflow credential shape".into());
        }
        let escape = |value: &str| value.replace('\\', "\\\\").replace('"', "\\\"");
        let mut auth = tempfile::NamedTempFile::new()?;
        writeln!(auth, "user = \"{}:{}\"", escape(&user), escape(&password))?;
        auth.flush()?;
        Ok(Self {
            base,
            auth,
            output: output.into(),
            sequence: 0,
        })
    }
    fn request(
        &mut self,
        route: &str,
        method: &str,
        body: Option<&Value>,
        file: Option<&File>,
    ) -> DynResult<Value> {
        self.sequence += 1;
        let label = format!("mlflow-{:03}", self.sequence);
        let endpoint = self.base.join(route)?;
        let mut args = vec![
            "--disable".into(),
            "--silent".into(),
            "--show-error".into(),
            "--fail-with-body".into(),
            "--proto".into(),
            "=https".into(),
            "--max-time".into(),
            "120".into(),
            "--max-filesize".into(),
            "4194304".into(),
            "--config".into(),
            self.auth.path().to_string_lossy().into_owned(),
            "--request".into(),
            method.into(),
            endpoint.to_string(),
        ];
        if let Some(body) = body {
            let path = self.output.join(format!("{label}.request.json"));
            write(&path, body)?;
            args.extend([
                "--header".into(),
                "Content-Type: application/json".into(),
                "--data-binary".into(),
                format!("@{}", path.display()),
            ]);
        }
        if let Some(file) = file {
            file.verify()?;
            args.extend([
                "--upload-file".into(),
                file.path.to_string_lossy().into_owned(),
            ]);
        }
        let spec = command::spec(&self.output, "curl", &args)?;
        command::public_json(&spec, 140, &self.output.join(&label))
    }
    fn api(&mut self, operation: &str, body: &Value) -> DynResult<Value> {
        self.request(
            &format!("/api/2.0/mlflow/{operation}"),
            "POST",
            Some(body),
            None,
        )
    }
    fn ensure_run(
        &mut self,
        key: &str,
        name: &str,
        tags: &BTreeMap<String, String>,
    ) -> DynResult<Value> {
        if key.len() > 512
            || !key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b))
        {
            return Err("invalid idempotent AgentX run key".into());
        }
        let response=self.api("runs/search",&json!({"experiment_ids":["6"],"filter":format!("tags.ax_run_key = '{key}'"),"max_results":2}))?;
        let runs = response.get("runs").and_then(Value::as_array);
        if runs.is_some_and(|rows| rows.len() > 1) {
            return Err("duplicate MLflow run keys".into());
        }
        if let Some(run) = runs.and_then(|rows| rows.first()) {
            let stored = run["data"]["tags"]
                .as_array()
                .ok_or("MLflow tags missing")?;
            for (key, value) in tags {
                if !stored
                    .iter()
                    .any(|tag| tag["key"] == *key && tag["value"] == *value)
                {
                    return Err("existing MLflow run identity conflicts".into());
                }
            }
            return Ok(run.clone());
        }
        let mut tags = tags.clone();
        tags.insert("ax_run_key".into(), key.into());
        tags.insert("mlflow.runName".into(), name.into());
        let response=self.api("runs/create",&json!({"experiment_id":"6","start_time":now()?,"run_name":name,
            "tags":tags.into_iter().map(|(key,value)|json!({"key":key,"value":value})).collect::<Vec<_>>()}))?;
        response
            .get("run")
            .cloned()
            .ok_or_else(|| "MLflow create response missing run".into())
    }
    fn finish(&mut self, run: &Value) -> DynResult<Value> {
        let id = run_id(run)?;
        self.api(
            "runs/update",
            &json!({"run_id":id,"status":"FINISHED","end_time":now()?}),
        )?;
        let saved = self.request(
            &format!("/api/2.0/mlflow/runs/get?run_id={id}"),
            "GET",
            None,
            None,
        )?;
        if saved["run"]["info"]["status"] != "FINISHED" {
            return Err("MLflow completion readback failed".into());
        }
        Ok(saved["run"].clone())
    }
    fn artifact(&mut self, run: &Value, name: &str, file: &File) -> DynResult<()> {
        let uri = run["info"]["artifact_uri"]
            .as_str()
            .ok_or("MLflow artifact URI missing")?;
        let path = uri
            .strip_prefix("mlflow-artifacts:/6/")
            .ok_or("existing experiment6 artifact proxy required")?;
        if path.split('/').any(|part| part == "..") || name.contains('/') {
            return Err("invalid MLflow artifact route".into());
        }
        let route = format!("/api/2.0/mlflow-artifacts/artifacts/6/{path}/{name}");
        self.request(&route, "PUT", None, Some(file))?;
        self.sequence += 1;
        let destination = self
            .output
            .join(format!("artifact-{}.verified", self.sequence));
        let spec = command::spec(
            &self.output,
            "curl",
            &[
                "--disable".into(),
                "--silent".into(),
                "--show-error".into(),
                "--fail".into(),
                "--proto".into(),
                "=https".into(),
                "--max-time".into(),
                "120".into(),
                "--config".into(),
                self.auth.path().to_string_lossy().into(),
                "--max-filesize".into(),
                std::fs::metadata(&file.path)?
                    .len()
                    .saturating_add(1)
                    .to_string(),
                "--output".into(),
                destination.to_string_lossy().into(),
                self.base.join(&route)?.to_string(),
            ],
        )?;
        command::run(
            &spec,
            140,
            &self.output.join(format!("artifact-{}", self.sequence)),
        )?;
        File {
            path: destination,
            sha256: file.sha256.clone(),
        }
        .verify()?;
        Ok(())
    }
}
fn run_id(run: &Value) -> DynResult<&str> {
    let id = run["info"]["run_id"]
        .as_str()
        .ok_or("MLflow run ID missing")?;
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid MLflow run ID".into());
    }
    Ok(id)
}
fn now() -> DynResult<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}
pub(super) fn run(plan: Plan, output: &Path) -> DynResult<()> {
    mlflow_export::validate_plan(&plan)?;
    std::fs::create_dir(output)?;
    let mut client = Client::new(output)?;
    let existing = client.request(
        "/api/2.0/mlflow/experiments/get?experiment_id=6",
        "GET",
        None,
        None,
    )?;
    mlflow_export::validate_experiment(&existing)?;
    let campaign_root = client.ensure_run(
        &plan.campaign_root.run_key,
        &plan.campaign_root.name,
        &plan.campaign_root.tags,
    )?;
    let mut group_tags = plan.tags.clone();
    group_tags.insert("mlflow.parentRunId".into(), run_id(&campaign_root)?.into());
    let parent = client.ensure_run(&plan.run_key, &plan.name, &group_tags)?;
    let parent_id = run_id(&parent)?.to_string();
    let mut index = Vec::new();
    for child in &plan.children {
        let mut tags = child.tags.clone();
        tags.insert("mlflow.parentRunId".into(), parent_id.clone());
        let run = client.ensure_run(&child.run_key, &child.name, &tags)?;
        let saved = if run["info"]["status"] != "FINISHED" {
            let timestamp = now()?;
            client.api("runs/log-batch",&json!({"run_id":run_id(&run)?,"params":[],"tags":[],
                "metrics":child.metrics.iter().map(|(key,value)|json!({"key":key,"value":value,"timestamp":timestamp,"step":0})).collect::<Vec<_>>()}))?;
            client.finish(&run)?
        } else {
            run.clone()
        };
        for (key, value) in &child.metrics {
            if !saved["data"]["metrics"].as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|row| row["key"] == *key && row["value"].as_f64() == Some(*value))
            }) {
                return Err("MLflow metric readback differs from campaign".into());
            }
        }
        index.push(json!({"run_key":child.run_key,"run_id":run_id(&run)?}));
    }
    client.artifact(&parent, "campaign-result.json", &plan.report)?;
    client.artifact(&parent, "campaign-evidence.tar.gz", &plan.archive)?;
    client.finish(&parent)?;
    write(
        &output.join("publication.json"),
        &json!({"experiment_id":"6","experiment_name":mlflow_export::EXPERIMENT_NAME,
        "campaign_run_id":run_id(&campaign_root)?,"campaign_status":"RUNNING until a separate eight-group audit; this publisher finishes only one model/backend group",
        "parent_run_id":parent_id,"children":index,"report_sha256":plan.report.sha256,"archive_sha256":plan.archive.sha256,
        "publication":"finished; artifacts uploaded, execution/work accounting is distinct from numerical qualification",
        "plan_sha256":inputs::digest(&serde_json::to_vec(&plan)?)}),
    )
}

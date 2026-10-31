//! One complete model/backend campaign: six balanced engine sweeps, no worker rebuilds.
use super::{
    Engine, Model, Prepared, Profile, campaign, comparison_cohort, inputs,
    prepared_tools::{self, Backend, File, Tools},
    results, write,
};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub(super) const ORDER: [&str; 6] = ["llama", "skippy", "skippy", "llama", "llama", "skippy"];
pub(super) const COHORT_SHA: &str =
    "1d4b4c739312ca34a871dbfe85c6e50d533496b2e72782bfdf57b18d515a6c46";
pub(super) const USAGE: &str = "automation agentx {pin|comparison|mlflow-export|mlflow-publish} --input ABS_JSON --output ABS_FRESH_PATH";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub schema_version: u32,
    pub campaign: String,
    pub workspace: PathBuf,
    pub model_key: String,
    pub model: PathBuf,
    pub corpus: File,
    pub tools: Tools,
}
impl Request {
    fn validate(&self) -> DynResult<Model> {
        if self.schema_version != 1
            || self.campaign.is_empty()
            || self.campaign.len() > 128
            || !self
                .campaign
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err("invalid final AgentX campaign identity".into());
        }
        prepared_tools::absolute(&self.workspace)?;
        prepared_tools::absolute(&self.model)?;
        if !self.workspace.join("Justfile").is_file() {
            return Err("owning workspace missing".into());
        }
        self.corpus.verify()?;
        if self.corpus.sha256 != inputs::DATA_SHA {
            return Err("final comparison corpus pin differs".into());
        }
        let catalog = std::fs::read(
            self.workspace
                .join("skippy/evals/kv-validation/models.json"),
        )?;
        if inputs::digest(&catalog)
            != "6a2b724cf32208186d80ac614e21a168bfc08ca72e55afdabc3dc39b338c9206"
        {
            return Err("final comparison requires the exact historical four-model catalog".into());
        }
        let model = inputs::select_model(&self.workspace, &self.model_key)?;
        inputs::verify(&self.model, &model.files[0].sha256)?;
        self.tools.verify(&self.workspace)?;
        super::agentic_config::verify_tokenizer_snapshot(
            &self.tools.tokenizer.path,
            &super::agentic_config::tokenizer_snapshot(&self.workspace, &model),
        )?;
        Ok(model)
    }
}
pub(super) fn command(args: &[String]) -> DynResult<()> {
    let [verb, input_flag, input, output_flag, output] = args else {
        return Err(USAGE.into());
    };
    if input_flag != "--input" || output_flag != "--output" {
        return Err("invalid final AgentX arguments".into());
    }
    let input = Path::new(input);
    let output = Path::new(output);
    prepared_tools::absolute(input)?;
    prepared_tools::absolute(output)?;
    let bytes = crate::automation::receipt_files::bounded(input, 1024 * 1024)?;
    if verb == "mlflow-export" {
        return super::mlflow_export::run(serde_json::from_slice(&bytes)?, output);
    }
    if verb == "mlflow-publish" {
        return super::mlflow_publish::run(serde_json::from_slice(&bytes)?, output);
    }
    if verb == "pin" {
        let mut request: Request = serde_json::from_slice(&bytes)?;
        request.corpus.sha256 =
            crate::product::digest::file_sha256(&request.corpus.path).map_err(|e| e.error)?;
        request.tools.pin()?;
        request.validate()?;
        return crate::automation::receipt_files::fresh(
            output,
            &serde_json::to_vec_pretty(&request)?,
        );
    }
    if verb == "comparison" {
        return run(serde_json::from_slice(&bytes)?, output);
    }
    Err(format!("unknown AgentX command: {verb}").into())
}
fn run(request: Request, output: &Path) -> DynResult<()> {
    std::fs::create_dir(output)?;
    write(&output.join("request.json"), &request)?;
    let mut arms = Vec::new();
    let mut result = execute(&request, output, &mut arms);
    let status = if result.is_ok() { "passed" } else { "failed" };
    let mut report = json!({"schema_version":1,"campaign":request.campaign,"model_key":request.model_key,"request":request,
        "backend":request.tools.backend,"status":status,"arms":arms,"error":result.as_ref().err().map(ToString::to_string),
        "repeat_order":ORDER,"concurrencies":[1,2,4],"scope":if request.tools.backend==Backend::Cpu{"cpu_diagnostic_not_gpu_performance"}else{"fixed_complete_AgentX_comparison"},
        "jy_job_id":std::env::var("JIANYANG_JOB_ID").ok(),"jy_run_id":std::env::var("JIANYANG_RUN_ID").ok(),
        "source_attribution":{"product":request.tools.product_source_commit,"native":request.tools.native_source_commit,
            "standard_upstream":request.tools.standard_upstream,"automation_binary_sha256":crate::product::digest::file_sha256(&std::env::current_exe()?).map_err(|e|e.error)?},
        "mlflow":{"experiment_id":"6","experiment_name":"agentx-skippy-vs-llama","publication":"not_published; export/audit separately"}});
    write(&output.join("result.json"), &report)?;
    if let Err(error) = results::archive(output) {
        report["status"] = json!("failed");
        report["error"] = json!(format!(
            "evidence archive failed: {error}; execution error: {:?}",
            result.as_ref().err().map(ToString::to_string)
        ));
        write(&output.join("result.json"), &report)?;
        result = Err(error);
    }
    if let Some(path) = std::env::var_os("JIANYANG_RESULT_PATH") {
        write(Path::new(&path), &report)?;
    }
    result
}
fn execute(request: &Request, output: &Path, arms: &mut Vec<Value>) -> DynResult<()> {
    admit_host(request.tools.backend, output)?;
    let model = request.validate()?;
    super::prepared_launch::doctor(&request.tools, &request.workspace, output)?;
    write(&output.join("prepared-identities.json"), request)?;
    let traces = output.join("traces");
    std::fs::create_dir(&traces)?;
    comparison_cohort::materialize(&request.corpus.path, &traces)?;
    let expected: Value = serde_json::from_slice(&std::fs::read(output.join("cohort.json"))?)?;
    if expected["expected_requests"] != 69 || expected["expected_output_tokens"] != 18713 {
        return Err("comparison source work must remain69requests/18713outputs".into());
    }
    for (index, engine) in ORDER.into_iter().enumerate() {
        let arm = output.join(format!("{}-{engine}", index + 1));
        std::fs::create_dir(&arm)?;
        let prepared = Prepared {
            tools: Some(request.tools.clone()),
            root: request.workspace.clone(),
            output: arm.clone(),
            model: model.clone(),
            weights: request.model.clone(),
            dataset: traces.clone(),
            profile: Profile::Comparison,
        };
        let outcome = campaign::run(
            &prepared,
            if engine == "llama" {
                Engine::Llama
            } else {
                Engine::Skippy
            },
            None,
        );
        let cells: Value = serde_json::from_slice(&std::fs::read(arm.join("results.json"))?)?;
        arms.push(json!({"index":index+1,"engine":engine,"cells":cells["cells"],"status":if outcome.is_ok(){"passed"}else{"failed"}}));
        write(
            &output.join("progress.json"),
            &json!({"arms":arms,"expected_arms":6}),
        )?;
        outcome?;
    }
    validate_work(arms)?;
    request.validate()?;
    Ok(())
}
fn admit_host(backend: Backend, output: &Path) -> DynResult<()> {
    let spec = super::command::spec(output, "hostname", &["-s".into()])?;
    super::command::run(&spec, 10, &output.join("hostname"))?;
    let host = std::fs::read_to_string(output.join("hostname.stdout.log"))?;
    if ["spark-db08", "spark-9f06", "spark-000280", "spark-000499"].contains(&host.trim()) {
        return Err("excluded benchmark host".into());
    }
    if (backend == Backend::Metal && std::env::consts::OS != "macos")
        || (backend == Backend::Cuda && std::env::consts::OS != "linux")
    {
        return Err("prepared backend does not match host OS".into());
    }
    Ok(())
}
pub(super) fn validate_work(arms: &[Value]) -> DynResult<()> {
    if arms.len() != 6 {
        return Err("incomplete ABBAAB campaign".into());
    }
    let mut work = None;
    for (arm, engine) in arms.iter().zip(ORDER) {
        if arm["engine"] != engine || arm["status"] != "passed" {
            return Err("ABBAAB engine order/outcome differs".into());
        }
        let cells = arm["cells"].as_array().ok_or("campaign cells missing")?;
        if cells.len() != 3 {
            return Err("comparison requires allc1/c2/c4 cells".into());
        }
        for (cell, concurrency) in cells.iter().zip([1, 2, 4]) {
            let hash = cell["completed_work"]["source_turn_output_sha256"]
                .as_str()
                .ok_or("completed-work hash missing")?;
            if cell["offered_sessions"] != concurrency
                || cell["completed_work"]["requests"] != 69
                || cell["output_tok_s"]
                    .as_f64()
                    .is_none_or(|v| !v.is_finite() || v <= 0.0)
                || cell["completed_work"]["output_tokens"] != 18713
                || cell["cohort"]["selected_cohort_sha256"] != COHORT_SHA
                || cell["phase_outcome"]["errors"] != 0
                || cell["phase_outcome"]["cancelled"] != 0
                || work.is_some_and(|previous| previous != hash)
            {
                return Err("fixed-work/cohort/concurrency mismatch".into());
            }
            work = Some(hash);
        }
    }
    Ok(())
}

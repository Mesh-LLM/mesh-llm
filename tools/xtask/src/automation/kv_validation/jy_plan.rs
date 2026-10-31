//! Manual JY specs using the observed lab/custom schema, not a new CI provider.
use super::{
    contract::{self, Backend, Cell, Tool},
    publish,
};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Node {
    pub id: String,
    pub hostname: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Fleet {
    pub schema_version: u32,
    pub backend: Backend,
    pub source_commit: String,
    pub prepared_image: String,
    pub automation: PathBuf,
    pub prepared: Tool,
    pub nodes: Vec<Node>,
    pub execution_seconds: u64,
}
impl Fleet {
    fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !contract::sha(&self.source_commit, 40)
            || self.backend == Backend::Metal
            || self
                .prepared_image
                .strip_prefix("sha256:")
                .is_none_or(|s| !contract::sha(s, 64))
            || self.nodes.is_empty()
            || self.nodes.len() > 16
            || !(30..=14400).contains(&self.execution_seconds)
        {
            return Err(
                "JY foundation requires pinned Linux CPU/CUDA image, source and bounded nodes/time"
                    .into(),
            );
        }
        contract::absolute(&self.automation)?;
        contract::absolute(&self.prepared.path)?;
        if !contract::sha(&self.prepared.sha256, 64) {
            return Err("JY prepared receipt SHA required".into());
        }
        let mut ids = BTreeSet::new();
        let mut hosts = BTreeSet::new();
        for node in &self.nodes {
            if !node.id.starts_with("spark-")
                || !safe_name(&node.id)
                || !safe_name(&node.hostname)
                || excluded(&node.id, &node.hostname)
                || !ids.insert(&node.id)
                || !hosts.insert(&node.hostname)
            {
                return Err("duplicate, excluded or malformed JY node".into());
            }
        }
        Ok(())
    }
}
fn safe_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}
pub(super) fn excluded(id: &str, hostname: &str) -> bool {
    ["spark-000280", "spark-000499"].contains(&id)
        || ["spark-db08", "spark-9f06"].contains(&hostname)
}
fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub(super) fn run(input: Fleet, output: &Path) -> DynResult<()> {
    input.validate()?;
    std::fs::create_dir(output)?;
    for (index, model) in contract::models()?.iter().enumerate() {
        let node = &input.nodes[index % input.nodes.len()];
        let request = Cell {
            schema_version: 1,
            prepared: input.prepared.clone(),
            model_key: model.key.clone(),
            model: "/model.gguf".into(),
            port: 19337,
            execution_seconds: input.execution_seconds,
        };
        let gpu = if input.backend == Backend::Cuda {
            " --gpus all"
        } else {
            ""
        };
        // This is only a platform adapter. Selection, identity, supervision and
        // pass/fail policy live in the typed run owner inside the immutable image.
        let command = format!(
            "set -eu\ncase \"$(hostname -s)\" in {}) ;; *) exit 90;; esac\nset +e\ndocker run --rm --init{gpu} --user \"$(id -u):$(id -g)\" --mount \"type=bind,src=$AGENTX_MODEL_PATH,dst=/model.gguf,readonly\" --mount \"type=bind,src=$JIANYANG_ARTIFACT_DIR,dst=/artifacts\" --env KV_VALIDATION_REQUEST_JSON {} {} automation kv-validation run-env --input KV_VALIDATION_REQUEST_JSON --output /artifacts/evidence\nworker_status=$?\nset -e\nif [ -f \"$JIANYANG_ARTIFACT_DIR/evidence/result.json\" ]; then cp \"$JIANYANG_ARTIFACT_DIR/evidence/result.json\" \"$JIANYANG_RESULT_PATH\"; fi\nexit \"$worker_status\"\n",
            node.hostname,
            quote(&input.prepared_image),
            quote(
                input
                    .automation
                    .to_str()
                    .ok_or("automation path Unicode required")?
            )
        );
        let spec = json!({"spec":{
            "budget":{"max_duration_s":input.execution_seconds+120,"grace_s":90,"max_usd":0},
            "description":"Explicit four-model KV foundation: dtype dispatch, deterministic c1 reuse, declared recurrent regressions; excludes long-c4/L3/performance.",
            "harness":{"args":{},"concurrency":[1],"name":"custom","percentiles":[50,90,95,99],"request_count":1,"seed":0,"trace":{"kind":"synthetic","ref":"none"},"version":"kv-validation-foundation-v1","warmup_requests":0},
            "labels":{"campaign":"kv-validation","model":model.key,"node":node.id},
            "name":format!("kv-validation-{}-{}",input.backend.kind(),model.key),"priority":20,"retries":0,
            "runtime":{"command":command,"env":{"HF_HUB_OFFLINE":"1","JIANYANG_COMMIT":input.source_commit,"JIANYANG_REPO_DIR":"mesh-llm","KV_VALIDATION_REQUEST_JSON":serde_json::to_string(&request)?},
                "flags":{"backend":input.backend.kind(),"model":model.reference(),"ref":format!("commit:{}",input.source_commit),"repo":"mesh-llm","validation":"kv-foundation","node":node.id},
                "model_package":{"egress":"lab_only","ref":model.reference(),"sha256":model.files[0].sha256},
                "models":[{"delivery":"stream","env":"AGENTX_MODEL_PATH","pin":false,"ref":model.reference()}],"volumes":[],
                "workspace":{"repos":[{"rev":input.source_commit,"submodules":false,"url":"https://github.com/Mesh-LLM/mesh-llm.git"}],"retain":"always"}},
            "target":{"executor":"lab","num_nodes":1,"os":"linux","spot":false,"nodes":[node.id]}
        }});
        publish(&output.join(format!("{}.json", model.key)), &spec)?;
    }
    publish(
        &output.join("plan.json"),
        &json!({"schema_version":1,"jobs":4,"catalog_sha256":contract::CATALOG_SHA,"backend":input.backend,"source_commit":input.source_commit,"prepared_image":input.prepared_image,"execution":"submit each model JSON with jy submit; no automatic dispatch"}),
    )
}

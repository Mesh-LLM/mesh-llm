//! Blocking identity/model admission runs only in a parent-supervised child.
use super::{acceptance::Version, native_identity, options};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub round: u64,
    pub version: Version,
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub commit: String,
    pub native_build: PathBuf,
    pub native_build_sha256: String,
    pub native_profile: String,
    pub model: PathBuf,
    pub model_sha256: String,
    pub model_id: String,
    pub ctx_size: u32,
    pub split_layer: u32,
    pub layer_end: u32,
    pub n_gpu_layers: i32,
    pub adaptive_target_ms: f64,
    pub stage_ports: [u16; 2],
    pub openai_port: u16,
}
pub(super) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub(in crate::automation) fn bounded(path: &Path, maximum: u64) -> DynResult<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() {
        return Err("adaptive input/receipt must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened adaptive input/receipt must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > maximum {
        return Err("adaptive input/receipt exceeds bound".into());
    }
    Ok(bytes)
}
pub(in crate::automation) fn fresh(path: &Path, bytes: &[u8]) -> DynResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("adaptive receipt output must be fresh".into()),
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output parent absent")?)?;
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        let digest = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if self.schema_version != 1
            || self.native_profile != "standalone-static-skippy-server"
            || self.round == 0
            || self.model_id.is_empty()
            || self.model_id.len() > 4096
            || !digest(&self.binary_sha256)
            || !digest(&self.native_build_sha256)
            || !digest(&self.model_sha256)
            || ![40, 64].contains(&self.commit.len())
            || !self
                .commit
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || [&self.binary, &self.native_build, &self.model]
                .iter()
                .any(|p| !p.is_absolute())
            || self.ctx_size == 0
            || self.split_layer == 0
            || self.split_layer >= self.layer_end
            || self.n_gpu_layers < -1
            || !self.adaptive_target_ms.is_finite()
            || self.adaptive_target_ms <= 0.0
            || self.stage_ports.contains(&0)
            || self.openai_port == 0
            || self.stage_ports[0] == self.stage_ports[1]
            || self.stage_ports.contains(&self.openai_port)
        {
            return Err("invalid adaptive arm identity/context/topology/policy".into());
        }
        Ok(())
    }
    fn admitted(&mut self) -> DynResult<Value> {
        self.validate()?;
        self.binary = self.binary.canonicalize()?;
        self.native_build = self.native_build.canonicalize()?;
        self.model = self.model.canonicalize()?;
        if !self.binary.is_file() || !self.model.is_file() || !self.native_build.is_dir() {
            return Err("adaptive static binary/native-build paths refused".into());
        }
        let binary = crate::product::digest::file_sha256(&self.binary).map_err(|e| e.error)?;
        if binary != self.binary_sha256 {
            return Err("adaptive binary digest mismatch".into());
        }
        native_identity::verify(&self.native_build, &self.native_build_sha256)?;
        let model = crate::automation::replay_matrix::model_preflight::verify(
            &self.model,
            &self.model_sha256,
            u64::from(self.ctx_size),
        )?;
        let dimensions =
            crate::automation::replay_matrix::model_preflight::dimensions::inspect(&self.model)?
                .ok_or("complete GGUF dimensions required")?;
        if dimensions.block_count != u64::from(self.layer_end) {
            return Err("adaptive stage range differs from actual GGUF layers".into());
        }
        Ok(serde_json::to_value(model)?)
    }
    pub fn config(&self, index: usize) -> Value {
        let peer = 1 - index;
        let endpoint = json!({"stage_id":format!("stage-{peer}"),"stage_index":peer,"endpoint":format!("tcp://127.0.0.1:{}",self.stage_ports[peer])});
        json!({"run_id":format!("adaptive-{}-{}",self.round,if self.version==Version::Old {"old"}else{"new"}),
            "topology_id":"skippy-adaptive-prefill-ab-two-stage","model_id":self.model_id,"model_path":self.model,
            "source_model_sha256":self.model_sha256,"stage_id":format!("stage-{index}"),"stage_index":index,
            "layer_start":if index==0 {0}else{self.split_layer},"layer_end":if index==0 {self.split_layer}else{self.layer_end},
            "ctx_size":self.ctx_size,"lane_count":1,"n_gpu_layers":self.n_gpu_layers,"cache_type_k":"f16","cache_type_v":"f16",
            "load_mode":"runtime-slice","execution_contract":"","bind_addr":format!("127.0.0.1:{}",self.stage_ports[index]),
            "upstream":if index==1 {endpoint.clone()}else{Value::Null},"downstream":if index==0 {endpoint}else{Value::Null},
            "kv_cache":{"mode":"disabled"}})
    }
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let output = Path::new(opts["--output"]);
    if !output.is_absolute() || !Path::new(opts["--input"]).is_absolute() {
        return Err("adaptive identity worker paths must be absolute".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("adaptive identity output must be fresh".into()),
    }
    let bytes = bounded(Path::new(opts["--input"]), 64 * 1024)?;
    let mut input: Input = serde_json::from_slice(&bytes)?;
    let model = input.admitted()?;
    let configs = [input.config(0), input.config(1)];
    fresh(
        output,
        &serde_json::to_vec(&json!({"schema_version":1,"request_sha256":digest(&bytes),
        "admitted":input,"model_identity":model,"configs":configs}))?,
    )
}

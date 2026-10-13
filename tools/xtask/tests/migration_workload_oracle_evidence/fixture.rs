use super::super::{VerifyRequest, WorkloadClass, WriteRequest};
use std::fs;
use std::path::PathBuf;

pub(super) const MODEL_HASH: &str =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
pub(super) const CANDIDATE_HASH: &str =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
pub(super) const ORACLE_HASH: &str =
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
pub(super) const PROJECTOR_HASH: &str =
    "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
pub(super) const METRICS: &str = r#"{"sample_rate_hz":24000,"channels":1,"sample_count":24000,"relative_rms_error":0.02,"waveform_cosine":0.9995}"#;

pub(super) struct Fixture {
    pub(super) directory: tempfile::TempDir,
    pub(super) model: PathBuf,
    pub(super) write: WriteRequest,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.gguf");
        let candidate = directory.path().join("skippy-server");
        let oracle = directory.path().join("llama-server");
        let log = directory.path().join("comparison.txt");
        fs::write(&model, b"abc").unwrap();
        fs::write(&candidate, b"").unwrap();
        fs::write(&oracle, b"hello").unwrap();
        fs::write(&log, "embedding local-monolithic oracle passed: fixture\n").unwrap();
        let write = WriteRequest {
            output: directory.path().join("evidence.json"),
            comparison_log: log,
            model_class: "embedding".into(),
            smoke_lane: "embedding-smoke".into(),
            model_id: "fixture".into(),
            model_sha256: MODEL_HASH.into(),
            projector_path: None,
            candidate_executable: candidate,
            oracle_executable: oracle,
            pinned_patch_sha: "a".repeat(40),
            work_dir: directory.path().to_owned(),
        };
        Self {
            directory,
            model,
            write,
        }
    }

    pub(super) fn verify(&self) -> VerifyRequest {
        VerifyRequest {
            evidence: self.write.output.clone(),
            model_class: WorkloadClass::parse(&self.write.model_class).unwrap(),
            smoke_lane: self.write.smoke_lane.clone(),
            oracle_lane: self
                .write
                .smoke_lane
                .strip_suffix("-smoke")
                .unwrap()
                .to_owned()
                + "-oracle",
            model_id: self.write.model_id.clone(),
            model_path: self.model.clone(),
            projector_path: self.write.projector_path.clone(),
            candidate_executable: self.write.candidate_executable.clone(),
            oracle_executable: self.write.oracle_executable.clone(),
            pinned_patch_sha: self.write.pinned_patch_sha.clone(),
        }
    }

    pub(super) fn select(&mut self, class: &str, executable: &str) {
        self.write.model_class = class.into();
        self.write.oracle_executable = self.directory.path().join(executable);
        fs::write(&self.write.oracle_executable, b"hello").unwrap();
        fs::write(
            &self.write.comparison_log,
            format!("{class} local-monolithic oracle passed: fixture\n"),
        )
        .unwrap();
    }

    pub(super) fn projector(&mut self) {
        let path = self.directory.path().join("projector.gguf");
        fs::write(&path, b"hello\n").unwrap();
        self.write.projector_path = Some(path);
    }

    pub(super) fn body(&self) -> serde_json::Value {
        serde_json::json!({
            "status": "pass", "class": self.write.model_class,
            "smoke_lane": self.write.smoke_lane,
            "oracle_lane": self.verify().oracle_lane,
            "model_id": "fixture", "model_sha256": MODEL_HASH,
            "projector_sha256": self.write.projector_path.as_ref().map(|_| PROJECTOR_HASH),
            "candidate_executable_sha256": CANDIDATE_HASH,
            "oracle_executable": self.write.oracle_executable.file_name().unwrap().to_str().unwrap(),
            "oracle_executable_sha256": ORACLE_HASH,
            "pinned_patch_sha": "a".repeat(40),
            "comparison": format!("{} local-monolithic oracle passed: fixture", self.write.model_class)
        })
    }

    pub(super) fn save(&self, body: &serde_json::Value) {
        fs::write(&self.write.output, serde_json::to_vec(body).unwrap()).unwrap();
    }

    pub(super) fn tts_result(&self, metrics: &str) {
        fs::write(
            self.write.work_dir.join("tts-oracle-result.json"),
            format!(
                r#"{{"status":"pass","pinned_patch_sha":"{}","metrics":{metrics}}}"#,
                self.write.pinned_patch_sha
            ),
        )
        .unwrap();
    }
}

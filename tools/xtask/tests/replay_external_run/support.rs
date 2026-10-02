use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub struct Fixture {
    pub state: tempfile::TempDir,
    pub input: PathBuf,
    pub config: PathBuf,
    pub output: PathBuf,
    pub version: PathBuf,
    pub executable: PathBuf,
}

impl Fixture {
    pub fn new(mixed: bool) -> Self {
        let state = tempfile::Builder::new()
            .prefix("external run ")
            .tempdir_in(std::env::temp_dir().canonicalize().unwrap())
            .unwrap();
        let version = state.path().join("version.txt");
        std::fs::write(&version, "fixture engine 1.2\n").unwrap();
        let engine = state.path().join("engine-server");
        write_executable(
            &engine,
            &format!(
                "#!/bin/sh\ncase \"$1\" in --version|-c) /bin/cat {}; exit 0;; esac\nprintf 'server\\n' >> {}\nexec {} \"$@\"\n",
                quote(&version),
                quote(&state.path().join("launches.txt")),
                quote(Path::new(env!("CARGO_BIN_EXE_laya-product-fixture")))
            ),
        );
        let executable = state.path().join("venv-server");
        std::os::unix::fs::symlink(&engine, &executable).unwrap();
        let config = state.path().join("engines.json");
        write_json(
            &config,
            &json!({"schema_version":1,"comparison":{"model":"fixture/model"},"arms":[{
            "label":"external.fixture","engine":"llama.cpp","executable":"./venv-server","model":"unmaterialized/model.gguf",
            "context_size":131072,"max_concurrency":4,"tokenizer":"unmaterialized/tokenizer","extra_args":["--fixture-extra","two words"]}]}),
        );
        let manifest = state.path().join("manifest.json");
        let trajectory = |session: &str| {
            json!({"session_id":session,"source_dataset":"fixture","agent_framework":"fixture","recorded_model":null,
            "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
                {"role":"user","content":"next"},{"role":"assistant","content":"final"}]})
        };
        write_json(
            &manifest,
            &json!({"cohorts":{"warmup":[trajectory("warmup")],
            "1":[trajectory("one-a"),trajectory("one-b")],
            "2":[trajectory("two-a"),trajectory("two-b"),trajectory("two-c"),trajectory("two-d")]}}),
        );
        let output = state.path().join("run");
        let input = state.path().join("input.json");
        let builds = if mixed {
            vec![mesh_build(state.path())]
        } else {
            Vec::new()
        };
        write_json(
            &input,
            &json!({"manifest":manifest,"requirements":{"concurrency":[1,2],
            "minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["fixture"]},
            "builds":builds,"context_qualification":"captured","model":"fixture/model","passes":2,
            "max_output_tokens":2048,"request_timeout_seconds":2,"startup_timeout_seconds":3,
            "timeout_seconds":60,"output":output,"require_output_match":true}),
        );
        Self {
            state,
            input,
            config,
            output,
            version,
            executable,
        }
    }
    pub fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "execute-run", "--input"])
            .arg(&self.input)
            .arg("--engine-config")
            .arg(&self.config)
            .output()
            .unwrap()
    }
    pub fn document(&self) -> Value {
        read_json(&self.output.join("run.json"))
    }
    pub fn modify_input(&self, mutate: impl FnOnce(&mut Value)) {
        modify(&self.input, mutate);
    }
    pub fn modify_config(&self, mutate: impl FnOnce(&mut Value)) {
        modify(&self.config, mutate);
    }
    pub fn launches(&self) -> usize {
        std::fs::read_to_string(self.state.path().join("launches.txt"))
            .unwrap()
            .lines()
            .count()
    }
    pub fn fail_server(&self) {
        write_executable(
            &self.state.path().join("engine-server"),
            &format!(
                "#!/bin/sh\ncase \"$1\" in --version|-c) /bin/cat {}; exit 0;; esac\nprintf 'server\\n' >> {}\nexit 9\n",
                quote(&self.version),
                quote(&self.state.path().join("launches.txt")),
            ),
        );
    }
}

fn mesh_build(root: &Path) -> Value {
    let runtime_root = root.join("native-runtimes");
    let runtime = runtime_root.join("fixture");
    std::fs::create_dir_all(&runtime).unwrap();
    let bytes = b"fixture runtime";
    std::fs::write(runtime.join("runtime.so"), bytes).unwrap();
    let mut tree = Sha256::new();
    tree.update(10_u64.to_be_bytes());
    tree.update(b"runtime.so");
    tree.update(Sha256::digest(bytes));
    let binary = Path::new(env!("CARGO_BIN_EXE_laya-product-fixture"));
    json!({"label":"mesh","ref":"fixture-main","commit":"0123456789abcdef0123456789abcdef01234567",
        "binary":binary,"binary_sha256":hex::encode(Sha256::digest(std::fs::read(binary).unwrap())),
        "runtime_root":runtime_root,"runtime":runtime,"runtime_sha256":hex::encode(tree.finalize()),"backend":"metal"})
}

pub fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
pub fn write_json(path: &Path, value: &Value) {
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub fn modify(path: &Path, mutate: impl FnOnce(&mut Value)) {
    let mut value = read_json(path);
    mutate(&mut value);
    write_json(path, &value);
}
pub fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn write_executable(path: &Path, text: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, text).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}

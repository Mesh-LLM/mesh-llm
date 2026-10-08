use super::*;
use sha2::{Digest, Sha256};

struct Fixture {
    directory: tempfile::TempDir,
    side: plan::Side,
    entry: plan::Entry,
    model: PathBuf,
    runtime: PathBuf,
    binary_sha256: String,
    model_sha256: String,
    inherited: BTreeMap<OsString, OsString>,
    output: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let canonical = directory.path().canonicalize().unwrap();
        let binary = canonical.join("not-an-executable-fixture");
        let model = canonical.join("not-a-real-model.gguf");
        let runtime = canonical.join("runtime");
        std::fs::write(&binary, b"fixture binary bytes").unwrap();
        std::fs::write(&model, b"fixture model bytes").unwrap();
        std::fs::create_dir(&runtime).unwrap();
        let output = canonical.join("trial-evidence");
        Self {
            directory,
            side: plan::Side {
                binary,
                mode: plan::Mode::EventDisabled,
                side_id: "event-disabled".into(),
            },
            entry: plan::Entry {
                scenario: "fixture".into(),
                pair_index: 0,
                prompt_seed: 1,
                side_order_first: "event-disabled".into(),
            },
            model,
            runtime,
            binary_sha256: hex::encode(Sha256::digest(b"fixture binary bytes")),
            model_sha256: hex::encode(Sha256::digest(b"fixture model bytes")),
            inherited: Default::default(),
            output,
        }
    }
    fn input(&self) -> Input<'_> {
        Input {
            side: &self.side,
            entry: &self.entry,
            binary_sha256: &self.binary_sha256,
            model: &self.model,
            model_sha256: &self.model_sha256,
            native_runtime_root: &self.runtime,
            directory: &self.output,
            max_tokens: 1,
            readiness: Duration::from_secs(1),
            request: Duration::from_secs(1),
            poll: Duration::from_millis(10),
            shutdown: Duration::from_secs(1),
            remaining: Duration::from_secs(100),
            inherited_profile: &self.inherited,
        }
    }
}

#[test]
fn cancelled_trial_refuses_before_admission_or_private_state_creation() {
    let fixture = Fixture::new();
    let cancellation = Cancellation::default();
    cancellation.cancel();
    assert!(execute(&fixture.input(), &cancellation).is_err());
    assert!(!fixture.output.exists());
}

#[test]
fn malformed_admitted_identity_refuses_before_owned_process_admission() {
    let fixture = Fixture::new();
    for binary_changed in [true, false] {
        let mut input = fixture.input();
        let wrong = "invalid".to_string();
        if binary_changed {
            input.binary_sha256 = &wrong;
        } else {
            input.model_sha256 = &wrong;
        }
        assert!(execute(&input, &Cancellation::default()).is_err());
        assert!(!fixture.output.exists());
    }
}

#[test]
fn absent_runtime_root_and_insufficient_whole_trial_budget_never_launch() {
    let fixture = Fixture::new();
    let missing = fixture.directory.path().join("missing-runtime");
    let mut input = fixture.input();
    input.native_runtime_root = &missing;
    assert!(execute(&input, &Cancellation::default()).is_err());
    assert!(!fixture.output.exists());
    let mut input = fixture.input();
    input.remaining = Duration::ZERO;
    assert!(execute(&input, &Cancellation::default()).is_err());
    assert!(!fixture.output.exists());
}

#[test]
fn worker_contract_is_validated_before_launch_input_publication_or_process_creation() {
    let fixture = Fixture::new();
    let mut input = fixture.input();
    input.max_tokens = 0;
    assert!(execute(&input, &Cancellation::default()).is_err());
    assert!(fixture.output.is_dir());
    assert!(!fixture.output.join("worker-input.json").exists());
    assert!(!fixture.output.join("server.stdout.log").exists());
    assert!(!fixture.output.join("worker.stdout.log").exists());
}

fn public(value: &Value) -> String {
    match value {
        Value::Public(value) | Value::Secret(value) => value.to_string_lossy().into_owned(),
    }
}

#[test]
fn launcher_constructs_supported_neutral_host_and_worker_without_executing_either() {
    let mut fixture = Fixture::new();
    fixture
        .inherited
        .insert("CUDA_VISIBLE_DEVICES".into(), "2".into());
    fixture
        .inherited
        .insert("MESH_LLM_EVENT_SYSTEM_TRIAL_MODE".into(), "off".into());
    let state =
        PrivateState::create(fixture.directory.path(), "event-benchmark-launch-fixture").unwrap();
    state.prepare().unwrap();
    let (owner, snapshot, inheritance) =
        launches(&fixture.input(), &state, 54321, Duration::from_secs(5)).unwrap();
    let server = owner.server.unwrap();
    let argv: Vec<_> = server.spec.arguments.iter().map(public).collect();
    assert_eq!(&argv[..3], &["serve", "--local-model-only", "--model"]);
    assert!(
        !argv
            .iter()
            .any(|arg| arg == "--console" || arg == "--headless")
    );
    let strategy = argv
        .iter()
        .position(|arg| arg == "--speculative-strategy")
        .unwrap();
    assert_eq!(argv[strategy + 1], "disabled");
    assert_eq!(
        snapshot[trial_environment::SELECTOR].value,
        serde_json::json!("event-disabled")
    );
    assert_eq!(
        snapshot[trial_environment::GATE].value,
        serde_json::json!(true)
    );
    assert_eq!(
        public(&server.spec.environment[std::ffi::OsStr::new("CUDA_VISIBLE_DEVICES")]),
        "2"
    );
    assert_eq!(
        inheritance.device_and_backend_settings["CUDA_VISIBLE_DEVICES"],
        "2"
    );
    assert_eq!(
        public(
            &server.spec.environment[std::ffi::OsStr::new("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR")]
        ),
        fixture.runtime.to_string_lossy()
    );
    let worker = owner.worker.unwrap();
    let worker_args: Vec<_> = worker.spec.arguments.iter().map(public).collect();
    assert_eq!(
        &worker_args[..3],
        &["automation", "event-benchmark-run", "measurement-worker"]
    );
    assert!(worker.spec.executable.is_absolute());
    assert_eq!(
        owner.api_readiness.as_ref().unwrap(),
        &("http://127.0.0.1:54321/v1".into(), 54321)
    );
    state.finish(Ok::<(), String>(())).unwrap();
}

#[test]
fn coordinator_regular_checks_leave_byte_revalidation_to_owned_worker() {
    let fixture = Fixture::new();
    let expected = fixture.binary_sha256.clone();
    std::fs::write(
        &fixture.side.binary,
        b"changed bytes observed by worker revalidation",
    )
    .unwrap();
    assert!(identity(&fixture.side.binary, &expected).is_ok());
    // Actual byte mismatch refusal is owned by worker_frontends::identity's focused fixture.
    let noncanonical = fixture
        .side
        .binary
        .parent()
        .unwrap()
        .join(".")
        .join(fixture.side.binary.file_name().unwrap());
    // Rust path equality normalizes dot components; parent traversal remains a noncanonical alias.
    let alias = fixture
        .side
        .binary
        .parent()
        .unwrap()
        .join("runtime/..")
        .join(fixture.side.binary.file_name().unwrap());
    assert!(identity(&alias, &expected).is_err());
    assert!(identity(&noncanonical, "invalid").is_err());
}

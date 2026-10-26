use super::*;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExportArtifact {
    path: PathBuf,
    path_in_repo: String,
    sha256: String,
    byte_size: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExportInput {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    artifact: ExportArtifact,
    receipt_request_sha256: String,
    credential_file: PathBuf,
    execution_timeout_ms: u64,
}
fn export(base: &Path) {
    let args = std::fs::read_to_string(base.join("export-argv")).unwrap();
    let args: Vec<_> = args.lines().collect();
    assert_eq!(args[0], "publish-regular-receipt");
    assert_eq!(args[1], "--input");
    assert_eq!(args[3], "--output-directory");
    let bytes = std::fs::read(args[2]).unwrap();
    let typed: ExportInput = serde_json::from_slice(&bytes).unwrap();
    let hash = admission::digest(&serde_json::to_vec(&typed).unwrap());
    let input = serde_json::to_value(&typed).unwrap();
    let receipt = std::fs::read(input["artifact"]["path"].as_str().unwrap()).unwrap();
    assert_eq!(input["artifact"]["sha256"], admission::digest(&receipt));
    assert_eq!(input["artifact"]["byte_size"], receipt.len() as u64);
    let native: Value = serde_json::from_slice(&receipt).unwrap();
    assert_eq!(native["request_sha256"], input["receipt_request_sha256"]);
    std::fs::write(base.join("exported-native.json"), &receipt).unwrap();
    let corrupt = std::fs::read_to_string(base.join("export-mode")).unwrap() == "yes";
    let value = json!({"schema_version":1,"request_sha256":hash,"status":"PUBLISHED_REGULAR_RECEIPT","receipt_request_sha256":input["receipt_request_sha256"],"artifact_sha256":if corrupt {json!("0".repeat(64))}else{input["artifact"]["sha256"].clone()},"input_custody_verified":true,"error":null,"publication":{"schema_version":1,"repo":input["repo"],"parent_commit":input["parent_commit"],"commit_oid":"f".repeat(40),"completed":true,"source_custody_verified":true,"mutation_attempted":true,"remote_verified_paths":[input["artifact"]["path_in_repo"]],"error":null}});
    std::fs::create_dir(args[4]).unwrap();
    std::fs::write(
        Path::new(args[4]).join("publication.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
}
pub(super) fn run() {
    let base = PathBuf::from(std::env::var_os("QUANT_DELIVERY_TEST_ROOT").unwrap());
    match std::env::var("QUANT_DELIVERY_TEST_ROLE").unwrap().as_str() {
        "worker" => {
            let result = super::super::super::run(&[
                "quant-job-worker".into(),
                "--input".into(),
                base.join("worker-input.json").to_str().unwrap().into(),
                "--output-directory".into(),
                base.join("delivery").to_str().unwrap().into(),
            ]);
            if let Err(e) = result {
                panic!("{e}");
            }
        }
        "export" => export(&base),
        _ => panic!("unadmitted finite delivery role"),
    }
}

use super::*;
fn store(c: &Context, repo: &str) -> PathBuf {
    c.base.join("remote").join(repo.replace('/', "_"))
}
fn head(at: &Path) -> String {
    std::fs::read_to_string(at.join("HEAD")).unwrap()
}
pub(super) fn repository(c: &Context) {
    let repo = c.get("--repo");
    let at = store(c, repo);
    let output = c.output();
    std::fs::create_dir(&output).unwrap();
    std::fs::create_dir_all(at.join("0".repeat(40))).unwrap();
    std::fs::write(at.join("HEAD"), "0".repeat(40)).unwrap();
    let opts = RepositoryOptions {
        repo,
        credential_file: c.get("--credential-file"),
        output_directory: &output,
        timeout_seconds: c.get("--timeout-seconds").parse().unwrap(),
        confirm: true,
    };
    write(
        &output.join("repository.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&opts).unwrap()),
      "status":"REPOSITORY_READY","repository":{"repo":repo,"observed_parent":head(&at),"completed":true,"error":null}}),
    );
}
pub(super) fn upload(c: &Context) {
    let path = PathBuf::from(c.get("--artifact"));
    let output = c.output();
    std::fs::create_dir(&output).unwrap();
    let repo = c.get("--repo");
    let name = c.get("--relative-path");
    let at = store(c, repo);
    let prior = head(&at);
    let n = u64::from_str_radix(&prior[24..], 16).unwrap() + 1;
    let commit = format!("{n:040x}");
    let dest = at.join(&commit);
    copy_tree(&at.join(&prior), &dest);
    let remote = dest.join(name);
    std::fs::create_dir_all(remote.parent().unwrap()).unwrap();
    std::fs::copy(&path, &remote).unwrap();
    let id = identity(&path);
    assert_eq!(id, identity(&remote));
    std::fs::write(at.join("HEAD"), &commit).unwrap();
    let unlink = c.args.iter().any(|s| s == "--unlink-after-success");
    let opts = UploadOptions {
        repo,
        revision: c.get("--revision"),
        artifact: &path,
        relative_path: name,
        credential_file: c.get("--credential-file"),
        output_directory: &output,
        maximum_attempts: 8,
        timeout_seconds: c.get("--timeout-seconds").parse().unwrap(),
        dataset: false,
        create_pr: false,
        unlink_after_success: unlink,
        admit_only: false,
        confirm: true,
    };
    if repo == "fixture/package" {
        c.event("package-upload");
    }
    if unlink {
        std::fs::remove_file(&path).unwrap();
    }
    write(
        &output.join("upload.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&opts).unwrap()),"status":"PUBLISHED",
      "publication":{"repo":repo,"revision":"main","path":name,"completed":true,"error":null,"source_custody_verified":true,
      "unlink_requested":unlink,"unlinked":unlink,"identity":id,"attempts":[{"commit_oid":commit,"remote_verified":true,"error":null}]}}),
    );
}
pub(super) fn verify(c: &Context) {
    let output = c.output();
    std::fs::create_dir(&output).unwrap();
    let path = PathBuf::from(c.get("--artifact"));
    let repo = c.get("--repo");
    let commit = c.get("--commit");
    let name = c.get("--relative-path");
    assert_eq!(
        identity(&path),
        identity(&store(c, repo).join(commit).join(name))
    );
    let opts = VerifyOptions {
        repo,
        commit,
        artifact: &path,
        relative_path: name,
        credential_file: c.get("--credential-file"),
        output_directory: &output,
        timeout_seconds: c.get("--timeout-seconds").parse().unwrap(),
    };
    write(
        &output.join("verification.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&serde_json::to_vec(&opts).unwrap()),"status":"IMMUTABLE_VERIFIED",
      "verification":{"repo":repo,"commit":commit,"path":name,"identity":identity(&path),"completed":true,"error":null,"local_custody_verified":true}}),
    );
}
pub(super) fn commit(c: &Context) {
    c.event("common-commit");
    let bytes = std::fs::read(c.get("--input")).unwrap();
    let request: Value = serde_json::from_slice(&bytes).unwrap();
    let output = c.output();
    std::fs::create_dir(&output).unwrap();
    let artifacts = output.join("artifacts");
    std::fs::create_dir(&artifacts).unwrap();
    let snapshot =
        store(c, request["repo"].as_str().unwrap()).join(request["commit"].as_str().unwrap());
    if c.mode == "roster" {
        std::fs::remove_file(snapshot.join("Q4/model-00002-of-00002.gguf")).unwrap();
    }
    let mut verified = vec![];
    let mut complete = true;
    for a in request["artifacts"].as_array().unwrap() {
        let name = a["path"].as_str().unwrap();
        let source = snapshot.join(name);
        if !source.is_file() {
            complete = false;
            break;
        }
        let id = identity(&source);
        if id["sha256"] != a["sha256"] || id["byte_size"] != a["byte_size"] {
            complete = false;
            break;
        }
        let target = artifacts.join(name);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::copy(source, target).unwrap();
        verified.push(a.clone());
    }
    write(
        &output.join("commit.json"),
        &json!({"schema_version":1,"request_sha256":admission::digest(&bytes),"status":if complete{"QUANT_COMMIT_BYTES_VERIFIED"}else{"FAILED"},
      "verification":{"repo":request["repo"],"commit":request["commit"],"artifact_root":artifacts,"verified":verified,"completed":complete,"error":if complete{Value::Null}else{json!("missing immutable roster")}}}),
    );
}

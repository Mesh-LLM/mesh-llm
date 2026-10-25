//! Finite supervised protocol peer: derives receipt fields from actual pinned file bytes.
use super::*;
use serde::Serialize;
use std::io::Write as _;
#[path = "child/native.rs"]
mod native;
#[path = "child/remote.rs"]
mod remote;
pub(super) struct Context {
    base: PathBuf,
    args: Vec<String>,
    mode: String,
}
impl Context {
    fn get(&self, flag: &str) -> &str {
        let i = self.args.iter().position(|s| s == flag).unwrap();
        &self.args[i + 1]
    }
    fn event(&self, s: &str) {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.base.join("events"))
            .unwrap();
        writeln!(f, "{s}").unwrap();
    }
    fn output(&self) -> PathBuf {
        PathBuf::from(self.get("--output-directory"))
    }
    fn json(&self, value: &Value) {
        let mut out = std::io::stdout().lock();
        writeln!(
            out,
            "\nQUANT_JOB_JSON:{}",
            serde_json::to_string(value).unwrap()
        )
        .unwrap();
    }
}
pub(super) fn run() {
    let base = PathBuf::from(std::env::var_os("QUANT_JOB_CHAIN_ROOT").unwrap());
    let args = std::fs::read_to_string(base.join("argv"))
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let c = Context {
        mode: std::fs::read_to_string(base.join("mode")).unwrap(),
        base,
        args,
    };
    match c.args[0].as_str() {
        "run-quant" => native::window(&c),
        "verify-job" => native::verify(&c),
        "write-package" => native::package(&c),
        "verify-package-v2" => native::package_verify(&c),
        "ensure-repo" => remote::repository(&c),
        "upload" => remote::upload(&c),
        "verify-upload" => remote::verify(&c),
        "verify-quant-commit" => remote::commit(&c),
        _ => panic!("unadmitted finite child command"),
    }
}
fn identity(path: &Path) -> Value {
    let b = std::fs::read(path).unwrap();
    json!({"sha256":admission::digest(&b),"byte_size":b.len()})
}
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let p = e.unwrap().path();
        let out = to.join(p.file_name().unwrap());
        if p.is_dir() {
            copy_tree(&p, &out);
        } else {
            std::fs::copy(&p, out).unwrap();
        }
    }
}
fn write(path: &Path, v: &Value) {
    std::fs::write(path, serde_json::to_vec(v).unwrap()).unwrap();
}
#[derive(Serialize)]
struct RepositoryOptions<'a> {
    repo: &'a str,
    credential_file: &'a str,
    output_directory: &'a Path,
    timeout_seconds: u64,
    confirm: bool,
}
#[derive(Serialize)]
struct UploadOptions<'a> {
    repo: &'a str,
    revision: &'a str,
    artifact: &'a Path,
    relative_path: &'a str,
    credential_file: &'a str,
    output_directory: &'a Path,
    maximum_attempts: u8,
    timeout_seconds: u64,
    dataset: bool,
    create_pr: bool,
    unlink_after_success: bool,
    admit_only: bool,
    confirm: bool,
}
#[derive(Serialize)]
struct VerifyOptions<'a> {
    repo: &'a str,
    commit: &'a str,
    artifact: &'a Path,
    relative_path: &'a str,
    credential_file: &'a str,
    output_directory: &'a Path,
    timeout_seconds: u64,
}

//! Inert staging/provisioning/publication children for actual parent CLI orchestration only.
use super::hf_mtp_default_cli::{hash, pin};
use serde_json::{Value as Json, json};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
pub(super) fn wrapper(root: &Path, kind: &str, mode: &str, source: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let quote = |p: &Path| format!("'{}'", p.to_str().unwrap().replace('\'', "'\\''"));
    let binary = root.join(format!("{kind}-helper"));
    let args = root.join(format!("{kind}-arguments"));
    let test = std::env::current_exe().unwrap();
    let script = format!(
        "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > {}\nexport MTP_FIXTURE_ARGS={} MTP_FIXTURE_KIND='{kind}' MTP_FIXTURE_MODE='{mode}' MTP_FIXTURE_SOURCE={}\nexec {} --exact hf_mtp_default_fixture::fixture_native_child --ignored --nocapture\n",
        quote(&args),
        quote(&args),
        quote(source),
        quote(&test)
    );
    std::fs::write(&binary, script).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    binary
}
fn compact(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut quoted, mut escaped) = (false, false);
    for &b in bytes {
        if quoted {
            out.push(b);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                quoted = false;
            }
        } else if b == b'"' {
            quoted = true;
            out.push(b);
        } else if !b.is_ascii_whitespace() {
            out.push(b);
        }
    }
    out
}
fn value(args: &[String], flag: &str) -> PathBuf {
    args.windows(2)
        .find(|a| a[0] == flag)
        .map(|a| PathBuf::from(&a[1]))
        .unwrap()
}
#[test]
#[ignore = "private owned inert child; invoked by owning actual CLI fixture only"]
fn fixture_native_child() {
    let args = std::fs::read_to_string(std::env::var_os("MTP_FIXTURE_ARGS").unwrap())
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let kind = std::env::var("MTP_FIXTURE_KIND").unwrap();
    let mode = std::env::var("MTP_FIXTURE_MODE").unwrap();
    match kind.as_str() {
        "staging" => stage(&args),
        "repository" => repository(&args),
        "publisher" => publish(&args, &mode),
        _ => panic!("closed inert helper kind"),
    }
}
fn stage(args: &[String]) {
    assert_eq!(args.len(), 2);
    assert_eq!(args[0], "--input");
    let raw = std::fs::read(&args[1]).unwrap();
    let request: Json = serde_json::from_slice(&raw).unwrap();
    let root = PathBuf::from(request["output_directory"].as_str().unwrap());
    std::fs::create_dir(&root).unwrap();
    let staged = root.join("mtp-src");
    std::fs::create_dir(&staged).unwrap();
    let original = PathBuf::from(std::env::var_os("MTP_FIXTURE_SOURCE").unwrap());
    let mut pins = std::collections::BTreeMap::new();
    for field in ["checkpoint", "tokenizer_source"] {
        for (n, h) in request[field]["files"].as_object().unwrap() {
            if pins.contains_key(n) {
                continue;
            }
            let bytes = std::fs::read(original.join(n)).unwrap();
            assert_eq!(hash(&bytes), h.as_str().unwrap());
            std::fs::write(staged.join(n), bytes).unwrap();
            pins.insert(n.clone(), h.clone());
        }
    }
    let profile = PathBuf::from(request["tokenizer_profile"].as_str().unwrap());
    let bytes = std::fs::read(profile).unwrap();
    assert_eq!(hash(&bytes), request["tokenizer_profile_sha256"]);
    std::fs::write(root.join("tokenizer-profile.json"), bytes).unwrap();
    let receipt = json!({"schema_version":1,"status":"STAGED_NOT_CONVERTED","error":null,"request_sha256":hash(&compact(&raw)),"request_transport_sha256":hash(&raw),"checkpoint_directory":staged,"checkpoint_files":pins.iter().map(|(n,h)|json!({"path":staged.join(n),"sha256":h})).collect::<Vec<_>>(),"tokenizer_profile":{"path":root.join("tokenizer-profile.json"),"sha256":request["tokenizer_profile_sha256"]},"checkpoint_source":request["checkpoint"],"tokenizer_source":request["tokenizer_source"]});
    std::fs::write(
        root.join("checkpoint-stitch.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
}
fn repository(args: &[String]) {
    assert_eq!(args[0], "ensure-repo");
    assert!(args.iter().any(|a| a == "--confirm"));
    let root = value(args, "--output-directory");
    std::fs::create_dir(&root).unwrap();
    let repo = args.windows(2).find(|a| a[0] == "--repo").unwrap()[1].clone();
    let credential = value(args, "--credential-file");
    let timeout = args
        .windows(2)
        .find(|a| a[0] == "--timeout-seconds")
        .unwrap()[1]
        .parse::<u64>()
        .unwrap();
    #[derive(serde::Serialize)]
    struct Options {
        repo: String,
        credential_file: PathBuf,
        output_directory: PathBuf,
        timeout_seconds: u64,
        confirm: bool,
    }
    let options = Options {
        repo: repo.clone(),
        credential_file: credential,
        output_directory: root.clone(),
        timeout_seconds: timeout,
        confirm: true,
    };
    let receipt = json!({"schema_version":1,"request_sha256":hash(&serde_json::to_vec(&options).unwrap()),"status":"REPOSITORY_READY","repository":{"repo":repo,"observed_parent":"a".repeat(40),"completed":true,"error":null}});
    std::fs::write(
        root.join("repository.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
}
fn publish(args: &[String], mode: &str) {
    assert_eq!(args[0], "publish");
    let raw = std::fs::read(value(args, "--input")).unwrap();
    let input: Json = serde_json::from_slice(&raw).unwrap();
    let root = value(args, "--output-directory");
    std::fs::create_dir(&root).unwrap();
    let shards = input["shards"].as_array().unwrap();
    assert_eq!(shards.len(), 3);
    for item in shards.iter().chain(input["sidecars"].as_array().unwrap()) {
        let actual = pin(Path::new(item["path"].as_str().unwrap()));
        assert_eq!(actual["sha256"], item["sha256"]);
        assert_eq!(
            std::fs::metadata(item["path"].as_str().unwrap())
                .unwrap()
                .len(),
            item["byte_size"]
        );
    }
    let paths = shards
        .iter()
        .chain(input["sidecars"].as_array().unwrap())
        .map(|a| a["path_in_repo"].clone())
        .collect::<Vec<_>>();
    let attempted = shards
        .iter()
        .map(|a| a["path_in_repo"].clone())
        .collect::<Vec<_>>();
    let complete = mode == "ok";
    let publication = json!({"schema_version":1,"repo":input["repo"],"parent_commit":input["parent_commit"],"ordered_paths":paths,"objects":if complete{shards.iter().map(|a|json!({"oid":a["sha256"],"size":a["byte_size"],"mutation_attempted":false,"uploaded_parts":0,"object_present":true,"source_custody_verified":true,"completed":true,"error":null})).collect::<Vec<_>>()}else{vec![]},"object_attempted_paths":if complete{attempted}else{vec![]},"commit_attempted":true,"commit_oid":if complete{Some("d".repeat(40))}else{None},"remote_verified_paths":if complete{paths}else{vec![]},"final_source_custody_verified":complete,"completed":complete,"error":null});
    let receipt = json!({"schema_version":1,"request_sha256":hash(&compact(&raw)),"status":if mode=="held"{"IN_PROGRESS"}else if complete{"PUBLISHED"}else{"FAILED"},"publication":publication,"source_custody_verified":complete,"error":if mode=="fail"{Some("inert publication refusal")}else{None}});
    if mode == "held" {
        std::fs::write(
            root.join("progress.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();
        std::fs::write(root.join("publication-held"), b"observed").unwrap();
        loop {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    std::fs::write(
        root.join("publication.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    if !complete {
        std::process::exit(1);
    }
}

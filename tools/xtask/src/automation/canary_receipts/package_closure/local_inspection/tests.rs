use super::*;
use crate::process::{
    self as supervisor, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness,
};
use serde_json::json;
use std::{collections::BTreeMap, time::Duration};
fn isolated(name: &str) -> bool {
    if std::env::var_os("LOCAL_INSPECTION_FIXTURE_CHILD").is_some() {
        return false;
    }
    let target = format!("{}::{name}", module_path!().split_once("::").unwrap().1);
    let output = supervisor::supervise_raw(
        &ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            cwd: std::env::current_dir().unwrap(),
            arguments: ["--exact", &target, "--nocapture", "--test-threads=1"]
                .into_iter()
                .map(|a| crate::process::Value::Public(a.into()))
                .collect(),
            environment: BTreeMap::from([(
                "LOCAL_INSPECTION_FIXTURE_CHILD".into(),
                crate::process::Value::Public("1".into()),
            )]),
        },
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(output.process.success(), "{:?}", output.process);
    assert!(
        String::from_utf8_lossy(output.stdout.unwrap().as_bytes()).contains("1 passed; 0 failed")
    );
    true
}
fn commit(root: &Path) -> String {
    process::text(root, &["add", "-A"]).unwrap();
    process::text(
        root,
        &[
            "-c",
            "user.name=Local Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    )
    .unwrap();
    process::text(root, &["rev-parse", "HEAD"]).unwrap()
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    authority: LocalRepairSource,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        for dir in [
            "third_party/llama.cpp/patches/model_support",
            "third_party/llama.cpp/patches/generated",
            "ci/llama-canary",
            "docs/skippy",
            "crates/skippy-ffi/src",
            "crates/mesh-llm-host-runtime/src/inference/skippy",
        ] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        fs::write(root.join(".gitignore"), ".deps/\n").unwrap();
        fs::write(
            root.join("third_party/llama.cpp/patches/0001-fixture.patch"),
            "fixture",
        )
        .unwrap();
        for (lane, name) in [
            ("model_support", "0001-model.patch"),
            ("generated", "0001-family-fixture.patch"),
        ] {
            fs::write(
                root.join(format!("third_party/llama.cpp/patches/{lane}/series")),
                format!("{name}\n"),
            )
            .unwrap();
            fs::write(
                root.join(format!("third_party/llama.cpp/patches/{lane}/{name}")),
                "fixture",
            )
            .unwrap();
        }
        fs::write(
            root.join("third_party/llama.cpp/upstream.txt"),
            format!("{}\n", "a".repeat(40)),
        )
        .unwrap();
        fs::write(root.join("crates/skippy-ffi/src/lib.rs"),"pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n").unwrap();
        let family = json!({"policy":{"profiles":{"full":{"status":"certified","required_lanes":["single-step","chain","state-handoff"]}}},"models":[{"family":"fixture","class":"causal_generation","architecture":"fixture","profile":"full","resources":{"estimated_model_bytes":1}}]});
        fs::write(
            root.join("ci/llama-canary/family-certified.json"),
            serde_json::to_vec(&family).unwrap(),
        )
        .unwrap();
        fs::write(root.join("docs/skippy/llama-parity-candidates.json"),serde_json::to_vec(&json!({"candidates":[{"llama_model":"fixture","family":"fixture","status":"needs_candidate"}]})).unwrap()).unwrap();
        process::text(&root, &["init", "--quiet"]).unwrap();
        let base = commit(&root);
        let native = root.join(".deps/llama.cpp");
        fs::create_dir_all(native.join("src/models")).unwrap();
        fs::write(
            native.join("src/models/fixture.cpp"),
            "begin_block(layer); end_block(layer);\n",
        )
        .unwrap();
        process::text(&native, &["init", "--quiet"]).unwrap();
        let head = commit(&native);
        for (name, value) in [
            (".mesh-llm-upstream-sha", "a".repeat(40)),
            (".mesh-llm-patched-sha", head),
            (".mesh-llm-prepare-schema", "5".into()),
            (
                ".mesh-llm-patch-digest",
                source::patch_digest(&root.join("third_party/llama.cpp/patches"))
                    .unwrap()
                    .as_str()
                    .to_owned(),
            ),
        ] {
            fs::write(native.join(name), format!("{value}\n")).unwrap();
        }
        let authority = LocalRepairSource {
            controller: Controller {
                root: root.clone(),
                revision: base.clone(),
                executable_sha256: executable_digest(&std::env::current_exe().unwrap()).unwrap(),
            },
            root: root.clone(),
            base,
        };
        Self {
            _temp: temp,
            root,
            authority,
        }
    }
    fn input(&self) -> Value {
        json!({"authority":{"controller":{"root":self.authority.controller.root,"revision":self.authority.controller.revision,"executable_sha256":self.authority.controller.executable_sha256},"root":self.authority.root,"base":self.authority.base}})
    }
    fn execute(&self, verb: &str, check: Option<bool>) -> DynResult<Value> {
        let mut input = self.input();
        if let Some(check) = check {
            input["check"] = json!(check);
        }
        execute(&serde_json::to_vec(&input)?, verb)
    }
}
#[test]
fn local_authority_composes_dirty_manifest_parity_and_split_write_check_without_workflow_identity()
{
    if isolated(
        "local_authority_composes_dirty_manifest_parity_and_split_write_check_without_workflow_identity",
    ) {
        return;
    }
    process::operation(|| {
        let f = Fixture::new();
        let path = f.root.join("ci/llama-canary/family-certified.json");
        let mut family: Value = serde_json::from_slice(&fs::read(&path)?)?;
        family["models"][0]["resources"]["estimated_model_bytes"] = json!(2);
        fs::write(&path, serde_json::to_vec(&family)?)?;
        assert_eq!(
            f.execute("local-manifest-policy", None)?["status"],
            "agent_manifest_policy_admitted"
        );
        assert_eq!(
            f.execute("local-parity-inventory", None)?["model_sources"],
            1
        );
        f.execute("local-split-roster", Some(false))?;
        f.execute("local-split-roster", Some(true))?;
        fs::write(
            f.root
                .join("crates/mesh-llm-host-runtime/src/inference/skippy/split-certified.json"),
            "stale",
        )?;
        assert!(f.execute("local-split-roster", Some(true)).is_err());
        family["models"][0]["family"] = json!("changed");
        fs::write(&path, serde_json::to_vec(&family)?)?;
        assert!(f.execute("local-manifest-policy", None).is_err());
        Ok(())
    })
    .unwrap();
}
#[test]
fn local_authority_rejects_changed_head_wrong_digest_selected_root_and_cross_schema_fields() {
    if isolated(
        "local_authority_rejects_changed_head_wrong_digest_selected_root_and_cross_schema_fields",
    ) {
        return;
    }
    process::operation(|| {
        let f=Fixture::new();let mut changed=f.authority.clone();changed.controller.executable_sha256=Digest::of_bytes(b"different");assert!(changed.validate().is_err());
        changed=f.authority.clone();changed.base="b".repeat(40);assert!(changed.validate().is_err());
        changed=f.authority.clone();changed.root=f.root.join(".deps/llama.cpp");assert!(changed.validate().is_err());
        for field in ["context","run_id","run_attempt","selected_source","check"] {let mut input=f.input();input[field]=json!("1");assert!(execute(&serde_json::to_vec(&input)?,"local-manifest-policy").is_err());}
        let mut input=f.input();input["authority"]["controller"]["run_id"]=json!("1");assert!(execute(&serde_json::to_vec(&input)?,"local-parity-inventory").is_err());
        let context=json!({"controller_root":f.root,"controller_revision":f.authority.base,"selected_source":"","run_id":"100","run_attempt":"1"});
        let mut workflow=json!({"context":context,"root":f.root,"base":f.authority.base});
        assert!(serde_json::from_value::<super::super::manifest_policy::Input>(workflow.clone()).is_ok());
        workflow["authority"]=f.input()["authority"].clone();
        assert!(serde_json::from_value::<super::super::manifest_policy::Input>(workflow).is_err());
        let local=f.input();assert!(serde_json::from_value::<super::super::manifest_policy::Input>(local.clone()).is_err());assert!(serde_json::from_value::<super::super::split_roster::Input>(local.clone()).is_err());assert!(serde_json::from_value::<super::super::parity_inventory::Input>(local).is_err());
        fs::write(f.root.join("other"),"changed")?;commit(&f.root);assert!(f.authority.validate().is_err());Ok(())
    }).unwrap();
}
#[cfg(unix)]
#[test]
fn executable_identity_rejects_special_paths_before_open_and_has_an_exact_digest() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("regular");
    fs::write(&file, b"actual controller identity").unwrap();
    assert_eq!(
        executable_digest(&file).unwrap(),
        Digest::of_bytes(b"actual controller identity")
    );
    assert!(executable_digest(directory.path()).is_err());
    let link = directory.path().join("link");
    std::os::unix::fs::symlink(&file, &link).unwrap();
    assert!(executable_digest(&link).is_err());
    let fifo = directory.path().join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: owned temporary path is NUL-terminated and mode has no special bits.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(executable_digest(&fifo).is_err());
}

#[cfg(unix)]
fn special(path: &Path, kind: &str, outside: &Path) {
    match kind {
        "fifo" => {
            let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
            // SAFETY: owned fixture pathname is NUL-terminated, ordinary FIFO permission bits.
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        }
        "directory" => fs::create_dir(path).unwrap(),
        "symlink" => std::os::unix::fs::symlink(outside, path).unwrap(),
        "oversize" => fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
            .set_len(8 * 1024 * 1024 + 1)
            .unwrap(),
        _ => panic!("unknown special fixture"),
    }
}
#[cfg(unix)]
fn clear_special(path: &Path, kind: &str) {
    if kind == "directory" {
        fs::remove_dir(path).unwrap();
    } else {
        fs::remove_file(path).unwrap();
    }
}
impl Fixture {
    fn workflow(&self, verb: &str, check: bool) -> DynResult<Value> {
        let context = json!({"controller_root":self.root,"controller_revision":self.authority.base,"selected_source":"","run_id":"100","run_attempt":"1"});
        let input = json!({"context":context,"root":self.root,"base":self.authority.base});
        match verb {
            "manifest" => manifest_policy::execute(&serde_json::from_value(input)?),
            "parity" => {
                let mut input = input;
                input["source_revision"] = json!(self.authority.base);
                parity_inventory::execute(&serde_json::from_value(input)?)
            }
            "split" => split_roster::execute(&serde_json::from_value(
                json!({"context":context,"root":self.root,"check":check}),
            )?),
            _ => Err("invalid fixture verb".into()),
        }
    }
}
#[cfg(unix)]
#[test]
fn policy_documents_reject_special_escaping_and_oversize_sources_in_local_and_workflow_compositions()
 {
    if isolated(
        "policy_documents_reject_special_escaping_and_oversize_sources_in_local_and_workflow_compositions",
    ) {
        return;
    }
    process::operation(|| {
        let f = Fixture::new();
        let outside = tempfile::tempdir()?;
        let target = outside.path().join("foreign.json");
        fs::write(&target, b"foreign sentinel")?;
        for relative in [
            "ci/llama-canary/family-certified.json",
            "docs/skippy/llama-parity-candidates.json",
        ] {
            let path = f.root.join(relative);
            let original = fs::read(&path)?;
            fs::write(&target, &original)?;
            fs::remove_file(&path)?;
            for kind in ["fifo", "directory", "symlink", "oversize"] {
                special(&path, kind, &target);
                assert!(f.execute("local-manifest-policy", None).is_err());
                assert!(f.execute("local-parity-inventory", None).is_err());
                assert!(f.workflow("manifest", false).is_err());
                assert!(f.workflow("parity", false).is_err());
                if relative.starts_with("ci/") {
                    assert!(f.execute("local-split-roster", Some(false)).is_err());
                    assert!(f.workflow("split", false).is_err());
                }
                assert_eq!(fs::read(&target)?, original);
                clear_special(&path, kind);
            }
            fs::write(&path, &original)?;
            let parent = path.parent().unwrap();
            let saved = parent.with_extension("saved");
            fs::write(outside.path().join(path.file_name().unwrap()), &original)?;
            fs::rename(parent, &saved)?;
            std::os::unix::fs::symlink(outside.path(), parent)?;
            assert!(f.execute("local-manifest-policy", None).is_err());
            assert!(f.execute("local-parity-inventory", None).is_err());
            assert!(f.workflow("manifest", false).is_err());
            assert!(f.workflow("parity", false).is_err());
            fs::remove_file(parent)?;
            fs::rename(saved, parent)?;
        }
        f.execute("local-manifest-policy", None)?;
        f.workflow("manifest", false)?;
        Ok(())
    })
    .unwrap();
}
#[cfg(unix)]
#[test]
fn split_output_rejects_special_paths_in_local_and_workflow_check_and_write_without_escaped_writes()
{
    if isolated(
        "split_output_rejects_special_paths_in_local_and_workflow_check_and_write_without_escaped_writes",
    ) {
        return;
    }
    process::operation(|| {
        let f = Fixture::new();
        let parent = f
            .root
            .join("crates/mesh-llm-host-runtime/src/inference/skippy");
        let output = parent.join("split-certified.json");
        let outside = tempfile::tempdir()?;
        let target = outside.path().join("split-certified.json");
        fs::write(&target, b"foreign sentinel")?;
        assert!(f.execute("local-split-roster", Some(true)).is_err());
        f.execute("local-split-roster", Some(false))?;
        f.workflow("split", true)?;
        fs::remove_file(&output)?;
        for kind in ["fifo", "directory", "symlink"] {
            special(&output, kind, &target);
            for check in [true, false] {
                assert!(f.execute("local-split-roster", Some(check)).is_err());
                assert!(f.workflow("split", check).is_err());
            }
            assert_eq!(fs::read(&target)?, b"foreign sentinel");
            clear_special(&output, kind);
        }
        special(&output, "oversize", &target);
        assert!(f.execute("local-split-roster", Some(true)).is_err());
        f.execute("local-split-roster", Some(false))?;
        f.workflow("split", true)?;
        fs::remove_file(&output)?;
        let saved = parent.with_file_name("skippy.saved");
        fs::rename(&parent, &saved)?;
        std::os::unix::fs::symlink(outside.path(), &parent)?;
        for check in [true, false] {
            assert!(f.execute("local-split-roster", Some(check)).is_err());
            assert!(f.workflow("split", check).is_err());
        }
        assert_eq!(fs::read(&target)?, b"foreign sentinel");
        fs::remove_file(&parent)?;
        fs::rename(saved, parent)?;
        Ok(())
    })
    .unwrap();
}

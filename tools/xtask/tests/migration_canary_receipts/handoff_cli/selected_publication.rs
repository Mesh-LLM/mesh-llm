//! Actual publication CLI with an independently certified selected-source package.
use super::*;
use std::path::Path;

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("/usr/bin/git")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
        ])
        .args(args)
        .current_dir(root)
        .env("GIT_MASTER", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().into()
}
fn fixture(pass: &str) -> (Fixture, Value, String) {
    let (fixture, mut input) = package(pass, false);
    let root = fixture.0.join("source");
    fs::create_dir(&root).unwrap();
    git(&root, &["init", "-b", "main"]);
    fs::write(root.join("base"), "base").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-m", "base"]);
    git(&root, &["checkout", "-b", "selected"]);
    fs::write(root.join("candidate"), "selected changes").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-m", "selected"]);
    let selected = git(&root, &["rev-parse", "HEAD"]);
    git(&root, &["checkout", "main"]);
    fs::write(root.join("controller"), "protected controller changes").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-m", "controller"]);
    let controller = git(&root, &["rev-parse", "HEAD"]);
    input["controller_revision"] = json!(controller);
    input["selected_source"] = json!(selected);
    input["source_root"] = json!(root);
    input["repository"] = json!("Mesh-LLM/mesh-llm");
    let path = fixture.0.join("package/identity.json");
    let mut identity: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    identity["candidate"] = json!(selected);
    identity["base"] = json!(selected);
    identity["mesh_source"] = json!(selected);
    identity["controller"] = json!(controller);
    identity["branch"] = json!("llama-canary/repair-123-4-aaaaaaaaaa");
    let bytes = serde_json::to_vec(&identity).unwrap();
    fs::write(path, &bytes).unwrap();
    input["identity_sha256"] = json!(Digest::of_bytes(&bytes));
    (fixture, input, selected)
}

#[test]
fn selected_publication_exports_exact_commit_and_keeps_controller_checkout() {
    let (fixture, input, selected) = fixture("verify-1");
    let output = invoke(&fixture, &input, "publication");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let root = fixture.0.join("source");
    let bundle = fixture.0.join("package/candidate.bundle");
    let heads = git(&root, &["bundle", "list-heads", bundle.to_str().unwrap()]);
    assert_eq!(
        heads,
        format!("{selected} refs/heads/llama-canary/repair-123-4-aaaaaaaaaa")
    );
    assert_eq!(
        git(&root, &["rev-parse", "HEAD"]),
        input["controller_revision"].as_str().unwrap()
    );
    assert!(!root.join("candidate").exists());
    let body = fs::read_to_string(fixture.0.join("package/pr-body.md")).unwrap();
    assert!(body.contains(&selected));
    assert!(body.contains("Both complete per-family passes succeeded"));
    assert!(body.contains("actions/runs/123"));
    // The bundle restores into the protected controller's history, including all
    // selected-branch commits; it does not require a candidate-owned bootstrap.
    git(&root, &["bundle", "verify", bundle.to_str().unwrap()]);
}

#[test]
fn selected_publication_rejects_first_pass_foreign_source_and_wrong_controller() {
    for case in [
        "first-pass",
        "foreign-source",
        "wrong-controller",
        "missing-root",
        "corrupt",
    ] {
        let (fixture, mut input, _) = fixture(if case == "first-pass" {
            "repair-1"
        } else {
            "verify-1"
        });
        match case {
            "foreign-source" => input["selected_source"] = json!("c".repeat(40)),
            "wrong-controller" => {
                git(&fixture.0.join("source"), &["checkout", "selected"]);
            }
            "missing-root" => {
                input.as_object_mut().unwrap().remove("source_root");
            }
            "corrupt" => fs::write(fixture.0.join("package/binaries.tar"), "corrupt").unwrap(),
            _ => (),
        }
        assert!(
            !invoke(&fixture, &input, "publication").status.success(),
            "{case}"
        );
        assert!(!fixture.0.join("package/pr-body.md").exists());
        assert!(!fixture.0.join("package/candidate.bundle").exists());
    }
}

#[test]
fn selected_publication_refuses_main_itself_and_never_replaces_existing_branch() {
    for no_changes in [true, false] {
        let (fixture, mut input, _) = fixture("verify-1");
        let root = fixture.0.join("source");
        if no_changes {
            let controller = input["controller_revision"].clone();
            let path = fixture.0.join("package/identity.json");
            let mut identity: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            for key in ["candidate", "base", "mesh_source"] {
                identity[key] = controller.clone();
            }
            let bytes = serde_json::to_vec(&identity).unwrap();
            fs::write(path, &bytes).unwrap();
            input["identity_sha256"] = json!(Digest::of_bytes(&bytes));
            input["selected_source"] = controller;
        } else {
            git(
                &root,
                &["branch", "llama-canary/repair-123-4-aaaaaaaaaa", "HEAD"],
            );
        }
        assert!(!invoke(&fixture, &input, "publication").status.success());
        assert!(!fixture.0.join("package/candidate.bundle").exists());
        assert!(!fixture.0.join("package/pr-body.md").exists());
        if !no_changes {
            assert_eq!(
                git(
                    &root,
                    &["rev-parse", "llama-canary/repair-123-4-aaaaaaaaaa"]
                ),
                input["controller_revision"].as_str().unwrap()
            );
        }
    }
}

use super::*;

const CLIENT: &str = include_str!("../../../../../mesh/deploy/docker/Dockerfile.client");
const FLY: &str = include_str!("../../../../../mesh/deploy/fly/Dockerfile");
const ENTRYPOINT: &str = include_str!("../../../../../mesh/deploy/docker/entrypoint.sh");
const WORKFLOW: &str = include_str!("../../../../../.github/workflows/docker.yml");
const ALL: Policy = Policy {
    shared: true,
    fly: true,
    entrypoint: true,
    no_qemu: true,
};

fn admit(source: &str) -> DynResult<()> {
    check(source, ALL, Some(ENTRYPOINT), Some(WORKFLOW))
}

#[test]
fn current_product_docker_sources_satisfy_policy() {
    admit(CLIENT).unwrap();
    admit(FLY).unwrap();
}

#[test]
fn each_explicit_tree_and_manifest_copy_is_causal_and_blanket_copy_refused() {
    for path in [
        "mesh/crates/",
        "skippy/crates/",
        "tools/xtask/",
        "scripts/",
        "mesh/scripts/",
        "skippy/scripts/",
    ] {
        let changed = CLIENT.replace(&format!("COPY {path} {path}"), "# removed tree copy");
        assert!(admit(&changed).is_err(), "{path}");
    }
    assert!(
        admit(&CLIENT.replace("COPY Cargo.toml Cargo.lock ./", "# removed manifest copy")).is_err()
    );
    for copy in [
        "COPY . .",
        "COPY . . # trailing comment",
        "COPY --chown=root . .",
        "COPY [\".\",\".\"]",
        "COPY --chown=root [\".\", \".\"]",
        "COPY --chown=root ./ ./",
        "COPY\t. .",
    ] {
        assert!(admit(&format!("{CLIENT}\n{copy}\n")).is_err());
    }
    for copy in [
        "COPY --from=builder . .",
        "COPY --from=builder [\".\", \".\"]",
        "COPY --chown=root --from=builder . .",
    ] {
        admit(&format!("{CLIENT}\n{copy}\n")).unwrap();
    }
    let comment = CLIENT.replace(
        "COPY mesh/crates/ mesh/crates/",
        "# COPY mesh/crates/ mesh/crates/",
    );
    assert!(admit(&comment).is_err());
}

#[test]
fn ui_order_patch_preparation_runtime_libraries_and_fly_stage_are_causal() {
    for fragment in [
        "COPY --from=ui-builder",
        "cargo build",
        "scripts/prepare-llama.sh pinned",
        "scripts/build-llama.sh",
        "skippy/llama_cpp/patches",
        "ca-certificates",
        "libgomp1",
        "libdbus-1-3",
        "AS ui-builder",
    ] {
        assert!(
            admit(&CLIENT.replace(fragment, "removed-contract")).is_err(),
            "{fragment}"
        );
    }
    assert!(admit(&format!("RUN cargo build\n{CLIENT}")).is_err());
}

#[test]
fn entrypoint_modes_and_no_qemu_are_causal_without_executing_them() {
    for source in ["qemu", "setup-QEMU-action", "# qEmU tooling", "(qemu)"] {
        assert!(has_qemu_token(source), "{source}");
    }
    for source in ["qemu_user", "myqemu", "qemu2", "_QEMU_", ""] {
        assert!(!has_qemu_token(source), "{source}");
    }
    for source in [
        "console) exec app ;;\nworker) exec worker ;;\n*) exec app ;;",
        "console|\"\") exec app ;;\nworker|node) exec worker ;;\n*) exec app ;;",
    ] {
        entrypoint_modes(source).unwrap();
        for api in [
            "api) exec app ;;",
            "api|worker) exec app ;;",
            "worker|api) exec app ;;",
            "\"api\") exec app ;;",
        ] {
            assert!(
                entrypoint_modes(&format!("{source}\n{api}\n")).is_err(),
                "{api}"
            );
        }
    }
    for label in ["console|", "worker)", "*)"] {
        assert!(
            check(
                CLIENT,
                ALL,
                Some(&ENTRYPOINT.replace(label, "removed-mode")),
                Some(WORKFLOW)
            )
            .is_err()
        );
    }
    assert!(
        check(
            CLIENT,
            ALL,
            Some(&format!("{ENTRYPOINT}\napi)\n")),
            Some(WORKFLOW)
        )
        .is_err()
    );
    assert!(
        check(
            CLIENT,
            ALL,
            Some(ENTRYPOINT),
            Some(&format!("{WORKFLOW}\n# uses setup-QEMU-action\n"))
        )
        .is_err()
    );
    assert!(
        check(
            "FROM scratch",
            Policy {
                shared: false,
                fly: false,
                entrypoint: false,
                no_qemu: false
            },
            None,
            None
        )
        .is_ok()
    );
}

#[test]
fn bounded_sources_refuse_missing_escape_directory_and_oversize() {
    let root = tempfile::tempdir().unwrap();
    assert!(read(root.path(), "missing").is_err());
    assert!(read(root.path(), "../outside").is_err());
    assert!(read(root.path(), "/etc/passwd").is_err());
    fs::create_dir(root.path().join("directory")).unwrap();
    assert!(read(root.path(), "directory").is_err());
    fs::write(root.path().join("large"), vec![b'a'; 1048577]).unwrap();
    assert!(read(root.path(), "large").is_err());
    fs::write(root.path().join("Dockerfile"), CLIENT).unwrap();
    assert_eq!(read(root.path(), "Dockerfile").unwrap(), CLIENT);
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("Dockerfile", root.path().join("link")).unwrap();
        assert!(read(root.path(), "link").is_err());
        use std::os::unix::ffi::OsStrExt as _;
        let fifo = std::ffi::CString::new(root.path().join("fifo").as_os_str().as_bytes()).unwrap();
        // SAFETY: the NUL-terminated path names only this test's temporary directory.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        assert!(read(root.path(), "fifo").is_err());
    }
}

#[test]
fn malformed_cli_boolean_and_duplicate_inputs_refuse_before_source_read() {
    for args in [
        vec!["--shared-core", "yes"],
        vec![
            "--dockerfile",
            "one",
            "--dockerfile",
            "two",
            "--shared-core",
            "false",
            "--fly-ui-builder",
            "false",
            "--entrypoint-modes",
            "false",
            "--workflow-no-qemu",
            "false",
        ],
    ] {
        let args = args.into_iter().map(String::from).collect::<Vec<_>>();
        assert!(run(&args, || panic!("must refuse before source discovery")).is_err());
    }
}

#[test]
fn current_workflow_uses_one_env_bound_native_owner_after_uncached_hosted_bootstrap() {
    let source = include_str!("../../../../../.github/workflows/docker-precheck.yml");
    assert!(source.contains("name: ${{ inputs.job_name }}"));
    assert!(source.contains("persist-credentials: false"));
    assert!(source.contains("runner-profile: hosted-bare"));
    assert!(source.contains("allow_depot_remote_cache: \"false\""));
    assert!(source.contains("allow_native_github_cache: \"false\""));
    assert!(source.contains("DOCKERFILE_PATH: ${{ inputs.dockerfile_path }}"));
    let run = source.split_once("        run: |\n").unwrap().1;
    assert!(run.contains("\"$MESH_LLM_AUTOMATION_BIN\" repository docker-precheck"));
    assert!(!run.contains("${{"));
    assert!(!run.contains("grep"));
    for argument in [
        "--dockerfile \"$DOCKERFILE_PATH\"",
        "--shared-core \"$SHARED_CORE\"",
        "--fly-ui-builder \"$FLY_UI_BUILDER\"",
        "--entrypoint-modes \"$ENTRYPOINT_MODES\"",
        "--workflow-no-qemu \"$WORKFLOW_NO_QEMU\"",
    ] {
        assert!(run.contains(argument), "{argument}");
    }
}

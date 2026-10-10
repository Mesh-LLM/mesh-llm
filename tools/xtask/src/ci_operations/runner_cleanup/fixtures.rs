use super::{Options, Plan, Profile, boundary};
#[path = "../../../tests/migration_lifecycle/cleanup_owner.rs"]
pub(super) mod ownership;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

pub(super) struct Fixture {
    pub root: tempfile::TempDir,
    pub workspace: PathBuf,
    pub temporary: PathBuf,
    pub env: BTreeMap<OsString, OsString>,
}
impl Fixture {
    pub fn new() -> Self {
        let root = tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap();
        ownership::register(&root);
        let location = root.path().canonicalize().unwrap();
        let workspace = location.join("workspace");
        let temporary = location.join("temp");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&temporary).unwrap();
        let mut env: BTreeMap<OsString, OsString> = [
            ("GITHUB_RUN_ID", "123"),
            ("GITHUB_RUN_ATTEMPT", "2"),
            ("CANARY_PASS_ID", "repair-1"),
            ("CANARY_SHARD_INDEX", "3"),
            ("CLEANUP_ARTIFACT_PATH", "ci-artifacts/linux"),
            ("CLEANUP_BINARY_PATH", "target/release/mesh-llm"),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect();
        env.insert(
            "GITHUB_WORKSPACE".into(),
            workspace.clone().into_os_string(),
        );
        env.insert("RUNNER_TEMP".into(), temporary.clone().into_os_string());
        env.insert(
            "CANARY_SOURCE_ROOT".into(),
            workspace.clone().into_os_string(),
        );
        Self {
            root,
            workspace,
            temporary,
            env,
        }
    }
    pub fn plan(&self, profile: Profile, uploaded: bool) -> Plan {
        boundary::from_environment(
            &Options {
                profile,
                evidence_uploaded: uploaded,
                package_uploaded: uploaded,
            },
            &self.env,
        )
        .unwrap()
    }
    pub fn seed(&self, path: &Path) {
        ownership::check(&self.root, path).unwrap();
        assert!(!path.is_symlink());
        std::fs::create_dir_all(path).unwrap();
        std::fs::write(path.join("payload"), b"sentinel").unwrap();
    }
    pub fn check_plan(&self, plan: &Plan) {
        for target in &plan.targets {
            assert!(target.base == self.workspace || target.base == self.temporary);
            ownership::check(&self.root, &target.base).unwrap();
            ownership::check(&self.root, &target.path).unwrap();
        }
        if let Some((workspace, replay)) = &plan.replay {
            assert_eq!(workspace, &self.workspace);
            assert_eq!(replay, &self.temporary.join("agentic-replay-worktrees"));
            ownership::check(&self.root, workspace).unwrap();
            ownership::check(&self.root, replay).unwrap();
        }
    }

    pub fn delete(&self, plan: &Plan) -> Vec<u8> {
        self.check_plan(plan);
        let admitted = super::admission::admit(plan).unwrap();
        let interrupt = interrupt().unwrap();
        let mut output = Vec::new();
        super::deletion::execute(admitted, None, (&interrupt, &mut output)).unwrap();
        interrupt.finish().unwrap();
        output
    }
}

// Signal registration is process-global; parallel fixtures must wait for the
// preceding fixture to release it. Production still rejects overlapping owners.
pub(super) fn interrupt()
-> Result<crate::command_interrupt::Interrupt, crate::command_interrupt::Reason> {
    use crate::command_interrupt::{Interrupt, Reason};
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match Interrupt::install() {
            Err(Reason::ScopeBusy) if std::time::Instant::now() < deadline => {
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
            result => return result,
        }
    }
}

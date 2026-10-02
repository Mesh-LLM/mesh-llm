use crate::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

pub(super) fn source(path: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path),
    )
    .unwrap()
}

pub(super) struct Fixture(pub tempfile::TempDir);
impl Fixture {
    pub(super) fn new() -> Self {
        let fixture = Self(
            tempfile::Builder::new()
                .prefix("build contract ")
                .tempdir()
                .unwrap(),
        );
        fs::create_dir_all(fixture.root().join("scripts/lib")).unwrap();
        fs::create_dir(fixture.root().join("bin")).unwrap();
        for name in [
            "build-host.sh",
            "build-release.sh",
            "build-development-product.sh",
            "build-linux.sh",
            "lib/macos-deployment-target.sh",
            "lib/macos-deployment-target.txt",
        ] {
            fixture.write(
                &format!("scripts/{name}"),
                &source(&format!("scripts/{name}")),
            );
        }
        fixture.write("bin/uname", "#!/bin/sh\nprintf 'Linux\\n'\n");
        fixture.write("bin/git", "#!/bin/sh\nset -eu\nif [ \"$1\" = -C ]; then shift 2; fi\ncase \"$1\" in rev-parse) printf 'abc123\\n';; status) :;; *) exit 81;; esac\n");
        fixture.write(
            "bin/cargo",
            r#"#!/bin/sh
set -eu
case "$1" in
  pkgid) printf '%s\n' "$@" > "$BUILD_FIXTURE_ROOT/pkgid.args"; printf 'file:///fixture/mesh-llm#0.68.0\n' ;;
  build)
    printf '%s\n' "$@" > "$BUILD_FIXTURE_ROOT/cargo.args"
    printf '%s\n' "${MESH_LLM_BUILD_VERSION:-<unset>}" > "$BUILD_FIXTURE_ROOT/version"
    printf 'cargo\n' >> "$BUILD_FIXTURE_ROOT/events"
    ;;
  *) exit 82 ;;
esac
"#,
        );
        fixture.write(
            "scripts/build-ui.sh",
            "#!/bin/sh\nprintf 'ui\\n' >> \"$BUILD_FIXTURE_ROOT/events\"\n",
        );
        for name in ["prepare-llama.sh", "build-llama.sh"] {
            fixture.write(&format!("scripts/{name}"), "#!/bin/sh\nprintf 'forbidden-native\\n' >> \"$BUILD_FIXTURE_ROOT/events\"\nexit 83\n");
        }
        fixture
    }
    pub(super) fn root(&self) -> &Path {
        self.0.path()
    }
    pub(super) fn write(&self, name: &str, text: &str) {
        let path = self.root().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    }
    pub(super) fn log(&self, name: &str) -> String {
        let path = self.root().join(name);
        if path.exists() {
            fs::read_to_string(path).unwrap()
        } else {
            String::new()
        }
    }
    pub(super) fn invoke(
        &self,
        script: &str,
        args: &[&str],
        extra: &[(&str, &str)],
    ) -> process::RawProcessReport {
        let mut environment = BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(
                    format!("{}:/usr/bin:/bin", self.root().join("bin").display()).into(),
                ),
            ),
            ("HOME".into(), Value::Public(self.root().into())),
            (
                "BUILD_FIXTURE_ROOT".into(),
                Value::Public(self.root().into()),
            ),
            ("CI".into(), Value::Public("true".into())),
        ]);
        for (name, value) in extra {
            environment.insert((*name).into(), Value::Public((*value).into()));
        }
        let result = process::supervise_raw(
            &ProcessSpec {
                executable: PathBuf::from("/bin/bash"),
                cwd: self.root().into(),
                arguments: std::iter::once(script)
                    .chain(args.iter().copied())
                    .map(|arg| Value::Public(arg.into()))
                    .collect(),
                environment,
            },
            &Limits {
                execution: Duration::from_secs(8),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(
            result.process.cleanup.complete
                && result.process.failure.is_none()
                && result.stdout.is_some()
                && result.stderr.is_some(),
            "{result:?}"
        );
        result
    }
}

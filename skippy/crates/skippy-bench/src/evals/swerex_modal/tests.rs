//! Synthetic byte fixtures exercise filesystem policy, not SDK interface compatibility.
use super::*;
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static ID: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    root: PathBuf,
    environment: PathBuf,
    package: PathBuf,
    patches: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "swerex-modal-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let environment = root.join("private environment");
        let package = environment.join("lib/site-packages/swerex");
        let patches = root.join("pinned patches");
        fs::create_dir_all(&package).unwrap();
        fs::create_dir_all(&patches).unwrap();
        fs::write(package.join("__init__.py"), b"SDK initializer fixture").unwrap();
        let metadata = environment.join("lib/site-packages/swe_rex-1.4.0.dist-info/METADATA");
        fs::create_dir_all(metadata.parent().unwrap()).unwrap();
        fs::write(metadata, b"Version: 1.4.0\n").unwrap();
        Self {
            root,
            environment,
            package,
            patches,
        }
    }
    fn populate(&self) {
        for mapping in contract::MAPPINGS {
            let source = self.patches.join(mapping.source);
            let destination = self.package.join(mapping.destination);
            fs::create_dir_all(source.parent().unwrap()).unwrap();
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            fs::write(source, b"replacement fixture").unwrap();
            fs::write(&destination, b"original fixture").unwrap();
            fs::set_permissions(destination, fs::Permissions::from_mode(0o640)).unwrap();
        }
    }
    fn apply(&self) -> Result<()> {
        let replacement = digest(b"replacement fixture");
        let original = digest(b"original fixture");
        let mappings = contract::MAPPINGS.map(|mapping| contract::Mapping {
            source: mapping.source,
            destination: mapping.destination,
            source_sha256: &replacement,
            replacement_sha256: &replacement,
            original_sha256: Some(&original),
        });
        apply_with(
            &self.package.join("__init__.py"),
            &self.environment,
            &self.patches,
            Some(&digest(b"Version: 1.4.0\n")),
            &mappings,
            |_, _| Ok(b"replacement fixture".to_vec()),
        )
    }
    fn originals_unchanged(&self) {
        for mapping in contract::MAPPINGS {
            let path = self.package.join(mapping.destination);
            assert_eq!(fs::read(&path).unwrap(), b"original fixture");
            assert!(!path.with_extension("py.bak").exists());
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn all_three_semantic_destinations_patch_once_and_preserve_original_backup_and_mode() {
    let fixture = Fixture::new();
    fixture.populate();
    fixture.apply().unwrap();
    fixture.apply().unwrap();
    for mapping in contract::MAPPINGS {
        let path = fixture.package.join(mapping.destination);
        assert_eq!(fs::read(&path).unwrap(), b"replacement fixture");
        assert_eq!(
            fs::read(path.with_extension("py.bak")).unwrap(),
            b"original fixture"
        );
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }
    assert!(
        !fixture
            .package
            .join("deployment/runtime/remote.py")
            .exists()
    );
}

#[test]
fn unknown_final_destination_refuses_before_any_replacement_or_backup() {
    let fixture = Fixture::new();
    fixture.populate();
    fs::write(
        fixture.package.join("runtime/remote.py"),
        b"unknown SDK source",
    )
    .unwrap();
    assert!(fixture.apply().is_err());
    for relative in ["deployment/modal.py", "deployment/config.py"] {
        let path = fixture.package.join(relative);
        assert_eq!(fs::read(&path).unwrap(), b"original fixture");
        assert!(!path.with_extension("py.bak").exists());
    }
}

#[test]
fn replacement_source_drift_refuses_whole_batch_before_writes() {
    let fixture = Fixture::new();
    fixture.populate();
    fs::write(
        fixture.patches.join("swerex/deployment/runtime/remote.py"),
        b"wrong pinned source",
    )
    .unwrap();
    assert!(fixture.apply().is_err());
    fixture.originals_unchanged();
}

#[test]
fn invalid_backup_is_refused_even_for_original_and_already_patched_sources() {
    for patched in [false, true] {
        let fixture = Fixture::new();
        fixture.populate();
        if patched {
            fixture.apply().unwrap();
        }
        fs::write(
            fixture.package.join("deployment/modal.py.bak"),
            b"foreign backup",
        )
        .unwrap();
        assert!(fixture.apply().is_err());
        let expected = if patched {
            b"replacement fixture".as_slice()
        } else {
            b"original fixture".as_slice()
        };
        assert_eq!(
            fs::read(fixture.package.join("deployment/config.py")).unwrap(),
            expected
        );
    }
}

#[test]
fn admitted_partial_previous_patch_can_resume_without_replacing_original_backup() {
    let fixture = Fixture::new();
    fixture.populate();
    let path = fixture.package.join("deployment/modal.py");
    let source = AdmittedSource::open(&path, &fixture.environment).unwrap();
    source.preserve_backup().unwrap();
    source.replace(b"replacement fixture").unwrap();
    fixture.apply().unwrap();
    assert_eq!(
        fs::read(path.with_extension("py.bak")).unwrap(),
        b"original fixture"
    );
    assert_eq!(
        fs::read(fixture.package.join("runtime/remote.py")).unwrap(),
        b"replacement fixture"
    );
}

#[test]
fn symlinked_destination_or_source_is_refused_without_following_it() {
    for replacement in [false, true] {
        let fixture = Fixture::new();
        fixture.populate();
        let path = if replacement {
            fixture.patches.join("swerex/deployment/modal.py")
        } else {
            fixture.package.join("deployment/modal.py")
        };
        let outside = fixture.root.join("outside");
        fs::write(&outside, b"must remain unchanged").unwrap();
        fs::remove_file(&path).unwrap();
        symlink(&outside, path).unwrap();
        assert!(fixture.apply().is_err());
        assert_eq!(fs::read(outside).unwrap(), b"must remain unchanged");
        assert!(!fixture.package.join("deployment/config.py.bak").exists());
    }
}

#[test]
fn wrong_environment_and_changed_distribution_metadata_refuse_before_writes() {
    let fixture = Fixture::new();
    fixture.populate();
    fs::write(
        fixture
            .environment
            .join("lib/site-packages/swe_rex-1.4.0.dist-info/METADATA"),
        b"Version: 1.1.0\n",
    )
    .unwrap();
    assert!(fixture.apply().is_err());
    fixture.originals_unchanged();
    let locator = fixture.package.join("__init__.py");
    assert!(AdmittedSource::open(&locator, &fixture.patches).is_err());
}

#[test]
fn actual_official_sdk_transform_preserves_modern_interfaces_and_safe_retry_boundary() {
    let modal =
        include_bytes!("../../../tests/fixtures/swerex_modal/official/deployment/modal.py.fixture");
    let remote =
        include_bytes!("../../../tests/fixtures/swerex_modal/official/runtime/remote.py.fixture");
    let config = include_bytes!(
        "../../../tests/fixtures/swerex_modal/official/deployment/config.py.fixture"
    );
    for (mapping, original) in
        contract::MAPPINGS
            .iter()
            .zip([modal.as_slice(), config.as_slice(), remote.as_slice()])
    {
        assert_eq!(digest(original), mapping.original_sha256.unwrap());
        let updated = transform::rewrite(mapping.destination, original).unwrap();
        assert_eq!(digest(&updated), mapping.replacement_sha256);
    }
    assert_eq!(
        transform::rewrite("deployment/config.py", config).unwrap(),
        config
    );
    let modal =
        String::from_utf8(transform::rewrite("deployment/modal.py", modal).unwrap()).unwrap();
    for retained in [
        "context_dir=str(build_context)",
        "await modal.Sandbox.create.aio(",
        "await self._sandbox.tunnels.aio()",
        "async def get_modal_log_url",
        "def from_ecr",
        "Ignoring duplicate start() call.",
    ] {
        assert!(modal.contains(retained), "lost {retained}");
    }
    assert!(modal.contains("pyenv install 3.11.13"));
    let remote =
        String::from_utf8(transform::rewrite("runtime/remote.py", remote).unwrap()).unwrap();
    assert!(remote.contains("headers[\"X-Request-ID\"] = request_id"));
    assert!(remote.contains("except (aiohttp.ClientError, asyncio.TimeoutError) as error:"));
    assert!(remote.contains("error.status not in (408, 429) and not 500 <= error.status < 600"));
    assert!(remote.contains("error.status == 511"));
    for (endpoint, request, response) in [
        ("run_in_session", "action", "Observation"),
        ("execute", "command", "CommandResponse"),
        ("write_file", "request", "WriteFileResponse"),
    ] {
        assert!(remote.contains(&format!(
            "self._request(\"{endpoint}\", {request}, {response})"
        )));
    }
    assert_eq!(remote.matches("num_retries=4)").count(), 4);
    assert_eq!(
        remote
            .matches("timeout=aiohttp.ClientTimeout(total=self._get_timeout()),")
            .count(),
        3
    );
    assert!(!remote.contains("import requests"));
}

#[test]
fn transferred_runtime_response_processing_is_outside_transport_retry_catch() {
    let original =
        include_bytes!("../../../tests/fixtures/swerex_modal/official/runtime/remote.py.fixture");
    let updated =
        String::from_utf8(transform::rewrite("runtime/remote.py", original).unwrap()).unwrap();
    let request = updated
        .split("    async def _request(")
        .nth(1)
        .unwrap()
        .split("    async def create_session(")
        .next()
        .unwrap();
    let catch = request
        .find("                except (aiohttp.ClientError, asyncio.TimeoutError) as error:")
        .unwrap();
    let processing = request
        .find("                    await self._handle_response_errors(response)")
        .unwrap();
    assert!(processing > catch);
    let boundary = request
        .find("                # Response processing is outside")
        .unwrap();
    assert!(boundary > catch && boundary < processing);
    assert!(!request[..boundary].contains("await self._handle_response_errors"));
    assert_eq!(request.matches("response.release()").count(), 2);
    assert!(request.contains("                finally:\n                    response.release()"));
    assert_eq!(
        request
            .matches("headers[\"X-Request-ID\"] = request_id")
            .count(),
        1
    );
}
#[test]
fn current_profile_read_only_admission_refuses_original_or_stale_remote_source() {
    let fixture = Fixture::new();
    let environment = fixture.root.join("prepared");
    let package = environment.join("lib/python3.11/site-packages/swerex");
    fs::create_dir_all(package.join("deployment")).unwrap();
    fs::create_dir_all(package.join("runtime")).unwrap();
    let modal =
        include_bytes!("../../../tests/fixtures/swerex_modal/official/deployment/modal.py.fixture");
    let config = include_bytes!(
        "../../../tests/fixtures/swerex_modal/official/deployment/config.py.fixture"
    );
    let remote =
        include_bytes!("../../../tests/fixtures/swerex_modal/official/runtime/remote.py.fixture");
    fs::write(
        package.join("deployment/modal.py"),
        transform::rewrite("deployment/modal.py", modal).unwrap(),
    )
    .unwrap();
    fs::write(package.join("deployment/config.py"), config).unwrap();
    let current = transform::rewrite("runtime/remote.py", remote).unwrap();
    fs::write(package.join("runtime/remote.py"), &current).unwrap();
    admit_patched(&environment).unwrap();
    for stale in [remote.as_slice(), b"earlier sealed unsafe V2 runtime"] {
        fs::write(package.join("runtime/remote.py"), stale).unwrap();
        assert!(admit_patched(&environment).is_err());
        assert_eq!(fs::read(package.join("runtime/remote.py")).unwrap(), stale);
        assert!(!package.join("runtime/remote.py.bak").exists());
    }
}

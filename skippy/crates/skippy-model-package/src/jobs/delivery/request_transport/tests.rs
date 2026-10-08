use super::*;
#[test]
fn mounted_locator_requires_bounded_exact_bytes_and_strict_readonly_mount_descendant() {
    let mount = ModelMount {
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
        mount_path: "/models/request".into(),
    };
    let bytes = b"{}";
    let mut locator = MountedRequest {
        schema_version: 1,
        path: "/models/request/input.json".into(),
        sha256: admission::digest(bytes),
        byte_size: 2,
        repo: mount.repo.clone(),
        revision: mount.revision.clone(),
    };
    assert!(locator.admit(bytes, std::slice::from_ref(&mount)).is_ok());
    locator.path = mount.mount_path.clone();
    assert!(locator.admit(bytes, std::slice::from_ref(&mount)).is_err());
    locator.path = "/models/request/../foreign".into();
    assert!(locator.admit(bytes, std::slice::from_ref(&mount)).is_err());
    locator.path = "/models/request/input.json".into();
    locator.byte_size = MAX_REQUEST_BYTES as u64 + 1;
    assert!(locator.admit(bytes, &[mount]).is_err());
}

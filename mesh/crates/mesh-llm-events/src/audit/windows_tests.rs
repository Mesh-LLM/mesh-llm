use super::*;
use std::os::windows::process::CommandExt;
use std::path::Path;

#[test]
fn refused_rotation_preserves_the_active_name() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("audit.log");
    std::fs::write(&path, b"first").unwrap();
    let held = std::fs::File::open(&path).unwrap();
    // Deny new sibling names while retaining delete access to the file itself.
    let change_directory_acl = |args: &[&str]| {
        let result = std::process::Command::new("icacls")
            .arg(directory.path())
            .args(args)
            .creation_flags(0x0800_0000)
            .output()
            .unwrap();
        assert!(result.status.success(), "{result:?}");
    };
    change_directory_acl(&["/deny", "*S-1-1-0:(WD)"]);
    let rotation = discard_rotated_file(&path);
    let contents = std::fs::read(&path);
    change_directory_acl(&["/remove:d", "*S-1-1-0"]);

    assert_eq!(
        rotation.unwrap_err().kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert_eq!(contents.unwrap(), b"first");
    discard_rotated_file(&path).unwrap();
    std::fs::write(&path, b"next").unwrap();
    drop(held);
    assert_eq!(std::fs::read(path).unwrap(), b"next");
}

#[test]
fn rotation_reuses_names_while_another_sink_holds_the_old_file() {
    for max_files in [1, 2] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("audit.log");
        let config = FileAuditSinkConfig {
            path: path.clone(),
            max_file_size: 1,
            max_files,
            ..FileAuditSinkConfig::default()
        };
        let idle = FileAuditSink::new(config.clone()).unwrap();
        let writer = FileAuditSink::new(config).unwrap();
        let emit = |sink: &FileAuditSink, action: &str| {
            sink.emit_audit(&AuditEvent::new(
                AuditCategory::System,
                action,
                AuditOutcome::Success,
            ))
            .unwrap();
        };
        emit(&writer, "first");
        let held = std::fs::File::open(&path).unwrap();
        emit(&writer, "second");
        emit(&writer, "third");
        emit(&idle, "fourth");

        let action = |file: std::fs::File| {
            serde_json::from_reader::<_, AuditEvent>(file)
                .unwrap()
                .action
        };
        assert_eq!(action(std::fs::File::open(&path).unwrap()), "fourth");
        if max_files == 2 {
            assert_eq!(
                action(std::fs::File::open(rotated_path(&path, 1)).unwrap()),
                "third"
            );
        }
        assert_eq!(action(held), "first");
        drop(idle);
        drop(writer);
        // The retained generations and the sibling lock are the only entries.
        assert_eq!(
            std::fs::read_dir(directory.path()).unwrap().count(),
            max_files + 1
        );
    }
}

fn grant_everyone_read(path: &Path, directory: bool) {
    let grant = if directory {
        "*S-1-1-0:(OI)(CI)R"
    } else {
        "*S-1-1-0:R"
    };
    let output = std::process::Command::new("icacls")
        .arg(path)
        .args(["/grant", grant])
        .creation_flags(0x0800_0000)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_acl(path, directory, false);
}

fn assert_acl(path: &Path, directory: bool, private: bool) {
    let script = r#"
$ErrorActionPreference = 'Stop'
$path = $env:MESH_LLM_TEST_ACL_PATH
if ($env:MESH_LLM_TEST_ACL_DIRECTORY -eq '1') {
    $acl = [System.IO.Directory]::GetAccessControl($path)
    $inheritance = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
} else {
    $acl = [System.IO.File]::GetAccessControl($path)
    $inheritance = [System.Security.AccessControl.InheritanceFlags]::None
}
$rules = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
if ($env:MESH_LLM_TEST_ACL_PRIVATE -eq '1') {
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    if (-not $acl.AreAccessRulesProtected -or $rules.Count -ne 1 -or
        $rules[0].IdentityReference -ne $sid -or
        $rules[0].AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow -or
        $rules[0].FileSystemRights -ne [System.Security.AccessControl.FileSystemRights]::FullControl -or
        $rules[0].InheritanceFlags -ne $inheritance) {
        throw ('Expected one protected current-user FullControl rule: ' + $acl.Sddl)
    }
} elseif (-not ($rules | Where-Object { $_.IdentityReference.Value -eq 'S-1-1-0' })) {
    throw 'Fixture did not grant Everyone access'
}
"#;
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ])
        .env("MESH_LLM_TEST_ACL_PATH", path)
        .env(
            "MESH_LLM_TEST_ACL_DIRECTORY",
            if directory { "1" } else { "0" },
        )
        .env("MESH_LLM_TEST_ACL_PRIVATE", if private { "1" } else { "0" })
        .creation_flags(0x0800_0000)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
}

#[test]
fn initialization_replaces_explicit_foreign_file_and_directory_grants() {
    let directory = tempfile::tempdir().unwrap();
    let private = directory.path().join("audit's [private] directory");
    std::fs::create_dir(&private).unwrap();
    let path = private.join("audit's [private].log");
    let lock_path = private.join(".audit's [private].log.lock");
    std::fs::write(&path, "").unwrap();
    std::fs::write(&lock_path, "").unwrap();
    grant_everyone_read(&private, true);
    grant_everyone_read(&path, false);
    grant_everyone_read(&lock_path, false);

    let sink = FileAuditSink::new(FileAuditSinkConfig {
        path: path.clone(),
        ..FileAuditSinkConfig::default()
    })
    .unwrap();

    assert_acl(&private, true, true);
    assert_acl(&path, false, true);
    assert_acl(&lock_path, false, true);
    sink.emit_audit(&AuditEvent::new(
        AuditCategory::System,
        "private",
        AuditOutcome::Success,
    ))
    .unwrap();
}

#[test]
fn replacement_removes_explicit_foreign_grants_before_append() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("audit's [private].log");
    let old_path = directory.path().join("external-rotation.log");
    let sink = FileAuditSink::new(FileAuditSinkConfig {
        path: path.clone(),
        ..FileAuditSinkConfig::default()
    })
    .unwrap();
    std::fs::rename(&path, &old_path).unwrap();
    std::fs::write(&path, "").unwrap();
    grant_everyone_read(&path, false);

    sink.emit_audit(&AuditEvent::new(
        AuditCategory::System,
        "private",
        AuditOutcome::Success,
    ))
    .unwrap();

    assert_acl(&path, false, true);
    assert!(std::fs::read_to_string(&path).unwrap().contains("private"));
    assert!(std::fs::read_to_string(old_path).unwrap().is_empty());
}

#[test]
fn private_permissions_reject_a_replaced_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("audit.log");
    let file = open_guarded_append_file(&path, true).unwrap();
    std::fs::rename(&path, directory.path().join("old.log")).unwrap();
    std::fs::write(&path, "").unwrap();
    let error = apply_private_file_permissions(&path, &file).unwrap_err();
    assert!(error.to_string().contains("audit path changed"));
}

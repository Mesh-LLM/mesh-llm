use super::*;
use std::os::windows::process::CommandExt;
use std::path::Path;

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

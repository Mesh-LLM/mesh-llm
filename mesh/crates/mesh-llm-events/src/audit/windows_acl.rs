use std::io;
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};

pub(super) fn apply_owner_only_acl(path: &Path, directory: bool) -> io::Result<()> {
    // Replace the complete DACL in one update, including explicit foreign grants.
    // Paths remain data and are never interpolated into this script.
    let script = r#"
$ErrorActionPreference = 'Stop'
$path = $env:MESH_LLM_AUDIT_ACL_PATH
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$directory = $env:MESH_LLM_AUDIT_ACL_DIRECTORY -eq '1'
if ($directory) {
    $acl = [System.Security.AccessControl.DirectorySecurity]::new()
    $inheritance = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
} else {
    $acl = [System.Security.AccessControl.FileSecurity]::new()
    $inheritance = [System.Security.AccessControl.InheritanceFlags]::None
}
$acl.SetAccessRuleProtection($true, $false)
$rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
    $sid, [System.Security.AccessControl.FileSystemRights]::FullControl,
    $inheritance, [System.Security.AccessControl.PropagationFlags]::None,
    [System.Security.AccessControl.AccessControlType]::Allow)
$acl.AddAccessRule($rule)
if ($directory) {
    [System.IO.Directory]::SetAccessControl($path, $acl)
} else {
    [System.IO.File]::SetAccessControl($path, $acl)
}
"#;
    let status = Command::new("powershell.exe")
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ])
        .env("MESH_LLM_AUDIT_ACL_PATH", path)
        .env(
            "MESH_LLM_AUDIT_ACL_DIRECTORY",
            if directory { "1" } else { "0" },
        )
        .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "failed to apply owner-only audit ACL",
        ))
    }
}
